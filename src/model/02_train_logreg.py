# 02_train_logreg.py
# Train a simple Logistic Regression baseline and evaluate on VALIDATION.
# Now also saves a ROC curve plot for the validation set.
#
# Outputs:
#   models/logreg.joblib
#   models/scaler.joblib
#   data/valid_preds_logreg.csv
#   data/valid_metrics_logreg.txt
#   data/threshold_logreg.txt
#   data/roc_curve_logreg_valid.png

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
)

SEED = 42
ID_COL, LABEL_COL = "usubjid", "label"

def main():
    # --- load data ---
    print("[info] loading train/valid CSVs …")
    train = pd.read_csv("data/train.csv")
    valid = pd.read_csv("data/valid.csv")

    # everything except id + label are features
    feature_cols = [c for c in train.columns if c not in (ID_COL, LABEL_COL)]
    Xtr, ytr = train[feature_cols].values, train[LABEL_COL].values
    Xva, yva = valid[feature_cols].values, valid[LABEL_COL].values

    # quick sanity
    print(f"[debug] n_train={len(train)}  n_valid={len(valid)}  n_features={len(feature_cols)}")

    # --- scale features (logistic prefers standardized inputs) ---
    print("[info] fitting scaler …")
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)

    # --- train logistic regression (handle imbalance) ---
    print("[info] training logistic regression …")
    clf = LogisticRegression(
        penalty="elasticnet",
        l1_ratio=0.5,
        solver="saga",
        class_weight="balanced",   # important due to 1.9% positives
        max_iter=5000,
        random_state=SEED,
        n_jobs=None,               # keep deterministic
    )
    clf.fit(Xtr_s, ytr)

    # --- predict probabilities on validation ---
    print("[info] scoring on validation …")
    p_va = clf.predict_proba(Xva_s)[:, 1]

    # --- pick a threshold: aim for ~90% specificity on VALIDATION ---
    print("[info] selecting threshold targeting ~90% specificity …")
    best_thr, best_sens = 0.5, 0.0
    tn = fp = fn = tp = 0

    for t in np.linspace(0, 1, 501):
        yhat = (p_va >= t).astype(int)
        tn_, fp_, fn_, tp_ = confusion_matrix(yva, yhat).ravel()
        spec = tn_ / (tn_ + fp_) if (tn_ + fp_) else 0
        sens = tp_ / (tp_ + fn_) if (tp_ + fn_) else 0
        # keep the highest sensitivity among thresholds that meet specificity >= 0.90
        if spec >= 0.90 and sens > best_sens:
            best_thr, best_sens = t, sens
            tn, fp, fn, tp = tn_, fp_, fn_, tp_

    # fallback if nothing met 0.90 specificity
    if best_sens == 0.0 and (tn + fp) == 0:
        yhat = (p_va >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(yva, yhat).ravel()
        best_thr = 0.5
        print("[warn] could not reach 0.90 specificity; using threshold=0.5")

    # --- compute metrics at chosen threshold ---
    yhat = (p_va >= best_thr).astype(int)
    auc = roc_auc_score(yva, p_va)
    acc = accuracy_score(yva, yhat)
    f1  = f1_score(yva, yhat)
    spec = tn / (tn + fp) if (tn + fp) else 0
    sens = tp / (tp + fn) if (tp + fn) else 0

    print(f"[done] AUC={auc:.3f}  Acc={acc:.3f}  F1={f1:.3f}  Spec={spec:.3f}  Sens={sens:.3f}  Thr={best_thr:.3f}")

    # --- save artifacts ---
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    joblib.dump(clf, "models/logreg.joblib")
    joblib.dump(scaler, "models/scaler.joblib")

    pd.DataFrame({
        "usubjid": valid[ID_COL],
        "y_true":  yva,
        "y_prob":  p_va,
        "y_pred":  yhat
    }).to_csv("data/valid_preds_logreg.csv", index=False)

    with open("data/valid_metrics_logreg.txt", "w") as f:
        f.write(
            "ROC-AUC: {:.3f}\nAccuracy: {:.3f}\nF1: {:.3f}\nSpecificity: {:.3f}\nSensitivity: {:.3f}\n"
            "Threshold: {:.3f}\nTP={:,}, FP={:,}, TN={:,}, FN={:,}\n".format(
                auc, acc, f1, spec, sens, best_thr, tp, fp, tn, fn
            )
        )

    with open("data/threshold_logreg.txt", "w") as f:
        f.write(str(best_thr))

    # --- ROC curve (VALIDATION) ---
    print("[info] creating ROC curve plot (validation) …")
    fpr, tpr, thr = roc_curve(yva, p_va)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    # mark the chosen operating point
    # find the closest threshold index to best_thr (for marker)
    idx = (np.abs(thr - best_thr)).argmin() if len(thr) else None
    if idx is not None and 0 <= idx < len(tpr):
        plt.scatter(fpr[idx], tpr[idx], s=40, label=f"Chosen thr={best_thr:.3f}", zorder=3)

    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("Logistic Regression — ROC (Validation)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_png = "data/roc_curve_logreg_valid.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[ok] saved ROC plot to {out_png}")

    print("Saved: models/logreg.joblib, models/scaler.joblib, data/valid_preds_logreg.csv, "
          "data/valid_metrics_logreg.txt, data/threshold_logreg.txt, data/roc_curve_logreg_valid.png")

if __name__ == "__main__":
    main()
