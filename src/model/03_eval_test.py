# 03_eval_test.py
# Evaluate the saved logistic model on the TEST set using the validation-chosen threshold.

import os, joblib, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix

ID_COL, LABEL_COL = "usubjid", "label"

# --- load artifacts & data ---
test = pd.read_csv("data/test.csv")
feature_cols = [c for c in test.columns if c not in (ID_COL, LABEL_COL)]

clf = joblib.load("models/logreg.joblib")        # trained logistic model
scaler = joblib.load("models/scaler.joblib")     # standardizer used for training
with open("data/threshold_logreg.txt") as f:
    thr = float(f.read().strip())

Xte, yte = test[feature_cols].values, test[LABEL_COL].values
Xte_s = scaler.transform(Xte)

# --- predict probabilities and apply fixed threshold ---
p_te = clf.predict_proba(Xte_s)[:, 1]
yhat = (p_te >= thr).astype(int)

# --- metrics ---
auc = roc_auc_score(yte, p_te)
acc = accuracy_score(yte, yhat)
f1  = f1_score(yte, yhat)
tn, fp, fn, tp = confusion_matrix(yte, yhat).ravel()
spec = tn / (tn + fp) if (tn + fp) else 0
sens = tp / (tp + fn) if (tp + fn) else 0

print(f"[TEST] AUC={auc:.3f}  Acc={acc:.3f}  F1={f1:.3f}  Spec={spec:.3f}  Sens={sens:.3f}  Thr={thr:.3f}")
print(f"[TEST] Confusion matrix: TP={tp}  FP={fp}  TN={tn}  FN={fn}")

# --- save outputs ---
pd.DataFrame({"usubjid": test[ID_COL], "y_true": yte, "y_prob": p_te, "y_pred": yhat}) \
  .to_csv("data/test_preds_logreg.csv", index=False)

with open("data/test_metrics_logreg.txt", "w") as f:
    f.write(f"AUC: {auc:.3f}\nAccuracy: {acc:.3f}\nF1: {f1:.3f}\nSpecificity: {spec:.3f}\n"
            f"Sensitivity: {sens:.3f}\nThreshold: {thr:.3f}\nTP={tp}, FP={fp}, TN={tn}, FN={fn}\n")
print("[TEST] Evaluation complete. Predictions and metrics saved.")