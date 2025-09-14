# 02b_train_hgb.py
# Train Histogram Gradient Boosting (tree-based) and pick a threshold on VALIDATION.

import os, joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

SEED = 42
ID_COL, LABEL_COL = "usubjid", "label"

# Load data
train = pd.read_csv("data/train.csv")
valid = pd.read_csv("data/valid.csv")
feature_cols = [c for c in train.columns if c not in (ID_COL, LABEL_COL)]

Xtr, ytr = train[feature_cols].values, train[LABEL_COL].values
Xva, yva = valid[feature_cols].values, valid[LABEL_COL].values

# Class imbalance: give positives more weight
wtr = compute_sample_weight(class_weight="balanced", y=ytr)

# Model (no scaling needed)
hgb = HistGradientBoostingClassifier(
    learning_rate=0.1, max_iter=500, random_state=SEED
)
hgb.fit(Xtr, ytr, sample_weight=wtr)

# Validation probabilities
p_va = hgb.predict_proba(Xva)[:, 1]

# Pick threshold targeting ~90% specificity on VALIDATION
best_thr, best_sens = 0.5, 0.0
tn=fp=fn=tp=0
for t in np.linspace(0, 1, 501):
    yhat = (p_va >= t).astype(int)
    tn_, fp_, fn_, tp_ = confusion_matrix(yva, yhat).ravel()
    spec = tn_ / (tn_ + fp_) if (tn_ + fp_) else 0
    sens = tp_ / (tp_ + fn_) if (tp_ + fn_) else 0
    if spec >= 0.90 and sens > best_sens:
        best_thr, best_sens = t, sens
        tn, fp, fn, tp = tn_, fp_, fn_, tp_

# Fallback to 0.5 if nothing met 0.90 specificity
if best_sens == 0.0 and (tn+fp) == 0:
    yhat = (p_va >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yva, yhat).ravel()
    best_thr = 0.5

auc = roc_auc_score(yva, p_va)
acc = accuracy_score(yva, (p_va >= best_thr).astype(int))
f1  = f1_score(yva, (p_va >= best_thr).astype(int))
spec = tn / (tn + fp) if (tn + fp) else 0
sens = tp / (tp + fn) if (tp + fn) else 0

# Save artifacts
os.makedirs("models", exist_ok=True)
joblib.dump(hgb, "models/hgb.joblib")
pd.DataFrame({"usubjid": valid[ID_COL], "y_true": yva, "y_prob": p_va}) \
  .to_csv("data/valid_preds_hgb.csv", index=False)
with open("data/threshold_hgb.txt", "w") as f:
    f.write(str(best_thr))

print(f"[HGB|VALID] AUC={auc:.3f}  Acc={acc:.3f}  F1={f1:.3f}  Spec={spec:.3f}  Sens={sens:.3f}  Thr={best_thr:.3f}")
print("Saved: models/hgb.joblib, data/valid_preds_hgb.csv, data/threshold_hgb.txt")
