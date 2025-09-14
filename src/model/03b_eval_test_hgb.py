# 03b_eval_test_hgb.py
# Evaluate saved HGB model on TEST using the validation-chosen threshold.

import os, joblib, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix

ID_COL, LABEL_COL = "usubjid", "label"

test = pd.read_csv("data/test.csv")
feature_cols = [c for c in test.columns if c not in (ID_COL, LABEL_COL)]

hgb = joblib.load("models/hgb.joblib")
with open("data/threshold_hgb.txt") as f:
    thr = float(f.read().strip())

p_te = hgb.predict_proba(test[feature_cols].values)[:, 1]
yte  = test[LABEL_COL].values
yhat = (p_te >= thr).astype(int)

auc = roc_auc_score(yte, p_te)
acc = accuracy_score(yte, yhat)
f1  = f1_score(yte, yhat)
tn, fp, fn, tp = confusion_matrix(yte, yhat).ravel()
spec = tn / (tn + fp) if (tn + fp) else 0
sens = tp / (tp + fn) if (tp + fn) else 0

print(f"[HGB|TEST] AUC={auc:.3f}  Acc={acc:.3f}  F1={f1:.3f}  Spec={spec:.3f}  Sens={sens:.3f}  Thr={thr:.3f}")
print(f"[HGB|TEST] Confusion matrix: TP={tp}  FP={fp}  TN={tn}  FN={fn}")

pd.DataFrame({"usubjid": test[ID_COL], "y_true": yte, "y_prob": p_te, "y_pred": yhat}) \
  .to_csv("data/test_preds_hgb.csv", index=False)
