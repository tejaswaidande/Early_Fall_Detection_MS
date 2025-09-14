# 04_make_results_summary.py
# Collects metrics for Logistic Regression and HGB on VALID and TEST into one CSV.

import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix

ID, Y = "usubjid", "y_true"

def load_preds(path_csv, thr=None):
    df = pd.read_csv(path_csv)
    p = df["y_prob"].values
    y = df["y_true"].values
    if thr is None:
        # default 0.5 if no threshold provided
        thr = 0.5
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return {
        "auc": roc_auc_score(y, p),
        "acc": accuracy_score(y, yhat),
        "f1": f1_score(y, yhat),
        "spec": tn / (tn + fp) if (tn + fp) else 0,
        "sens": tp / (tp + fn) if (tp + fn) else 0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "thr": thr
    }

def read_thr(path_txt, fallback=0.5):
    try:
        with open(path_txt) as f:
            return float(f.read().strip())
    except:
        return fallback

rows = []

# Logistic — VALID
thr_lr = read_thr("data/threshold_logreg.txt", 0.5)
rows.append({"model":"logreg","split":"valid", **load_preds("data/valid_preds_logreg.csv", thr_lr)})

# Logistic — TEST
# (03_eval_test.py already produced test_preds_logreg.csv)
rows.append({"model":"logreg","split":"test", **load_preds("data/test_preds_logreg.csv", thr_lr)})

# HGB — VALID
thr_hgb = read_thr("data/threshold_hgb.txt", 0.5)
rows.append({"model":"hgb","split":"valid", **load_preds("data/valid_preds_hgb.csv", thr_hgb)})

# HGB — TEST
rows.append({"model":"hgb","split":"test", **load_preds("data/test_preds_hgb.csv", thr_hgb)})

out = pd.DataFrame(rows)
os.makedirs("data", exist_ok=True)
out.to_csv("data/results_summary.csv", index=False)
print(out)
print("\n[done] Wrote data/results_summary.csv")
