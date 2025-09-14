# 06_validate_importance_and_bins.py
# What this does:
#   (A) Permutation importance for the saved Logistic Regression on VALIDATION (scoring = ROC-AUC)
#   (B) Fall-rate by value bins for EDSS + key KFSS domains (deciles if possible)
#
# Outputs:
#   data/permutation_importance_logreg.csv
#   data/permutation_importance_logreg.png
#   data/bin_rates_EDSS.csv (and PNG)
#   data/bin_rates_<KFSS>.csv (and PNG for each)

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

# ---------------- config ----------------
ID_COL, LABEL_COL = "usubjid", "label"
VALID_PATH = "data/valid.csv"
MODEL_PATH = "models/logreg.joblib"
SCALER_PATH = "models/scaler.joblib"
OUT_DIR = "data"

# features to make bin-wise fall-rate plots for:
BIN_FEATURES = [
    "EDSS",
    "KFSS_Cerebellar",
    "KFSS_Pyramidal",
    "KFSS_BowelBladder",
    "KFSS_Sensory",
]
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

print("[info] Loading validation data …")
valid = pd.read_csv(VALID_PATH)
if ID_COL not in valid.columns or LABEL_COL not in valid.columns:
    raise SystemExit("[error] valid.csv must contain 'usubjid' and 'label' columns.")
feature_cols = [c for c in valid.columns if c not in (ID_COL, LABEL_COL)]

X_val = valid[feature_cols].values
y_val = valid[LABEL_COL].values

print("[info] Loading model + scaler …")
clf = joblib.load(MODEL_PATH)       # LogisticRegression inside CalibratedClassifier? (ours is plain LR)
scaler = joblib.load(SCALER_PATH)

# Scale inputs the same way as during training
X_val_s = scaler.transform(X_val)

# --- (A) Permutation importance ---
print("[info] Computing permutation importance on validation (scoring = roc_auc) …")
# sklearn will use clf.predict_proba internally for roc_auc
pi = permutation_importance(
    estimator=clf,
    X=X_val_s,
    y=y_val,
    scoring="roc_auc",
    n_repeats=50,
    random_state=42,
    n_jobs=None,  # single-thread for reproducibility
)

imp_df = pd.DataFrame({
    "feature": feature_cols,
    "importance_mean": pi.importances_mean,
    "importance_std": pi.importances_std
}).sort_values("importance_mean", ascending=False)

imp_csv = os.path.join(OUT_DIR, "permutation_importance_logreg.csv")
imp_df.to_csv(imp_csv, index=False)
print(f"[ok] Wrote permutation importance to {imp_csv}")
print(imp_df.head(10))

# quick bar plot of top 12
top = imp_df.head(12).iloc[::-1]  # reverse for horizontal plot
plt.figure(figsize=(7, 5))
plt.barh(top["feature"], top["importance_mean"])
plt.title("Permutation importance (LogReg, VALID, ROC-AUC)")
plt.xlabel("Mean importance (AUC drop when shuffled)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "permutation_importance_logreg.png"), dpi=200)
plt.close()
print("[ok] Saved permutation importance plot.")

# --- (B) Fall-rate by bins (deciles if unique enough) ---
def make_bin_rates(df, feature_name, out_dir=OUT_DIR):
    if feature_name not in df.columns:
        print(f"[warn] {feature_name} not in validation; skipping.")
        return

    x = df[feature_name].copy()
    y = df[LABEL_COL].copy()

    # robust binning: try deciles; if too few unique values, fall back to quantiles with fewer bins or cut
    try_bins = [10, 8, 5, 4]
    binned = None
    for nb in try_bins:
        try:
            binned = pd.qcut(x, q=nb, duplicates="drop")
            # ensure we got at least 3 bins
            if binned.nunique() >= 3:
                break
        except Exception:
            continue
    if binned is None or binned.nunique() < 3:
        # fallback: simple fixed-width bins into 4 parts
        binned = pd.cut(x, bins=4)

    # compute fall rate in each bin
    grp = pd.DataFrame({feature_name: x, LABEL_COL: y, "bin": binned})
    summary = grp.groupby("bin").agg(
        count=(LABEL_COL, "size"),
        fallers=(LABEL_COL, "sum"),
        fall_rate=(LABEL_COL, "mean")
    ).reset_index()

    # save CSV
    out_csv = os.path.join(out_dir, f"bin_rates_{feature_name.replace('/', '_')}.csv")
    summary.to_csv(out_csv, index=False)
    print(f"[ok] Wrote bin rates for {feature_name} -> {out_csv}")
    print(summary)

    # simple plot
    plt.figure(figsize=(7, 4))
    # to keep bin order readable as string
    labels = summary["bin"].astype(str)
    plt.plot(range(len(labels)), summary["fall_rate"], marker="o")
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.ylabel("Fall rate")
    plt.title(f"Fall rate by {feature_name} bins (VALID)")
    plt.tight_layout()
    out_png = os.path.join(out_dir, f"bin_rates_{feature_name.replace('/', '_')}.png")
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[ok] Saved plot for {feature_name}.")

print("[info] Computing fall-rate by bins for key features …")
for fx in BIN_FEATURES:
    make_bin_rates(valid, fx)

print("[done] Importance + bin analyses complete.")
