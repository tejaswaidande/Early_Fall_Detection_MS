# 05_logreg_feature_importance.py
# Show which features increase/decrease fall risk in the Logistic Regression model.

import os, joblib, pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ID_COL, LABEL_COL = "usubjid", "label"

# 1) Load the trained model + scaler and the training data (for feature names)
clf = joblib.load("models/logreg.joblib")      # from 02_train_logreg.py
scaler = joblib.load("models/scaler.joblib")
train = pd.read_csv("data/train.csv")

# 2) Get the feature columns (everything except id + label)
feature_cols = [c for c in train.columns if c not in (ID_COL, LABEL_COL)]

# 3) Coefficients: positive => higher risk | negative => lower risk
coefs = clf.coef_.ravel()
imp = pd.DataFrame({"feature": feature_cols, "coef": coefs})
imp["abs_coef"] = imp["coef"].abs()
imp = imp.sort_values("abs_coef", ascending=False)

# 4) Save a CSV for your report
os.makedirs("data", exist_ok=True)
imp.to_csv("data/logreg_feature_importance.csv", index=False)
print(imp.head(10))

# 5) Quick bar chart (top 12 by absolute size)
top = imp.head(12).sort_values("coef")
plt.figure(figsize=(7, 5))
plt.barh(top["feature"], top["coef"])
plt.axvline(0, linestyle="--")
plt.title("Logistic Regression: feature effects on fall risk")
plt.xlabel("Coefficient (positive = higher risk)")
plt.tight_layout()
plt.savefig("data/logreg_feature_importance.png", dpi=200)
print("[done] Wrote data/logreg_feature_importance.csv and .png")
