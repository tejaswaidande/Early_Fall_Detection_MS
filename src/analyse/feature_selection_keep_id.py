# src/feature_selection_keep_id.py
import numpy as np
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME   = os.getenv("DB_NAME", "thesis_data")
ID_COL = "usubjid"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

print("[info] Loading final_features_keep_id …")
df = pd.DataFrame(list(db["final_features_keep_id"].find({})))
if df.empty:
    raise SystemExit("[error] final_features_keep_id is empty.")
if "_id" in df.columns:
    df.drop(columns=["_id"], inplace=True)

# numeric columns only (skip the ID)
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != ID_COL]

# 1) drop zero-variance
zero_var = [c for c in num_cols if df[c].nunique() <= 1]
print(f"[info] Zero-variance columns to drop: {zero_var}")
df.drop(columns=zero_var, inplace=True)
num_cols = [c for c in num_cols if c not in zero_var]

# 2) drop highly correlated (|r| > 0.90)
print("[info] Correlation filtering on numeric features …")
corr = df[num_cols].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if (upper[col] > 0.90).any()]
print(f"[info] Highly correlated columns to drop: {to_drop}")
df.drop(columns=to_drop, inplace=True)

print("[ok] Selected feature shape:", df.shape)
print(df.head(3)[[ID_COL] + [c for c in df.columns if c != ID_COL][:5]])

# save back to Mongo
out_col = "final_selected_features_keep_id"
print(f"[info] Replacing collection '{out_col}' …")
db[out_col].drop()
db[out_col].insert_many(df.to_dict("records"))
print("[done] Wrote final_selected_features_keep_id.")
client.close()
