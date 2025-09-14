# src/feature_cleaning_keep_id.py
import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME   = os.getenv("DB_NAME", "thesis_data")
ID_COL = "usubjid"

print("[info] Connecting to Mongo…")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# 1) Load from cleaned_features (this should still have usubjid)
print("[info] Loading cleaned_features…")
df = pd.DataFrame(list(db["cleaned_features"].find({})))
if df.empty:
    raise SystemExit("[error] cleaned_features is empty.")
if "_id" in df.columns:
    df.drop(columns=["_id"], inplace=True)

if ID_COL not in df.columns:
    raise SystemExit("[error] 'usubjid' is not present in cleaned_features. Stop here and ping me.")

# Keep ID as string, never touch it during numeric ops
df[ID_COL] = df[ID_COL].astype(str)

# 2) Text→numeric mapping for questionnaire-style fields (same idea you used before)
text_mapping = {
    "None of the time": 0, "A little of the time": 1, "Some of the time": 2,
    "Moderate amount of the time": 2, "Most of the time": 3, "All of the time": 4,
    "Mostly true": 4, "Mostly false": 0, "Never": 0, "Rarely": 1,
    "Sometimes": 2, "Often": 3, "Always": 4
}

for col in df.columns:
    if col == ID_COL: 
        continue
    if df[col].dtype == object:
        df[col] = df[col].replace(text_mapping)
        # second pass: turn numeric-looking strings into numbers
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 3) Drop columns with >50% missing (but NEVER drop the ID)
missing_pct = df.drop(columns=[ID_COL]).isna().mean() * 100
to_drop = missing_pct[missing_pct > 50].index.tolist()
print(f"[info] Dropping {len(to_drop)} high-missing columns (>50%): {to_drop[:10]}{' …' if len(to_drop)>10 else ''}")
df.drop(columns=to_drop, inplace=True)

# 4) Impute numeric columns with median (skip ID)
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != ID_COL]
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

print("[ok] Final feature shape (with ID preserved):", df.shape)
print(df.head(3)[[ID_COL] + [c for c in df.columns if c != ID_COL][:5]])

# 5) Write to Mongo as a new collection
out_col = "final_features_keep_id"
print(f"[info] Replacing collection '{out_col}' …")
db[out_col].drop()
db[out_col].insert_many(df.to_dict("records"))
print("[done] Wrote final_features_keep_id.")
client.close()
