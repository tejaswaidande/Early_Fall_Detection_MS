# Purpose: Create patient-level fall labels from Medical History (MH)

import os, sys, re
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME   = os.getenv("DB_NAME", "thesis_data")

print(f"[info] Connecting to MongoDB at {MONGO_URI} ...")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
print(f"[ok] Connected. Using DB = {DB_NAME}")

# We’ll take the patient list from the features table that keeps IDs
FEATURES_COLLECTION = "final_selected_features_keep_id"
ID_COL = "usubjid"

# Falls text sources (Medical History)
FALLS_COLLECTION = "mh"
FALLS_TEXT_FIELDS = ["mhterm", "mhdecod"]   # search both if present

# Word-boundary pattern to avoid false positives like "ALMOTRIPTAN" or "felling"
# Matches: fall, falls, falling, fell, slip/slipped/slips, trip/tripped/trips
FALL_REGEX = r"\bfall(s|ing)?\b|\bfell\b|\bslip(ped|s)?\b|\btrip(ped|s)?\b"

print(f"[info] Fetching patient IDs from '{FEATURES_COLLECTION}' ...")
feat = list(db[FEATURES_COLLECTION].find({}, {ID_COL:1, "_id":0}))
if not feat:
    print(f"[error] No patients found in '{FEATURES_COLLECTION}'.")
    sys.exit(1)

patients_df = pd.DataFrame(feat).drop_duplicates()
if ID_COL not in patients_df.columns:
    print(f"[error] '{ID_COL}' not found in documents from '{FEATURES_COLLECTION}'.")
    sys.exit(1)

patients_df[ID_COL] = patients_df[ID_COL].astype(str)
print(f"[ok] Found {len(patients_df)} unique patients.")
print(patients_df.head(3))

# Build OR query across available MH text fields
mh = db[FALLS_COLLECTION]
mh_fields_present = [f for f in FALLS_TEXT_FIELDS if mh.find_one({}, {f:1, "_id":0}) and True]
if not mh_fields_present:
    print("[warn] MH has none of the expected text fields; labeling everyone 0.")
    fall_usubjids = set()
else:
    or_clauses = [{f: {"$regex": FALL_REGEX, "$options": "i"}} for f in mh_fields_present]
    query = {"$or": or_clauses}
    print(f"[info] Searching MH fields {mh_fields_present} with regex: {FALL_REGEX!r}")
    # Pull matches (minor inefficiency ok)
    matches = list(mh.find(query, {ID_COL:1, **{f:1 for f in mh_fields_present}, "_id":0}))
    if not matches:
        print("[warn] No fall-like mentions found in MH; labeling everyone 0.")
        fall_usubjids = set()
    else:
        mdf = pd.DataFrame(matches).dropna(subset=[ID_COL]).drop_duplicates()
        # Convert to string IDs for safe matching
        mdf[ID_COL] = mdf[ID_COL].astype(str)
        fall_usubjids = set(mdf[ID_COL].unique())
        # Show a couple of examples for audit
        print(f"[ok] MH matches: {len(mdf)} rows across {len(fall_usubjids)} patients.")
        print(mdf.head(5))

# Create labels (1 = has any fall-like MH mention, 0 = none)
patients_df["label"] = patients_df[ID_COL].apply(lambda x: 1 if x in fall_usubjids else 0)

# Summary
total = len(patients_df)
positives = int(patients_df["label"].sum())
print(f"[summary] Total patients: {total} | Falls=1: {positives} | Non-falls=0: {total - positives}")

# Save
os.makedirs("data", exist_ok=True)
out_path = os.path.join("data", "labels.csv")
patients_df[[ID_COL, "label"]].to_csv(out_path, index=False)
print(f"[ok] Saved labels to: {out_path}")

client.close()
print("[done] Label generation from MH complete.")
