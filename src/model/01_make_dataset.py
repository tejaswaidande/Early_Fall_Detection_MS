# 01_make_dataset.py
# Merge features + labels, make stratified train/valid/test splits (70/15/15)

import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME   = os.getenv("DB_NAME", "thesis_data")
FEATURES_COLLECTION = "final_selected_features_keep_id"
ID_COL, LABEL_COL = "usubjid", "label"
SEED = 42

print("[info] connecting to mongo…")
db = MongoClient(MONGO_URI)[DB_NAME]

print(f"[info] loading features from '{FEATURES_COLLECTION}' …")
feat = pd.DataFrame(list(db[FEATURES_COLLECTION].find({}, {"_id":0})))
if feat.empty:
    raise SystemExit("[error] no features found.")
feat[ID_COL] = feat[ID_COL].astype(str)

print("[info] loading labels.csv …")
labels = pd.read_csv("data/labels.csv")
labels[ID_COL] = labels[ID_COL].astype(str)

print("[info] merging on usubjid …")
df = feat.merge(labels, on=ID_COL, how="inner")
print("[ok] merged shape:", df.shape)

# quick sanity check: class balance
print("[info] label counts:\n", df[LABEL_COL].value_counts(dropna=False))

# split: first test (15%), then valid (15% of full -> 17.65% of remaining)
train_valid, test = train_test_split(
    df, test_size=0.15, stratify=df[LABEL_COL], random_state=SEED
)
valid_size = 0.15 / (1 - 0.15)  # ≈0.1765
train, valid = train_test_split(
    train_valid, test_size=valid_size, stratify=train_valid[LABEL_COL], random_state=SEED
)

# show split balances
for name, part in [("train", train), ("valid", valid), ("test", test)]:
    counts = part[LABEL_COL].value_counts()
    print(f"[ok] {name} size={len(part)} | label counts:\n{counts}")

os.makedirs("data", exist_ok=True)
train.to_csv("data/train.csv", index=False)
valid.to_csv("data/valid.csv", index=False)
test.to_csv("data/test.csv", index=False)

# also save the feature list (everything except id + label)
feature_cols = [c for c in df.columns if c not in (ID_COL, LABEL_COL)]
with open("data/feature_list.txt", "w") as f:
    for c in feature_cols: f.write(c + "\n")

print("[done] wrote data/train.csv, data/valid.csv, data/test.csv and data/feature_list.txt")
