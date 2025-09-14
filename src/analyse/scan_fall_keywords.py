# src/analyse/scan_fall_keywords.py
# Purpose: scan ALL MongoDB collections to find where "falls" are recorded.
# Output: data/fall_keyword_scan.csv (collection, field, total_docs, match_count, distinct_usubjid, example_usubjid, example_text)

import os
import re
import sys
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

# -------------------- config --------------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME   = os.getenv("DB_NAME", "thesis_data")

# keywords to search for (simple heuristic)
FALL_KEYWORDS = ["fall", "fell", "falls", "falling", "slip", "slipped", "trip", "tripped"]
REGEX_PATTERN = "|".join(FALL_KEYWORDS)  # 'fall|fell|...'
REGEX = {"$regex": REGEX_PATTERN, "$options": "i"}  # case-insensitive

# how many docs to sample per collection to infer "text-like" fields
SAMPLE_DOCS_FOR_SCHEMA = 300

# in many SDTM-style datasets this is the patient id field
USUBJID_CANDIDATES = ["usubjid", "USUBJID", "subject_id", "SubjectId"]

# ------------------------------------------------

def connect_db():
    print(f"[info] connecting to MongoDB at {MONGO_URI} ...")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    print(f"[ok] connected. using DB = {DB_NAME}")
    return client, db

def infer_text_fields(coll, sample_n=SAMPLE_DOCS_FOR_SCHEMA):
    """
    Sample a few documents and find top-level fields that look string-like.
    Minor inefficiency: we just scan a small sample each time.
    """
    text_fields = set()
    cursor = coll.find({}, {"_id": 0}).limit(sample_n)
    count = 0
    for doc in cursor:
        count += 1
        for k, v in doc.items():
            # only consider top-level fields
            if isinstance(v, str):
                text_fields.add(k)
            # handle list of strings (sometimes terms are lists)
            elif isinstance(v, list) and v and all(isinstance(x, str) for x in v[:5]):
                text_fields.add(k)
            # (skip dict/nested for simplicity)
    # small debug print
    print(f"[debug] inferred {len(text_fields)} text-like fields from {count} sampled docs.")
    return sorted(text_fields)

def pick_usubjid_field(coll):
    """
    Try to find which field name is the patient id.
    Just check the candidates and return the first that exists in a sample doc.
    """
    doc = coll.find_one({}, {"_id":0})
    if not doc:
        return None
    for name in USUBJID_CANDIDATES:
        if name in doc:
            return name
    # fallback: case-insensitive search for 'usubjid'
    for k in doc.keys():
        if k.lower() == "usubjid":
            return k
    return None

def main():
    client, db = connect_db()

    try:
        os.makedirs("data", exist_ok=True)
        results = []

        collections = db.list_collection_names()
        print(f"[info] found {len(collections)} collections.")
        print(collections)

        for coll_name in collections:
            print(f"\n[scan] collection: {coll_name}")
            coll = db[coll_name]

            # total docs
            total_docs = coll.count_documents({})
            print(f"[info] total docs: {total_docs}")

            if total_docs == 0:
                print("[warn] empty collection, skipping.")
                continue

            # infer candidate text fields
            text_fields = infer_text_fields(coll)
            if not text_fields:
                print("[warn] no obvious text fields found, skipping.")
                continue

            # identify patient id field (if exists)
            usubjid_field = pick_usubjid_field(coll)
            if usubjid_field:
                print(f"[info] detected patient id field: {usubjid_field}")
            else:
                print("[warn] no obvious patient id field in this collection.")

            # test each text field for fall keyword matches
            for field in text_fields:
                try:
                    q = {field: REGEX}
                    match_count = coll.count_documents(q)

                    if match_count > 0:
                        # get example doc
                        projection = {"_id":0, field:1}
                        if usubjid_field:
                            projection[usubjid_field] = 1

                        example = coll.find_one(q, projection)
                        example_usubjid = example.get(usubjid_field) if (example and usubjid_field) else None
                        example_text = example.get(field) if example else None
                        # shorten long text for display
                        if isinstance(example_text, str) and len(example_text) > 120:
                            example_text = example_text[:117] + "..."

                        # count distinct patients if possible (minor inefficiency on big collections)
                        distinct_patients = None
                        if usubjid_field:
                            try:
                                distinct_list = coll.distinct(usubjid_field, q)
                                distinct_patients = len(distinct_list)
                            except Exception as e:
                                print(f"[warn] distinct usubjid failed for {coll_name}.{field}: {e}")

                        results.append({
                            "collection": coll_name,
                            "field": field,
                            "total_docs": total_docs,
                            "match_count": match_count,
                            "distinct_usubjid": distinct_patients,
                            "example_usubjid": example_usubjid,
                            "example_text": example_text
                        })
                        print(f"[hit] {coll_name}.{field}: matches={match_count}, distinct_usubjid={distinct_patients}, example={example_usubjid} | {example_text}")
                    else:
                        # uncomment for verbose: print(f"[nohit] {coll_name}.{field}")
                        pass

                except Exception as e:
                    print(f"[error] failed scanning {coll_name}.{field}: {e}")

        # build report
        if results:
            df = pd.DataFrame(results)
            # sort by match_count desc, then distinct_usubjid desc
            df = df.sort_values(by=["match_count", "distinct_usubjid"], ascending=[False, False])
            out_path = os.path.join("data", "fall_keyword_scan.csv")
            df.to_csv(out_path, index=False)
            print(f"\n[ok] wrote report to {out_path}")
            print(df.head(10))
        else:
            print("\n[info] No matches for fall-like keywords were found in any scanned fields.")

    finally:
        client.close()
        print("[done] scan complete.")

if __name__ == "__main__":
    main()
