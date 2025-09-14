import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB connection details
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "thesis_data")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

print("Connected to MongoDB")

# --------------------------------------
# QS extractor - main functional tests
# --------------------------------------
def extract_qs_data():
    print("Extracting QS data...")
    qs_collection = db["qs"]

    qs_cursor = qs_collection.find({}, {"usubjid": 1, "qstest": 1, "qsorres": 1})
    qs_records = list(qs_cursor)
    
    if not qs_records:
        print("No records found in QS collection!")
        return pd.DataFrame()
    
    qs_df = pd.DataFrame(qs_records)
    
    # Pivot to one row per subject
    qs_pivot = qs_df.pivot_table(
        index="usubjid",
        columns="qstest",
        values="qsorres",
        aggfunc="first"
    ).reset_index()

    # Rename columns to clean feature names
    qs_pivot = qs_pivot.rename(columns={
        "EDSS01-Expanded Disability Score": "EDSS",
        "KFSS1-Pyramidal Functions": "KFSS_Pyramidal",
        "KFSS1-Sensory Functions": "KFSS_Sensory",
        "KFSS1-Cerebellar Functions": "KFSS_Cerebellar",
        "KFSS1-Brain Stem Functions": "KFSS_BrainStem",
        "KFSS1-Visual Functions": "KFSS_Visual",
        "KFSS1-Bowel and Bladder Functions": "KFSS_BowelBladder",
        "KFSS1-Cerebral Functions": "KFSS_Cerebral",
        "KFSS1-Other": "KFSS_Other",
        "Timed 25 Foot Walk": "T25FW",
        "Nine Hole Peg Test": "NineHPT",
        "PASAT": "PASAT",
        "SDMT": "SDMT"
    })

    print(f"Extracted QS data with shape: {qs_pivot.shape}")
    return qs_pivot

# --------------------------------------
# DM extractor - demographics
# --------------------------------------
def extract_demographics():
    print("Extracting DM (demographics) data...")
    dm_collection = db["dm"]
    
    dm_cursor = dm_collection.find({}, {
        "usubjid": 1,
        "age": 1,
        "sex": 1,
        "race": 1
    })
    
    dm_records = list(dm_cursor)
    if not dm_records:
        print("No records found in DM collection!")
        return pd.DataFrame()
    
    dm_df = pd.DataFrame(dm_records)
    print(f"Extracted DM data with shape: {dm_df.shape}")
    return dm_df

# --------------------------------------
# MH extractor - comorbidities
# --------------------------------------
def extract_medical_history():
    print("Extracting MH (medical history) data...")
    mh_collection = db["mh"]
    
    mh_cursor = mh_collection.find({}, {"usubjid": 1, "mhterm": 1})
    mh_records = list(mh_cursor)
    
    if not mh_records:
        print("No records found in MH collection!")
        return pd.DataFrame()
    
    mh_df = pd.DataFrame(mh_records)

    # Group by subject and collect list of comorbidities
    mh_grouped = mh_df.groupby("usubjid")["mhterm"].apply(list).reset_index()
    mh_grouped.rename(columns={"mhterm": "Comorbidities"}, inplace=True)

    print(f"Extracted MH data with shape: {mh_grouped.shape}")
    return mh_grouped

# --------------------------------------
# ADREL extractor - relapse counts
# --------------------------------------
def extract_relapse_data():
    print("Extracting Relapse data from ADREL...")
    adrel_collection = db["adrel"]

    adrel_cursor = adrel_collection.find({}, {"usubjid": 1})
    adrel_records = list(adrel_cursor)

    if not adrel_records:
        print("No records found in ADREL collection!")
        # Return empty dataframe with correct columns to avoid merge error
        return pd.DataFrame(columns=["usubjid", "RelapseCount"])

    adrel_df = pd.DataFrame(adrel_records)

    # Count total relapses per subject
    relapse_counts = adrel_df.groupby("usubjid").size().reset_index(name="RelapseCount")

    print(f"Extracted relapse data with shape: {relapse_counts.shape}")
    return relapse_counts

# --------------------------------------
# Merge everything together
# --------------------------------------
def build_feature_table():
    qs_data = extract_qs_data()
    dm_data = extract_demographics()
    mh_data = extract_medical_history()
    relapse_data = extract_relapse_data()

    print("Merging QS and DM...")
    merged = pd.merge(qs_data, dm_data, on="usubjid", how="outer")

    print("Merging with MH...")
    merged = pd.merge(merged, mh_data, on="usubjid", how="outer")

    print("Merging with Relapse data...")
    merged = pd.merge(merged, relapse_data, on="usubjid", how="outer")

    print("Final merged data shape:", merged.shape)
    return merged

# --------------------------------------
# Write final merged data into MongoDB
# --------------------------------------
def write_to_mongodb(df):
    feature_collection = db["compiled_features"]
    print("Dropping previous compiled_features collection if exists...")
    feature_collection.drop()
    
    records = df.to_dict(orient="records")
    print(f"Inserting {len(records)} records into compiled_features collection...")
    feature_collection.insert_many(records)
    print("Insertion complete!")

# --------------------------------------
# Main pipeline
# --------------------------------------
if __name__ == "__main__":
    print("Starting feature table creation process (with relapse integration)...")
    feature_table = build_feature_table()
    
    if not feature_table.empty:
        write_to_mongodb(feature_table)
    else:
        print("No data extracted. Nothing inserted.")

    print("All done!")
    client.close()
