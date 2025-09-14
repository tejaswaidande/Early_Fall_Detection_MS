
import os
import pandas as pd
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB connection setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "thesis_data")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
print("Connected to MongoDB")

# Load compiled feature data from MongoDB
compiled_collection = db["compiled_features"]
compiled_cursor = compiled_collection.find({})
compiled_records = list(compiled_cursor)

if not compiled_records:
    print("No data found in compiled_features collection!")
    exit()

# Convert to DataFrame
compiled_df = pd.DataFrame(compiled_records)
print(f"Loaded compiled feature table with shape: {compiled_df.shape}")

# ---------------------------------------------
# Handle MongoDB JSON types (cleanup)
# ---------------------------------------------

# Convert MongoDB style NaNs (like {"$numberDouble": "NaN"}) to proper NaN
compiled_df.replace({"$numberDouble": "NaN"}, np.nan, inplace=True)

# Drop Mongo internal _id column for simplicity
if '_id' in compiled_df.columns:
    compiled_df.drop(columns=['_id'], inplace=True)

# ---------------------------------------------
# Handling missing values
# ---------------------------------------------

missing_summary = compiled_df.isna().mean() * 100
print("\nMissing values (%):")
print(missing_summary.sort_values(ascending=False).head(15))

# Drop columns with excessive missingness (>60%)
high_missing_cols = missing_summary[missing_summary > 60].index.tolist()
print("\nDropping columns with >60% missing:", high_missing_cols)
compiled_df.drop(columns=high_missing_cols, inplace=True)

# Fill numeric missing values with median
numeric_cols = compiled_df.select_dtypes(include=[np.number]).columns.tolist()
compiled_df[numeric_cols] = compiled_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
compiled_df[numeric_cols] = compiled_df[numeric_cols].fillna(compiled_df[numeric_cols].median())

# ---------------------------------------------
# Categorical cleaning: sex and race
# ---------------------------------------------

# Normalize sex
compiled_df['sex'] = compiled_df['sex'].str.upper().map({'M': 0, 'F': 1, 'MALE': 0, 'FEMALE': 1})
compiled_df['sex'].fillna(-1, inplace=True)  # unknown

# Normalize race
race_mapping = {'WHITE': 0, 'BLACK': 1, 'ASIAN': 2, 'OTHER': 3}
compiled_df['race'] = compiled_df['race'].str.upper().map(race_mapping)
compiled_df['race'].fillna(-1, inplace=True)

# ---------------------------------------------
# Comorbidity flag extraction
# ---------------------------------------------

def extract_comorbidity_flag(comorbidities, keyword):
    if not isinstance(comorbidities, list):
        return 0
    return int(any(keyword.lower() in str(c).lower() for c in comorbidities))

compiled_df['has_diabetes'] = compiled_df['Comorbidities'].apply(lambda x: extract_comorbidity_flag(x, 'diabetes'))
compiled_df['has_arthritis'] = compiled_df['Comorbidities'].apply(lambda x: extract_comorbidity_flag(x, 'arthritis'))
compiled_df['has_hypertension'] = compiled_df['Comorbidities'].apply(lambda x: extract_comorbidity_flag(x, 'hypertension'))

compiled_df.drop(columns=['Comorbidities'], inplace=True)

# ---------------------------------------------
# Relapse Count - handle if missing
# ---------------------------------------------
if 'RelapseCount' in compiled_df.columns:
    compiled_df['RelapseCount'] = pd.to_numeric(compiled_df['RelapseCount'], errors='coerce').fillna(0)

# ---------------------------------------------
# Final check before writing to MongoDB
# ---------------------------------------------

print("\nFinal cleaned dataset preview:")
print(compiled_df.head())

# Write cleaned data into new MongoDB collection
cleaned_collection = db["cleaned_features"]
print("\nDropping previous cleaned_features collection if exists")
cleaned_collection.drop()

records = compiled_df.to_dict(orient='records')
print(f"Inserting {len(records)} cleaned records into cleaned_features collection")
cleaned_collection.insert_many(records)
print("Cleaning process complete")

client.close()
