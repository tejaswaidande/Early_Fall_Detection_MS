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

# Load cleaned feature data from MongoDB
cleaned_collection = db["cleaned_features"]
cleaned_cursor = cleaned_collection.find({})
cleaned_records = list(cleaned_cursor)

if not cleaned_records:
    print("No data found in cleaned_features collection!")
    exit()

# Convert to DataFrame
df = pd.DataFrame(cleaned_records)
print(f"Loaded cleaned feature table with shape: {df.shape}")

# Drop Mongo internal _id if still exists
if '_id' in df.columns:
    df.drop(columns=['_id'], inplace=True)

# ---------------------------------------------
# Handle type coercion: convert string numbers to numeric
# ---------------------------------------------
print("Converting string numbers to numeric where possible...")
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')  # don't force errors yet

# Manually handle common cases where numbers are still stored as strings
for col in df.columns:
    if df[col].dtype == object:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # second pass

# ---------------------------------------------
# Mapping textual questionnaire responses to numeric codes
# ---------------------------------------------

# Example mapping dictionary (can be expanded)
text_mapping = {
    "None of the time": 0,
    "A little of the time": 1,
    "Some of the time": 2,
    "Moderate amount of the time": 2,
    "Most of the time": 3,
    "All of the time": 4,
    "Mostly true": 4,
    "Mostly false": 0,
    "Never": 0,
    "Rarely": 1,
    "Sometimes": 2,
    "Often": 3,
    "Always": 4
}

# Apply text mapping
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].replace(text_mapping)
        df[col] = pd.to_numeric(df[col], errors='coerce')  # finally convert remaining

# ---------------------------------------------
# After mapping, check missing again
# ---------------------------------------------
missing_summary = df.isna().mean() * 100
print("\nUpdated missing values after text mapping:")
print(missing_summary.sort_values(ascending=False).head(15))

# Drop columns with still excessive missingness (>50%)
high_missing_cols = missing_summary[missing_summary > 50].index.tolist()
print("\nDropping columns with >50% missing after mapping:", high_missing_cols)
df.drop(columns=high_missing_cols, inplace=True)

# ---------------------------------------------
# Re-impute remaining missing numeric values
# ---------------------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Final shape check
print("\nFinal shape after full cleaning:", df.shape)

# ---------------------------------------------
# Write final fully cleaned dataset to MongoDB
# ---------------------------------------------
final_collection = db["final_features"]
print("\nDropping previous final_features collection if exists")
final_collection.drop()

records = df.to_dict(orient='records')
print(f"Inserting {len(records)} records into final_features collection")
final_collection.insert_many(records)
print("Full cleaning process complete!")

client.close()
