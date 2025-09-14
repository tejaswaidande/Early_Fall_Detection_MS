import pandas as pd
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "thesis_data")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Initialize report container
data_quality_report = {}

# Helper functions
def get_missing_stats(df):
    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100
    return pd.DataFrame({'MissingCount': missing, 'MissingPercent': percent})

def get_dtype_stats(df):
    return df.dtypes.apply(lambda x: str(x)).to_dict()

def get_uniqueness(df):
    return {col: df[col].nunique() for col in df.columns}

def get_outliers(df):
    numeric_cols = df.select_dtypes(include=[np.number])
    outlier_summary = {}
    for col in numeric_cols.columns:
        q1 = numeric_cols[col].quantile(0.25)
        q3 = numeric_cols[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = numeric_cols[(numeric_cols[col] < lower) | (numeric_cols[col] > upper)]
        outlier_summary[col] = {
            'OutlierCount': outliers.shape[0],
            'Min': numeric_cols[col].min(),
            'Max': numeric_cols[col].max(),
            'Mean': numeric_cols[col].mean()
        }
    return outlier_summary

def get_duplicate_stats(df):
    return df.duplicated().sum()

def analyse_collection(collection_name):
    print(f"Analyzing collection: {collection_name}")
    df = pd.DataFrame(list(db[collection_name].find()))
    if df.empty:
        return {"note": "Empty collection"}
    
    report = {
        "RowCount": df.shape[0],
        "ColumnCount": df.shape[1],
        "MissingValues": get_missing_stats(df).to_dict(),
        "DataTypes": get_dtype_stats(df),
        "Uniqueness": get_uniqueness(df),
        "Outliers": get_outliers(df),
        "DuplicateRows": get_duplicate_stats(df)
    }
    return report

# Perform analysis
for collection_name in db.list_collection_names():
    data_quality_report[collection_name] = analyse_collection(collection_name)

# Flatten for data_quality_report.csv
flat_records = []

for collection, report in data_quality_report.items():
    if "note" in report:
        flat_records.append({
            "Collection": collection,
            "RowCount": 0,
            "ColumnCount": 0,
            "DuplicateRows": "N/A",
            "MissingPercentOver50": "N/A"
        })
        continue

    row = {
        "Collection": collection,
        "RowCount": report["RowCount"],
        "ColumnCount": report["ColumnCount"],
        "DuplicateRows": report["DuplicateRows"]
    }

    # Count columns with >50% missing
    missing_df = pd.DataFrame(report["MissingValues"])
    if not missing_df.empty and "MissingPercent" in missing_df.index:
        over_50_missing = (missing_df.loc["MissingPercent"] > 50).sum()
        row["MissingPercentOver50"] = over_50_missing
    else:
        row["MissingPercentOver50"] = "N/A"

    flat_records.append(row)

# Save data_quality_report.csv
df_report = pd.DataFrame(flat_records)
df_report.to_csv("data_quality_report.csv", index=False)
print(f"\n Data Quality Report saved to: data_quality_report_v1.csv")

# ----------------------------------------------------------
# New function: Generate column_distincts.csv
# ----------------------------------------------------------
def generate_column_distribution_report(output_path="column_distincts.csv"):
    column_map = {}

    for collection_name in db.list_collection_names():
        df = pd.DataFrame(list(db[collection_name].find()))
        column_map[collection_name] = list(df.columns) if not df.empty else []

    # Determine maximum number of columns
    max_len = max(len(cols) for cols in column_map.values())

    # Create rows for CSV
    rows = []
    for file_name, cols in column_map.items():
        row = [file_name] + cols + [""] * (max_len - len(cols))  # Pad with blanks
        rows.append(row)

    # Column headers
    headers = ["File Name"] + [f"Column_{i+1}" for i in range(max_len)]
    df_columns = pd.DataFrame(rows, columns=headers)
    df_columns.to_csv(output_path, index=False)
    print(f" Column-wise report saved to: {output_path}")

# Generate the second report
generate_column_distribution_report()
