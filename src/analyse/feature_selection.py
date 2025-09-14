import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

# Load final cleaned data from MongoDB
final_collection = db["final_features"]
final_cursor = final_collection.find({})
final_records = list(final_cursor)

if not final_records:
    print("No data found in final_features collection")
    exit()

# Convert to DataFrame
df = pd.DataFrame(final_records)
print(f"Loaded final cleaned dataset with shape: {df.shape}")

# Drop Mongo internal _id if exists
if '_id' in df.columns:
    df.drop(columns=['_id'], inplace=True)

# ---------------------------------------------
# Remove constant (zero variance) features
# ---------------------------------------------
print("\nChecking for zero variance features...")
zero_var_cols = [col for col in df.columns if df[col].nunique() <= 1]
print("Zero variance columns:", zero_var_cols)
df.drop(columns=zero_var_cols, inplace=True)

# ---------------------------------------------
# Correlation matrix to detect highly correlated features
# ---------------------------------------------
print("\nComputing correlation matrix...")
corr_matrix = df.corr()

# Just for visualization: (optional)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# ---------------------------------------------
# Drop one of highly correlated feature pairs (threshold > 0.9)
# ---------------------------------------------
print("\nIdentifying highly correlated feature pairs...")
threshold = 0.9
corr_pairs = set()
cols = corr_matrix.columns
for i in range(len(cols)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            col1 = cols[i]
            col2 = cols[j]
            print(f"High correlation detected: {col1} vs {col2} (r = {corr_matrix.iloc[i, j]:.2f})")
            corr_pairs.add(col2)  # Keep one of them arbitrarily

# Drop correlated columns
print("\nDropping highly correlated columns:", corr_pairs)
df.drop(columns=list(corr_pairs), inplace=True)

# ---------------------------------------------
# Final feature list ready for ML
# ---------------------------------------------
print("\nFinal selected features:")
print(df.columns.tolist())

# Write selected features to MongoDB
selected_collection = db["final_selected_features"]
print("\nDropping previous final_selected_features collection if exists...")
selected_collection.drop()

records = df.to_dict(orient='records')
print(f"Inserting {len(records)} records into final_selected_features collection...")
selected_collection.insert_many(records)
print("Feature selection process complete!")
 
client.close()
