import os
import pandas as pd
from pathlib import Path
from pymongo import MongoClient
from dotenv import load_dotenv
from tqdm import tqdm  # For progress bars

# Load environment variables
load_dotenv()

# Constants
DATA_DIR = Path(r"C:\Users\Tejas\Desktop\thesis_project\data\raw_data")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "thesis_data")

def get_collection_name(file_path: Path) -> str:
    """Convert filename to collection name (e.g., 'ce.csv' -> 'ce')"""
    return file_path.stem.lower()

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names (lowercase, no spaces)"""
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def insert_csv_to_mongodb(file_path: Path, client: MongoClient):
    """Process and insert a single CSV file"""
    try:
        # Read CSV
        df = pd.read_csv(file_path)
        
        # Connect to DB
        db = client[DB_NAME]
        
        # Clean data
        df = clean_column_names(df)
        df = df.where(pd.notnull(df), None)  # Replace NaN with None for MongoDB
        
        # Generate collection name
        collection_name = get_collection_name(file_path)
        
        # Insert in batches (for large files)
        collection = db[collection_name]
        batch_size = 1000
        inserted_count = 0
        
        for i in tqdm(range(0, len(df), batch_size), desc=f"Inserting {file_path.name}"):
            batch = df.iloc[i:i + batch_size].to_dict('records')
            result = collection.insert_many(batch)
            inserted_count += len(result.inserted_ids)
        
        print(f"Inserted {inserted_count} docs to {collection_name}")

    except Exception as e:
        print(f"\nFailed to process {file_path.name}: {str(e)}")

def insert_excel_to_mongodb(file_path: Path, client: MongoClient):
    """Process and insert a single Excel file"""
    try:
        # Read Excel (handle multi-sheet files)
        dfs = pd.read_excel(file_path, sheet_name=None)
        
        # Connect to DB
        db = client[DB_NAME]
        
        for sheet_name, df in dfs.items():
            # Clean data
            df = clean_column_names(df)
            df = df.where(pd.notnull(df), None)  # Replace NaN with None for MongoDB
            
            # Generate collection name
            base_collection = get_collection_name(file_path)
            collection_name = f"{base_collection}_{sheet_name}" if len(dfs) > 1 else base_collection
            
            # Insert in batches (for large files)
            collection = db[collection_name]
            batch_size = 1000
            inserted_count = 0
            
            for i in tqdm(range(0, len(df), batch_size), desc=f"Inserting {file_path.name} ({sheet_name})"):
                batch = df.iloc[i:i + batch_size].to_dict('records')
                result = collection.insert_many(batch)
                inserted_count += len(result.inserted_ids)
            
            print(f"Inserted {inserted_count} docs to {collection_name}")

    except Exception as e:
        print(f"\nFailed to process {file_path.name}: {str(e)}")

def main():
    # Get all data files
    excel_files = list(DATA_DIR.glob("*.xlsx")) + list(DATA_DIR.glob("*.xls"))
    csv_files = list(DATA_DIR.glob("*.csv"))
    all_files = excel_files + csv_files
    
    if not all_files:
        print("No data files found in raw_data directory!")
        return
    
    print(f"Found {len(all_files)} files to process...")
    
    # Connect to MongoDB (reuse client for better performance)
    client = MongoClient(MONGO_URI)
    
    # Process each file
    for file_path in all_files:
        if file_path.suffix.lower() in ['.xlsx', '.xls']:
            insert_excel_to_mongodb(file_path, client)
        elif file_path.suffix.lower() == '.csv':
            insert_csv_to_mongodb(file_path, client)
    
    client.close()
    print("\nAll files processed!")

if __name__ == "__main__":
    main()
