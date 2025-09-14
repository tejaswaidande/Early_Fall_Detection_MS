import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_db():
    """
    Connect to MongoDB and return the database object.
    Handles connection errors and provides verbose feedback.
    """
    try:
        # Get connection settings with fallback defaults
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        db_name = os.getenv("DB_NAME", "thesis_data")

        print(f"Connecting to MongoDB at: {mongo_uri}")
        print(f"Using database: {db_name}")

        # Create client with timeout for connection attempt
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        
        # Test the connection
        client.server_info()  # Will raise an exception if connection fails
        print("Successfully connected to MongoDB!")
        
        return client[db_name]

    except Exception as e:
        print("\n  MongoDB connection failed!")
        print(f"Error details: {str(e)}")
        print("Possible fixes:")
        print("- Is MongoDB running? Try `net start MongoDB` (Windows)")
        print("- Check your .env file for correct MONGO_URI and DB_NAME")
        print("- For cloud Atlas: Is the IP whitelisted?")
        raise  # Re-raise the exception after logging

# Example usage:
if __name__ == "__main__":
    try:
        db = get_db()
        print(f"Available collections in '{db.name}': {db.list_collection_names()}")
    except Exception:
        print("Failed to initialize database connection.")
