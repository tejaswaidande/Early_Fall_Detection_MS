from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["thesis_data"]

# Count total patient records
patient_count = db["final_selected_features"].count_documents({})
print("Total patients in final_selected_features:", patient_count)
