import os
import pymongo
import pandas as pd
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATA_FILE_PATH = r"F:\PROJECTS\Insurance_project\data\insurance.csv"
database_name = os.getenv("DATABASE_NAME")
collection_name = os.getenv("COLLECTION_NAME")
database_url = os.getenv("DATABASE_URL")

if not database_name or not collection_name or not database_url:
    raise ValueError("One or more environment variables are missing!")

client = pymongo.MongoClient(database_url)

if __name__ == "__main__":
    # Load the data into a DataFrame
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and columns: {df.shape}")

    df.reset_index(drop=True, inplace=True)

    # Convert DataFrame to JSON
    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    # Clear the collection before inserting new records
    collection = client[database_name][collection_name]
    collection.delete_many({})  # Deletes all records in the collection
    print(f"All records deleted from {collection_name} collection.")

    # Insert the new records
    collection.insert_many(json_record)
    print(f"New records inserted into {collection_name} collection.")
