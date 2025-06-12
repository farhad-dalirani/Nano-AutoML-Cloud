import os
import json
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
import certifi
import pandas as pd
import numpy as np
import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from pathlib import Path

# Load environment variables
load_dotenv()

# Get MongoDB connection string from environment
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

ca = certifi.where()

class NetworkDataExtract:
    def __init__(self):
        """
        Initializes the NetworkDataExtract object.
        Currently a placeholder for future setup or logging.
        """
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e)

    def csv_to_json_converter(self, file_path: Path):
        """
        Converts a CSV file to a list of JSON records.

        Parameters:
        -----------
        file_path : Path
            Path to the CSV file.

        Returns:
        --------
        list
            A list of JSON (dictionary) records derived from the CSV.
        """
        try:
            data = pd.read_csv(filepath_or_buffer=file_path)
            data.reset_index(drop=True, inplace=True)
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e)
        
    def insert_data_to_mongodb(self, records, database, collection):
        """
        Inserts list of JSON records into specified MongoDB database and collection.

        Parameters:
        -----------
        records : list
            List of JSON-like dictionary records to be inserted.
        database : str
            Name of the MongoDB database.
        collection : str
            Name of the MongoDB collection.

        Returns:
        --------
        int
            Number of records successfully inserted.
        """
        try:
            # Connect to MongoDB
            mongo_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            
            # Access database and collection
            db = mongo_client[database]
            coll = db[collection]
            
            # Insert records
            coll.insert_many(records)
            
            return len(records)
        
        except Exception as e:
            raise NetworkSecurityException(e)


# Main ETL execution
if __name__ == '__main__':
    FILE_PATH="Network_Data/phisingData.csv"
    DATABASE="NETWORKSECURITY_DB"
    Collection="NetworkData"
    
    networkobj=NetworkDataExtract()
    records=networkobj.csv_to_json_converter(file_path=FILE_PATH)
    no_of_records=networkobj.insert_data_to_mongodb(records=records, database=DATABASE, collection=Collection)
    print(no_of_records)