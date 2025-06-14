import os
import json
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
import certifi
import pandas as pd
import numpy as np
import pymongo
from ml_pipeline.exception.exception import MLPipelineException
from ml_pipeline.logging.logger import logging
from pathlib import Path
from ucimlrepo import fetch_ucirepo 

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
            raise MLPipelineException(e)

    def extract_phishing_records(self, ):
        """
        Fetches the Phishing Websites dataset from the UCI repository, processes it into a DataFrame,
        and returns the data as a list of JSON-like dictionary records.

        Returns:
        --------
        list of dict
            A list of dictionary records, where each dictionary represents a column of the original dataset,
            keyed by row index.

        Raises:
        -------
        NetworkSecurityException
            If any error occurs during dataset fetching or processing.
        """
        try:
            # Get "Phishing Websites" dataset from UCI repository
            phishing_websites = fetch_ucirepo(id=327) 
  
            # Data (as pandas dataframes) 
            X = phishing_websites.data.features 
            y = phishing_websites.data.targets 

            # Concat features and target class into one dataframe             
            data = pd.DataFrame(data=X)
            data = pd.concat([X, y],axis=1)
            data.reset_index(drop=True, inplace=True)

            # Transpose the DataFrame, convert it to JSON, load it back as a Python dict, 
            # then extract the values as a list of records
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise MLPipelineException(e)
        
    def load_data_to_mongodb(self, records, database, collection):
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
            raise MLPipelineException(e)


# Main ETL execution
if __name__ == '__main__':
    DATABASE="MLPROJECT_DB"
    Collection="MLData"
    
    # Create an instance of the data extraction class
    networkobj=NetworkDataExtract()

    # Extract phishing dataset records from UCI repository
    records=networkobj.extract_phishing_records()

    # Load the extracted records into the specified MongoDB collection
    no_of_records=networkobj.load_data_to_mongodb(records=records, database=DATABASE, collection=Collection)

    # Output the number of records successfully inserted
    print(no_of_records)