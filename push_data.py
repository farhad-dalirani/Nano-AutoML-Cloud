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

# URL of MongoDB database that contains datasets
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

# Optional fallback for local development only
if MONGO_DB_URL is None:
    from dotenv import load_dotenv
    load_dotenv()
    MONGO_DB_URL = os.getenv("MONGO_DB_URL")

ca = certifi.where()


class DataExtract:
    def __init__(self, uci_id: int):
        """
        Initializes the DataExtract object.
        Currently a placeholder for future setup or logging.
        """
        try:
            self.uci_id = uci_id
        except Exception as e:
            raise MLPipelineException(e)

    def extract_phishing_records(
        self,
    ):
        """
        Fetches the dataset from the UCI repository, processes it into a DataFrame,
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
            phishing_websites = fetch_ucirepo(id=self.uci_id)

            # Data (as pandas dataframes)
            X = phishing_websites.data.features
            y = phishing_websites.data.targets

            # Concat features and target class into one dataframe
            data = pd.DataFrame(data=X)
            data = pd.concat([X, y], axis=1)
            data.reset_index(drop=True, inplace=True)

            # Transpose the DataFrame, convert it to JSON, load it back as a Python dict,
            # then extract the values as a list of records
            records = list(json.loads(data.T.to_json()).values())
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
if __name__ == "__main__":

    # Database name
    database_name = "MLPROJECT_DB"

    # Various example datasets
    datasets = {
        "UCI-Phishing-Websites-Classification": {
            "uci-id": 327,
            "collection-name": "UCI_Phishing_Websites",
        },
        "UCI-Bike-Sharing-Regression": {
            "uci-id": 275,
            "collection-name": "UCI_Bike_Sharing",
        },
        "UCI-Iris": {"uci-id": 53, "collection-name": "UCI_Iris"},
    }

    for key in datasets:
        dataset_name = key
        uci_id = datasets[key]["uci-id"]
        DB_collection_name = datasets[key]["collection-name"]

        # Create an instance of the data extraction class
        networkobj = DataExtract(uci_id=uci_id)

        # Extract dataset records from UCI repository
        records = networkobj.extract_phishing_records()

        # Load the extracted records into the specified MongoDB collection
        no_of_records = networkobj.load_data_to_mongodb(
            records=records, database=database_name, collection=DB_collection_name
        )

        # Output the number of records successfully inserted
        print(no_of_records)
