import os
import sys
import pymongo
import numpy as np
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from ml_pipeline.exception.exception import MLPipelineException
from ml_pipeline.logging.logger import logging

# Configuration for the Data Ingestion Config
from ml_pipeline.entity.config_entity import DataIngestionConfig

from ml_pipeline.entity.artifact_entity import DataIngestionArtifact

# Read .env file
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class DataIngestion:
    """
    A class to handle the ingestion of data from MongoDB into the local file system
    for machine learning purposes.
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Initializes the DataIngestion object with configuration.

        Parameters:
        -----------
        data_ingestion_config : DataIngestionConfig
            Configuration parameters for the ingestion process.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MLPipelineException(e)

    def export_collection_as_dataframe(self):
        """
        Reads a MongoDB collection and converts it to a Pandas DataFrame.
        Drops the '_id' column and replaces 'na' strings with NaN.

        Returns:
        --------
        pd.DataFrame
            The cleaned DataFrame from MongoDB.
        """
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            # Create a MongoDB client to connect to the server
            self.mongo_client = pymongo.MongoClient(host=MONGO_DB_URL)

            # Read the whole target collection
            collection = self.mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)
            df.replace({"na": np.nan}, inplace=True)

            logging.info("Data from database was retrieved.")

            return df

        except Exception as e:
            raise MLPipelineException(e)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        """
        Saves the DataFrame to the local feature store as a CSV file.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The DataFrame to be saved.

        Returns:
        --------
        pd.DataFrame
            The same DataFrame that was saved.
        """
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path

            # Create folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)

            logging.info(
                "Dataset name is {}".format(self.data_ingestion_config.collection_name)
            )
            logging.info("Retrieved data was saved locally in feature store.")

            return dataframe

        except Exception as e:
            raise MLPipelineException(e)

    def split_data_to_train_and_test(self, dataframe: pd.DataFrame):
        """
        Splits the input DataFrame into training and testing sets based on the configured ratio.
        Saves both sets locally as CSV files.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The input dataset to be split.
        """
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train/test split on the dataframe.")

            # Directory that encompases train/test split
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            # Save Train data
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )
            logging.info(
                "Train data was saved in {}".format(
                    self.data_ingestion_config.training_file_path
                )
            )

            # Save Test data
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            logging.info(
                "Test data was saved in {}".format(
                    self.data_ingestion_config.testing_file_path
                )
            )

        except Exception as e:
            raise MLPipelineException(e)

    def initiate_data_ingestion(self):
        """
        Runs the entire data ingestion pipeline:
        - Extracts data from MongoDB
        - Saves it to the feature store
        - Splits it into train and test sets

        Returns:
        --------
        DataIngestionArtifact
            An artifact containing the file paths for the train and test datasets.
        """
        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe=dataframe)
            self.split_data_to_train_and_test(dataframe=dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )

            return data_ingestion_artifact

        except Exception as e:
            raise MLPipelineException(e)
