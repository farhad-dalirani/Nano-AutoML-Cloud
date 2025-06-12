from datetime import datetime
import os
from networksecurity.constants import training_pipeline

class TrainingPipelineConfig:
    """
    Configuration class for setting up the training pipeline.

    Attributes:
        pipeline_name (str): Name of the pipeline.
        artifact_name (str): Base directory name where artifacts are stored.
        artifact_dir (str): Full path to the artifact directory including a timestamp.
        model_dir (str): Path where the final model will be stored.
        timestamp (str): Timestamp used to create unique directory paths.
    """
    def __init__(self, timestamp=datetime.now()):
        timestamp=timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name=training_pipeline.PIPELINE_NAME
        self.artifact_name=training_pipeline.ARTIFACT_DIR
        self.artifact_dir=os.path.join(self.artifact_name, timestamp)
        self.model_dir=os.path.join("final_model")
        self.timestamp: str=timestamp


class DataIngestionConfig:
    """
    Configuration class for managing data ingestion settings.

    Attributes:
        data_ingestion_dir (str): Root directory for all data ingestion-related files.
        feature_store_file_path (str): Path to the feature store file used in the pipeline.
        training_file_path (str): Path to the training dataset file.
        testing_file_path (str): Path to the testing dataset file.
        train_test_split_ratio (float): Ratio used to split the dataset into training and testing subsets.
        collection_name (str): Name of the MongoDB collection containing the raw data.
        database_name (str): Name of the MongoDB database containing the collection.
    """
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir:str=os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path: str = os.path.join(
                self.data_ingestion_dir, 
                training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR, training_pipeline.FILE_NAME
            )
        self.training_file_path: str = os.path.join(
                self.data_ingestion_dir, 
                training_pipeline.DATA_INGESTION_INGESTED_DIR, 
                training_pipeline.TRAIN_FILE_NAME
            )
        self.testing_file_path: str = os.path.join(
                self.data_ingestion_dir, 
                training_pipeline.DATA_INGESTION_INGESTED_DIR, 
                training_pipeline.TEST_FILE_NAME
            )
        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME