import os
import sys
import numpy as np
import pandas as pd

"""
Defining common constant variables for training pipeline
"""

# Name of the overall pipeline
PIPELINE_NAME: str = "MLPipeline"

# Directory of schema files
SCHEMA_DIR = "data_schema"

# Directory where all pipeline artifacts (e.g., models, metrics, logs) will be stored
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "input_data.csv"

# Names of the train and test split files
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

# Directory where the final trained models will be saved
SAVED_MODEL_DIR = os.path.join("saved_models")

MODEL_FILE_NAME = "model.pkl"

# Directory to save trained model for use
FINAL_MODEL_DIR = os.path.join("final_model")

LOGS_DIR = os.path.join("logs")

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
# Root directory for data ingestion artifacts
DATA_INGESTION_DIR_NAME: str = "data_ingestion"

# Directory for storing raw features extracted from data
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

# Directory to store ingested (train/test) data
DATA_INGESTION_INGESTED_DIR: str = "ingested"

# Ratio to split dataset into test and train sets
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2


"""
Data Validation related constant start with DATA_VALIDATION VAR NAME
"""
# Root directory for data validation outputs
DATA_VALIDATION_DIR_NAME: str = "data_validation"

# Directory to store valid data
DATA_VALIDATION_VALID_DIR: str = "validated"

# Directory to store invalid data
DATA_VALIDATION_INVALID_DIR: str = "invalid"

# Directory to store data drift reports
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"

# Filename of the drift report generated
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

# Filename for the serialized preprocessing object
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"


"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""
# Root directory for data transformation outputs
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"

# Directory for transformed datasets
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"

# Directory for data transformer object
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

# Parameters for the KNNImputer used to fill missing values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}

# Paths for the transformed train and test datasets
DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"
DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"


"""
Model Training ralated constant start with MODEL_TRAINER
"""
# Root directory for model training outputs
MODEL_TRAINER_DIR_NAME: str = "model_trainer"

# Directory to store the trained model
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"

# Filename for the saved trained model
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"

MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.05
