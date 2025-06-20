import os
import sys 
import numpy as np
import pandas as pd

"""
Defining common constant variables for training pipeline
"""
TARGET_COLUMN = "result"
PIPELINE_NAME: str = "MLPipeline"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "input_data.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

SAVED_MODEL_DIR =os.path.join("saved_models")
MODEL_FILE_NAME = "model.pkl"


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "MLData"
DATA_INGESTION_DATABASE_NAME: str = "MLPROJECT_DB"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2


"""
Data Validation related constant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"


"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

## KNN to replace nan values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}
DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"
DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"

# This is dataset-dependent — use it if you need to map specific
# classes in your dataset to others.
# Flag to enable or disable mapping of target class values during data transformation.
# If True, the mapping defined below will be applied to the target column.
# Dictionary that defines how to map target class values.
# In this case, all instances of -1 in the target will be replaced with 0.
# Example: if your dataset has classes [-2, -1, 0, 1] and you want to 
# normalize them to [0, 1, 2, 3], you could use a
#  mapping like {-2: 0, -1: 1, 0: 2, 1: 3}
DATA_TRANSFORMATION_ENABLE_TARGET_CLASS_MAPPING: bool=True
DATA_TRANSFORMATION_TARGET_CLASS_MAPPING: bool={-1: 0}