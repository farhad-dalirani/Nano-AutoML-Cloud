import os
import sys

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_pipeline.logging.logger import logging
from ml_pipeline.exception.exception import MLPipelineException
from ml_pipeline.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from ml_pipeline.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from ml_pipeline.entity.config_entity import DataTransformationConfig, TrainingPipelineConfig
from ml_pipeline.utils.main_utils.utils import save_numpy_array_data, save_object, read_yaml_file


class DataTransformation:
    def __init__(
            self, 
            data_validation_artifact: DataValidationArtifact,
            data_transformation_config:DataTransformationConfig,
            training_pipeline_config: TrainingPipelineConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact = data_validation_artifact
            self.data_transformation_config:DataTransformationConfig = data_transformation_config
            self.training_pipeline_config:TrainingPipelineConfig = training_pipeline_config
            self.schema = read_yaml_file(file_path=training_pipeline_config.schema_file_path)
        except Exception as e:
            raise MLPipelineException(e)

    @staticmethod
    def read_data(file_path: str)->pd.DataFrame:
        """
        Reads a CSV file into a pandas DataFrame.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: DataFrame containing the data.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MLPipelineException(e)
    
    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Creates a data transformer pipeline that preprocesses numerical and categorical features
        according to the schema defined in the training pipeline configuration.

        Processing steps:
        - Numerical features:
            - Missing value imputation using KNNImputer with predefined parameters.
            - Feature scaling using StandardScaler.
        - Categorical features:
            - Missing value imputation using the most frequent strategy.
            - One-hot encoding with handling of unknown categories.

        The method also validates the schema to ensure the target column is not included in
        the feature columns.

        Returns:
            ColumnTransformer: A composite transformer that applies the appropriate preprocessing
            pipelines to numerical and categorical features.

        Raises:
            MLPipelineException: If schema validation fails or any error occurs during pipeline creation.
        """
        try:
            numerical_columns = self.schema.get("numerical_columns", [])
            if numerical_columns is None:
                numerical_columns = []
            categorical_columns = self.schema.get("categorical_columns", [])
            if categorical_columns is None:
                categorical_columns = []
            ignore_columns = self.schema.get("ignore_columns", [])
            if ignore_columns is None:
                ignore_columns = []
            target_column = self.schema.get("target_column")

            # Validate schema: target column should NOT be in feature lists
            if target_column in numerical_columns:
                raise MLPipelineException(
                    f"Invalid schema: target column '{target_column}' should not be listed under 'numerical_columns'."
                )
            if target_column in categorical_columns:
                raise MLPipelineException(
                    f"Invalid schema: target column '{target_column}' should not be listed under 'categorical_columns'."
                )

            # Final list after removing target and ignored columns
            numerical_columns = [col for col in numerical_columns if col not in ignore_columns and col != target_column]
            categorical_columns = [col for col in categorical_columns if col not in ignore_columns and col != target_column]

            logging.info(f"Using numerical columns: {numerical_columns}")
            logging.info(f"Using categorical columns: {categorical_columns}")

            # Pipelines
            num_pipeline = Pipeline([
                ("imputer", KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_columns),
                    ("cat", cat_pipeline, categorical_columns)
                ],
                remainder='drop'
            )

            return preprocessor

        except Exception as e:
            raise MLPipelineException(e)

    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class.")
        try:
            logging.info("Starting data transformation.")

            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # Read data schema file for to check if target class values mapping to new values is needed  
            schema = read_yaml_file(file_path=self.training_pipeline_config.schema_file_path)

            # Training dataframe
            input_feature_train_df = train_df.drop(columns=[self.training_pipeline_config.target_column], axis=1)
            target_feature_train_df = train_df[self.training_pipeline_config.target_column]
            # Remap target class labels to new values based on predefined mapping
            if schema.get("DATA_TRANSFORMATION_ENABLE_TARGET_CLASS_MAPPING", False):
                mapping = schema.get("DATA_TRANSFORMATION_TARGET_CLASS_MAPPING", {})
                target_feature_train_df = target_feature_train_df.replace(mapping)
                logging.info("Trainset Target class values were remaped: {}".format(mapping))

            # Test dataframe
            input_feature_test_df = test_df.drop(columns=[self.training_pipeline_config.target_column], axis=1)
            target_feature_test_df = test_df[self.training_pipeline_config.target_column]
            # Remap target class labels to new values based on predefined mapping
            if schema.get("DATA_TRANSFORMATION_ENABLE_TARGET_CLASS_MAPPING", False):
                mapping = schema.get("DATA_TRANSFORMATION_TARGET_CLASS_MAPPING", {})
                target_feature_test_df = target_feature_test_df.replace(mapping)
                logging.info("Test Target class values were remaped: {}".format(mapping))

            # Get imputer for filling missing features
            preprocessor = self.get_data_transformer_object()

            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            train_arr = np.column_stack((transformed_input_train_feature, target_feature_train_df))
            test_arr = np.column_stack((transformed_input_test_feature, target_feature_test_df))

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)

            logging.info("Ended data transformation.")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            
            return data_transformation_artifact

        except Exception as e:
            raise MLPipelineException(e)


