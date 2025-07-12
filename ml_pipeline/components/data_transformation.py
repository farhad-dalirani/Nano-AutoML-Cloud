import os
import sys

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

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
    
    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        """
        Creates and returns a scikit-learn Pipeline object for data transformation.

        This method initializes a KNNImputer with predefined parameters and encapsulates 
        it within a scikit-learn Pipeline. The resulting pipeline can be used to impute 
        missing values in the dataset.

        Returns:
            Pipeline: A scikit-learn Pipeline object containing a KNNImputer step.

        Raises:
            MLPipelineException: If an error occurs during the creation of the pipeline.
        """
        logging.info("Entered to get_data_transformer_object of Transformation class.")
        try:
            imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info("Initialize KNNImputer with {}".format(DATA_TRANSFORMATION_IMPUTER_PARAMS))
            processor:Pipeline = Pipeline([("imputer", imputer)])
            return processor
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
            preprocessor = DataTransformation.get_data_transformer_object()

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


