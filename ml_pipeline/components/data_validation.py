import os
import sys
import pandas as pd
from scipy.stats import ks_2samp

from ml_pipeline.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from ml_pipeline.entity.config_entity import DataValidationConfig
from ml_pipeline.constants.training_pipeline import SCHEMA_FILE_PATH
from ml_pipeline.utils.main_utils.utils import read_yaml_file, write_yaml_file
from ml_pipeline.exception.exception import MLPipelineException
from ml_pipeline.logging.logger import logging


class DataValidation:
    def __init__(
            self, 
            data_ingestion_artifact: DataIngestionArtifact,
            data_validation_config: DataValidationConfig):
        """
        Initializes the DataValidation class with ingestion artifacts and config.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): Contains file paths for train/test data.
            data_validation_config (DataValidationConfig): Configuration settings for validation.
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
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

    def validate_number_of_columns(self, dataframe:pd.DataFrame)->bool:
        """
        Checks if the DataFrame has the expected number of columns.

        Args:
            dataframe (pd.DataFrame): Input data to validate.

        Returns:
            bool: True if the number of columns matches the schema, else False.
        """
        try:
            num_of_columns_in_schema = len(self._schema_config['columns'])
            
            logging.info("Required number of columns: {}".format(num_of_columns_in_schema))
            logging.info("Dataframe has columns: {}".format(len(dataframe.columns)))

            if num_of_columns_in_schema == len(dataframe.columns):
                return True
            else:
                return False

        except Exception as e:
            raise MLPipelineException(e)

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float=0.05)->bool:
        """
        Detects data drift between train/test datasets using the Kolmogorov-Smirnov test.

        Args:
            base_df (pd.DataFrame): Baseline dataset (e.g., training data).
            current_df (pd.DataFrame): Dataset to compare (e.g., test data).
            threshold (float): p-value threshold for drift detection. Default is 0.05.

        Returns:
            bool: True if no drift is detected, False otherwise.
        """
        try:
            status = True
            report={}
            for column_i in base_df.columns:
                d1 = base_df[column_i]
                d2 = current_df[column_i]
                
                # Calculate Kolmogorov–Smirnov 2-sample test
                is_same_dist = ks_2samp(d1, d2)
                if threshold <= is_same_dist.pvalue:
                    is_found=False
                else:
                    is_found=True
                    status=False

                report.update({
                    column_i:{
                        "p_value": float(is_same_dist.pvalue),
                        "drift_status": is_found
                    }
                })            

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report, replace=True)

            return status
        
        except Exception as e:
            raise MLPipelineException(e)

    def initiate_data_validation(self)->DataValidationArtifact:
        """
        Performs the data validation process.

        This includes reading train/test datasets, validating schema compliance,
        detecting dataset drift, and saving valid data and drift reports.

        Returns:
            DataValidationArtifact: Artifact containing validation status and output file paths.
        """
        try:
            # Path to training and test sets
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Read test and train data
            train_dataframe = DataValidation.read_data(file_path=train_file_path)
            test_dataframe = DataValidation.read_data(file_path=test_file_path)

            # Validate number of columns
            status_num_col_train = self.validate_number_of_columns(dataframe=train_dataframe)
            if status_num_col_train == False:
                error_message_num_col_train = "Train dataframe does not have all columns."
            status_num_col_test = self.validate_number_of_columns(dataframe=test_dataframe)
            if status_num_col_test == False:
                error_message_num_col_test = "Test dataframe does not have all columns."
            
            # Check data drift between test and train sets
            status_drift = self.detect_dataset_drift(
                base_df=train_dataframe, 
                current_df=test_dataframe
            )
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path, index=False, header=True
            )

            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path, index=False, header=True
            )

            overal_status = status_drift and status_num_col_train and status_num_col_test

            return DataValidationArtifact(
                validation_status=overal_status,
                valid_train_file_path=self.data_ingestion_artifact.train_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

        except Exception as e:
            raise MLPipelineException(e)
        

