from datetime import datetime
import os
from ml_pipeline.constants import training_pipeline

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


class DataValidationConfig:
    """
    Configuration class for managing data validation settings in the training pipeline.

    Attributes:
        data_validation_dir (str): Root directory for all data validation artifacts.
        valid_data_dir (str): Directory path to store validated (clean) data.
        invalid_data_dir (str): Directory path to store data that fails validation.
        valid_train_file_path (str): Path to the validated training dataset file.
        valid_test_file_path (str): Path to the validated testing dataset file.
        invalid_train_file_path (str): Path to the invalid training dataset file.
        invalid_test_file_path (str): Path to the invalid testing dataset file.
        drift_report_file_path (str): Path to the data drift report generated during validation.
    """
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, 
            training_pipeline.DATA_VALIDATION_DIR_NAME
        )
        self.valid_data_dir: str = os.path.join(
            self.data_validation_dir, 
            training_pipeline.DATA_VALIDATION_VALID_DIR
        )
        self.invalid_data_dir: str = os.path.join(
            self.data_validation_dir, 
            training_pipeline.DATA_VALIDATION_INVALID_DIR
        )
        self.valid_train_file_path: str = os.path.join(
            self.valid_data_dir, 
            training_pipeline.TRAIN_FILE_NAME
        )
        self.valid_test_file_path: str = os.path.join(
            self.valid_data_dir, 
            training_pipeline.TEST_FILE_NAME
        )
        self.invalid_train_file_path: str = os.path.join(
            self.invalid_data_dir, 
            training_pipeline.TRAIN_FILE_NAME
        )
        self.invalid_test_file_path: str = os.path.join(
            self.invalid_data_dir, 
            training_pipeline.TEST_FILE_NAME
        )
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )


class DataTransformationConfig:
    """
    Configuration class for managing data transformation settings in the training pipeline.

    Attributes:
        data_transformation_dir (str): Root directory for all data transformation artifacts.
        transformed_train_file_path (str): Path to the transformed training data file (in .npy format).
        transformed_test_file_path (str): Path to the transformed testing data file (in .npy format).
        transformed_object_file_path (str): Path to the serialized preprocessing object (e.g., scaler, encoder).
    """
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_TRANSFORMATION_DIR_NAME 
        )
        self.transformed_train_file_path: str = os.path.join( 
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TRAIN_FILE_NAME.replace("csv", "npy")
        )
        self.transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir,  
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TEST_FILE_NAME.replace("csv", "npy")
        )
        self.transformed_object_file_path: str = os.path.join( 
            self.data_transformation_dir, 
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            training_pipeline.PREPROCESSING_OBJECT_FILE_NAME
        )


class ModelTrainerConfig:
    """
    Configuration class for managing model training settings in the training pipeline.

    Attributes:
        model_trainer_dir (str): Root directory for all model training artifacts.
        trained_model_file_path (str): Path to the final trained model file.
        expected_accuracy (float): Minimum accuracy threshold that the model must achieve.
        overfitting_underfitting_threshold (float): Threshold used to detect overfitting or underfitting
                                                    based on the difference between training and testing performance.
        model_type (str): Type of model to be trained, either 'classification' or 'regression'.
    """
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, 
            training_pipeline.MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, 
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR, 
            training_pipeline.MODEL_FILE_NAME
        )
        self.expected_accuracy: float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold = training_pipeline.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD
        self.model_type = training_pipeline.MODEL_TYPE