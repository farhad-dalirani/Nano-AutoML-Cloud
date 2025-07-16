from datetime import datetime
import os
from ml_pipeline.constants import training_pipeline
from ml_pipeline.utils.main_utils.utils import read_schema_file


class TrainingPipelineConfig:
    """
    Configuration class for setting up the training pipeline.

    This class initializes and manages key paths and settings used during the training process,
    such as directories for artifacts and models, the target column, and the type of ML task.

    Attributes:
        pipeline_name (str): Name of the training pipeline (from constants).
        artifact_name (str): Root directory name for storing all pipeline artifacts.
        artifact_dir (str): Full path to the time-stamped artifact directory.
        model_dir (str): Directory path where the final model will be saved.
        timestamp (str): String timestamp used for directory versioning (format: MM_DD_YYYY_HH_MM_SS).
        target_column (str): Name of the target (label) column in the dataset.
        task_type (str): Type of machine learning task; must be either 'classification' or 'regression'.
        dataset_name (str): Name of dataset that model will be trained on.

    Raises:
        ValueError: If `task_type` is not one of ['classification', 'regression'].
    """

    def __init__(self, schema_file_path: str, timestamp: datetime = None):
        timestamp = timestamp or datetime.now()
        formatted_timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_name = training_pipeline.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name, formatted_timestamp)
        self.model_dir = os.path.join(training_pipeline.FINAL_MODEL_DIR)
        self.timestamp: str = formatted_timestamp

        self.schema_file_path = schema_file_path

        # Validate schema file
        if not schema_file_path.endswith((".yaml", ".yml")):
            raise ValueError(
                f"Schema file must end with .yaml or .yml, but got: {schema_file_path}"
            )
        # Check the schema file exists
        if not os.path.exists(self.schema_file_path):
            raise FileNotFoundError(
                f"Schema file not found at: {self.schema_file_path}"
            )

        # Open schema file
        schema = read_schema_file(schema_filepath=self.schema_file_path)

        self.target_column = schema.get("target_column")
        self.task_type = schema.get("task_type")
        self.dataset_name = schema.get("DB_collection_name")

        if not self.target_column:
            raise ValueError("Missing or empty 'target_column' in schema.")

        if not self.dataset_name:
            raise ValueError("Missing or empty 'DB_collection_name' in schema.")

        if self.task_type not in ["classification", "regression"]:
            raise ValueError(
                f'Task type must be "classification" or "regression", but got: {self.task_type}'
            )


class DataIngestionConfig:
    """
    Configuration class for setting up the training pipeline.

    This class initializes and manages key paths and settings used during the training process,
    such as directories for artifacts and models, the target column, and the type of ML task.

    Attributes:
        pipeline_name (str): Name of the training pipeline (from constants).
        artifact_name (str): Root directory name for storing all pipeline artifacts.
        artifact_dir (str): Full path to the time-stamped artifact directory.
        model_dir (str): Directory path where the final model will be saved.
        timestamp (str): String timestamp used for directory versioning (format: MM_DD_YYYY_HH_MM_SS).
        schema_file_path (str): Path to the YAML/YML schema file.
        target_column (str): Name of the target (label) column in the dataset.
        task_type (str): Type of machine learning task; must be either 'classification' or 'regression'.

    Raises:
        FileNotFoundError: If the schema file does not exist.
        ValueError: If `schema_file_path` does not end with .yaml or .yml.
        ValueError: If `task_type` is not one of ['classification', 'regression'].
        ValueError: If `target_column` is missing in the schema.
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME,
        )
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline.FILE_NAME,
        )
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TRAIN_FILE_NAME,
        )
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TEST_FILE_NAME,
        )
        self.train_test_split_ratio: float = (
            training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        )

        schema = read_schema_file(
            schema_filepath=training_pipeline_config.schema_file_path
        )
        self.database_name: str = schema["DB_name"]
        self.collection_name: str = schema["DB_collection_name"]


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

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_VALIDATION_DIR_NAME,
        )
        self.valid_data_dir: str = os.path.join(
            self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALID_DIR
        )
        self.invalid_data_dir: str = os.path.join(
            self.data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_DIR
        )
        self.valid_train_file_path: str = os.path.join(
            self.valid_data_dir, training_pipeline.TRAIN_FILE_NAME
        )
        self.valid_test_file_path: str = os.path.join(
            self.valid_data_dir, training_pipeline.TEST_FILE_NAME
        )
        self.invalid_train_file_path: str = os.path.join(
            self.invalid_data_dir, training_pipeline.TRAIN_FILE_NAME
        )
        self.invalid_test_file_path: str = os.path.join(
            self.invalid_data_dir, training_pipeline.TEST_FILE_NAME
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

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_TRANSFORMATION_DIR_NAME,
        )
        self.transformed_train_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TRAIN_FILE_NAME.replace("csv", "npy"),
        )
        self.transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TEST_FILE_NAME.replace("csv", "npy"),
        )
        self.transformed_object_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            training_pipeline.PREPROCESSING_OBJECT_FILE_NAME,
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
        dataset_name (str): name of dataset that model is trained on.
        model_type (str): Type of model to be trained, either 'classification' or 'regression'.
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.dataset_name = training_pipeline_config.dataset_name
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.MODEL_TRAINER_DIR_NAME,
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir,
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR,
            training_pipeline.MODEL_FILE_NAME,
        )
        self.final_trained_model_file_path: str = os.path.join(
            training_pipeline.FINAL_MODEL_DIR,
            self.dataset_name,
            training_pipeline.MODEL_FILE_NAME,
        )
        self.expected_accuracy: float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold = (
            training_pipeline.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD
        )
        self.model_type = training_pipeline_config.task_type
