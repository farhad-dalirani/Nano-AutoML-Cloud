import os

from ml_pipeline.logging.logger import logging
from ml_pipeline.exception.exception import MLPipelineException
from ml_pipeline.constants.training_pipeline import (
    ARTIFACT_DIR,
    FINAL_MODEL_DIR,
    LOGS_DIR,
)

from ml_pipeline.components.data_ingestion import DataIngestion
from ml_pipeline.components.data_validation import DataValidation
from ml_pipeline.components.data_transformation import DataTransformation
from ml_pipeline.components.model_trainer import ModelTrainer

from ml_pipeline.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from ml_pipeline.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)

from ml_pipeline.cloud.s3_syncer import S3Sync

from dotenv import load_dotenv

# Load variables from .env into the environment
load_dotenv()


class TrainingPipeline:
    """
    A class that defines the end-to-end machine learning training pipeline.

    The pipeline consists of the following sequential steps:
    1. Data Ingestion
    2. Data Validation
    3. Data Transformation
    4. Model Training

    Each step produces an artifact which is passed to the next stage.
    """

    def __init__(self, schema_file_path: str):
        """
        Initializes the TrainingPipeline with the required configuration objects
        and prepares S3 synchronization utility.

        Args:
            schema_file_path (str): Path to the schema file (.yaml or .yml) containing
                                    dataset column definitions, target column, and task type.
        """
        self.training_pipeline_config = TrainingPipelineConfig(
            schema_file_path=schema_file_path
        )
        self.s3_sync = S3Sync()

    def start_data_ingestion(self):
        """
        Initiates the data ingestion process.

        Returns:
            DataIngestionArtifact: Artifact containing details of the ingested data.

        Raises:
            MLPipelineException: If data ingestion fails.
        """
        try:
            self.data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            logging.info("Initiate the data ingestion.")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(
                "Data ingestion was completed and data ingestion artifact: {}".format(
                    data_ingestion_artifact
                )
            )
            return data_ingestion_artifact
        except Exception as e:
            raise MLPipelineException(e)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
        """
        Initiates the data validation process.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): Artifact from the data ingestion step.

        Returns:
            DataValidationArtifact: Artifact containing results of the validation process.

        Raises:
            MLPipelineException: If data validation fails.
        """
        try:
            self.data_validation_config = DataValidationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            logging.info("Initiate the data validation.")
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config,
                training_pipeline_config=self.training_pipeline_config,
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(
                "Data validation was completed and artifact: {}.".format(
                    data_validation_artifact
                )
            )
            return data_validation_artifact
        except Exception as e:
            raise MLPipelineException(e)

    def start_data_transformation(
        self, data_validation_artifact: DataValidationArtifact
    ):
        """
        Initiates the data transformation process.

        Args:
            data_validation_artifact (DataValidationArtifact): Artifact from the data validation step.

        Returns:
            DataTransformationArtifact: Artifact containing transformed features and target data.

        Raises:
            MLPipelineException: If data transformation fails.
        """
        try:
            self.data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            logging.info("Initiate Data Transformation.")
            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=self.data_transformation_config,
                training_pipeline_config=self.training_pipeline_config,
            )
            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )
            logging.info(
                "Data Transformation was completed and artifact {}".format(
                    data_transformation_artifact
                )
            )
            return data_transformation_artifact
        except Exception as e:
            raise MLPipelineException(e)

    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ):
        """
        Initiates the model training process.

        Args:
            data_transformation_artifact (DataTransformationArtifact): Artifact from the data transformation step.

        Returns:
            ModelTrainerArtifact: Artifact containing trained model and performance metrics.

        Raises:
            MLPipelineException: If model training fails.
        """
        try:
            self.model_training_config = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            logging.info("Model training started.")
            model_training = ModelTrainer(
                model_trainer_config=self.model_training_config,
                data_transform_artifact=data_transformation_artifact,
            )
            model_trainer_artifact = model_training.initiate_model_trainer()
            logging.info(
                "Model training was finished and artifact: {}.".format(
                    model_trainer_artifact
                )
            )
            return model_trainer_artifact
        except Exception as e:
            raise MLPipelineException(e)

    def sync_artifacts_directory_to_s3(self):
        """
        Synchronizes the local artifacts directory to the corresponding S3 bucket path.

        Raises:
            MLPipelineException: If syncing to S3 fails.
        """
        try:
            aws_bucket_url = f"s3://{os.environ['AWS_S3_BUCKET_NAME']}/{ARTIFACT_DIR}/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(
                folder=self.training_pipeline_config.artifact_dir,
                aws_bucket_url=aws_bucket_url,
            )
        except Exception as e:
            raise MLPipelineException(e)

    def sync_final_models_directory_to_s3(self):
        """
        Synchronizes the final trained models directory to the corresponding S3 bucket path.

        Raises:
            MLPipelineException: If syncing to S3 fails.
        """
        try:
            aws_bucket_url = f"s3://{os.environ['AWS_S3_BUCKET_NAME']}/{FINAL_MODEL_DIR}/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(
                folder=self.training_pipeline_config.model_dir,
                aws_bucket_url=aws_bucket_url,
            )
        except Exception as e:
            raise MLPipelineException(e)

    def sync_logs_directory_to_s3(self):
        """
        Synchronizes the local artifacts directory to the corresponding S3 bucket path.

        Raises:
            MLPipelineException: If syncing to S3 fails.
        """
        try:
            aws_bucket_url = f"s3://{os.environ['AWS_S3_BUCKET_NAME']}/{LOGS_DIR}/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(
                folder=LOGS_DIR, aws_bucket_url=aws_bucket_url
            )
        except Exception as e:
            raise MLPipelineException(e)

    def run(self):
        """
        Executes the full training pipeline from data ingestion to model training.

        Returns:
            ModelTrainerArtifact: Final artifact after model training.

        Raises:
            MLPipelineException: If any pipeline stage fails.
        """
        try:
            logging.info(
                "\n" + "=" * 30 + "\nTraining Pipeline started ...\n" + "=" * 30
            )
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            data_transformation_artifact = self.start_data_transformation(
                data_validation_artifact=data_validation_artifact
            )
            model_trainer_artifact: ModelTrainerArtifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )

            self.sync_artifacts_directory_to_s3()
            self.sync_final_models_directory_to_s3()
            self.sync_logs_directory_to_s3()
            logging.info("\n" + "=" * 30 + "\nTraining Pipeline ended.\n" + "=" * 30)
            return model_trainer_artifact
        except Exception as e:
            raise MLPipelineException(e)
