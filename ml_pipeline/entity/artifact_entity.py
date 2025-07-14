from dataclasses import dataclass
from typing import Optional


@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str


@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str


@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float


@dataclass
class RegressionMetricArtifact:
    mse: float
    mae: float
    r2_score: float


@dataclass
class ModelMetricArtifact:
    classification_metrics: Optional[ClassificationMetricArtifact] = None
    regression_metrics: Optional[RegressionMetricArtifact] = None


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: ModelMetricArtifact
    test_metric_artifact: ModelMetricArtifact
