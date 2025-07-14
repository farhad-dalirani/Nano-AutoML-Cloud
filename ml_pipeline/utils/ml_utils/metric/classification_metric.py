from ml_pipeline.entity.artifact_entity import (
    ClassificationMetricArtifact,
    RegressionMetricArtifact,
)
from ml_pipeline.exception.exception import MLPipelineException
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def get_classification_scores(y_true, y_pred) -> ClassificationMetricArtifact:
    """
    Calculates common classification metrics: F1 score, precision, and recall.

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as predicted by the classifier.

    Returns:
        ClassificationMetricArtifact: An object containing F1 score, precision score, and recall score.

    Raises:
        MLPipelineException: If an error occurs while calculating the classification metrics.
    """
    try:
        model_f1_score = f1_score(y_true=y_true, y_pred=y_pred, average="weighted")
        model_recall_score = recall_score(
            y_true=y_true, y_pred=y_pred, average="weighted"
        )
        model_precision_score = precision_score(
            y_true=y_true, y_pred=y_pred, average="weighted"
        )

        classification_metrics = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score,
        )
        return classification_metrics
    except Exception as e:
        raise MLPipelineException(e)


def get_regression_scores(y_true, y_pred) -> RegressionMetricArtifact:
    """
    Calculates common regression metrics: Mean Squared Error, Mean Absolute Error, and R² score.

    Args:
        y_true (array-like): Ground truth (actual) target values.
        y_pred (array-like): Predicted target values from the model.

    Returns:
        RegressionMetricArtifact: An object containing MSE, MAE, and R² score.

    Raises:
        MLPipelineException: If an error occurs during metric computation.
    """
    try:
        model_mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
        model_mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        model_r2 = r2_score(y_true=y_true, y_pred=y_pred)

        regression_metrics = RegressionMetricArtifact(
            mse=model_mse, mae=model_mae, r2_score=model_r2
        )
        return regression_metrics
    except Exception as e:
        raise MLPipelineException(e)
