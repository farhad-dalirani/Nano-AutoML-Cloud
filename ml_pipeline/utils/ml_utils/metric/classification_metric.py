from ml_pipeline.entity.artifact_entity import ClassificationMetricArtifact, RegressionMetricArtifact
from ml_pipeline.exception.exception import MLPipelineException
from sklearn.metrics import f1_score, precision_score, recall_score


def get_classification_scores(y_true, y_pred)->ClassificationMetricArtifact:
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
        model_f1_score = f1_score(y_true=y_true, y_pred=y_pred)
        model_recall_score = recall_score(y_true=y_true, y_pred=y_pred)
        model_precision_score=precision_score(y_true=y_true, y_pred=y_pred)

        classification_metrics = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score
        )
        return classification_metrics
    except Exception as e:
        raise MLPipelineException(e)
    

def get_regression_scores(y_true, y_pred)->RegressionMetricArtifact:
    NotImplementedError