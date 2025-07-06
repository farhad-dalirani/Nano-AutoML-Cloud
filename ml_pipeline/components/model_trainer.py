import os

import mlflow.sklearn

from ml_pipeline.logging.logger import logging
from ml_pipeline.exception.exception import MLPipelineException

from ml_pipeline.entity.artifact_entity import DataTransformationArtifact, ModelMetricArtifact, ModelTrainerArtifact
from ml_pipeline.entity.config_entity import ModelTrainerConfig

from ml_pipeline.utils.ml_utils.model.estimator import MLModel
from ml_pipeline.utils.main_utils.utils import save_object, load_object, load_numpy_array_data, evaluate_models
from ml_pipeline.utils.ml_utils.metric.classification_metric import get_classification_scores

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
import mlflow
from mlflow.models import infer_signature

class ModelTrainer:
    def __init__(self, 
                 model_trainer_config: ModelTrainerConfig, 
                 data_transform_artifact:DataTransformationArtifact):
        """
        Initializes the ModelTrainer with configuration and data transformation artifacts.

        Args:
            model_trainer_config (ModelTrainerConfig): Configuration for model training.
            data_transform_artifact (DataTransformationArtifact): Transformed data and preprocessor object.

        Raises:
            MLPipelineException: If any error occurs during initialization.
        """
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transform_artifact = data_transform_artifact
        except Exception as e:
            raise MLPipelineException(e)
    
    def track_mlflow(self,
                     best_model,
                     train_metrics: ModelMetricArtifact = None,
                     test_metrics: ModelMetricArtifact = None,
                     input_example=None):
        """
        Logs training and testing metrics, along with the model, to MLflow.

        Parameters:
        ----------
        best_model : Any
            The trained model to be logged.

        train_metrics : ModelMetricArtifact, optional
            Evaluation metrics for the training dataset.

        test_metrics : ModelMetricArtifact, optional
            Evaluation metrics for the testing dataset.

        input_example : pd.DataFrame or np.ndarray, optional
        Example input for the model used to infer the signature and for logging.

        Raises:
        ------
        ValueError
            If neither training nor testing metrics are provided.
        NotImplementedError
            If regression metrics logging is requested but not implemented.
        """

        if not train_metrics and not test_metrics:
            raise ValueError("At least one of train_metrics or test_metrics must be provided.")

        with mlflow.start_run():
            if train_metrics:
                self._log_metrics_to_mlflow(train_metrics, prefix="train")

            if test_metrics:
                self._log_metrics_to_mlflow(test_metrics, prefix="test")

            signiture = None
            if input_example is not None:
                try:
                    output_example = best_model.predict(input_example)
                    signiture = infer_signature(model_input=input_example, model_output=output_example)
                except Exception as e:
                    logging.info("Failed to infer model signiture.")

            mlflow.sklearn.log_model(
                sk_model=best_model, 
                artifact_path="model",
                signature=signiture,
                input_example=input_example
            )

    def _log_metrics_to_mlflow(self, metric_artifact: ModelMetricArtifact, prefix: str):
        """
        Logs classification metrics to MLflow with a given prefix.

        Parameters:
        ----------
        metric_artifact : ModelMetricArtifact
            The metrics to log.
        prefix : str
            Prefix to distinguish metrics (e.g., 'train' or 'test').
        """

        if metric_artifact.classification_metrics:
            cm = metric_artifact.classification_metrics
            metrics = {
                f"{prefix}_f1_score": cm.f1_score,
                f"{prefix}_precision_score": cm.precision_score,
                f"{prefix}_recall_score": cm.recall_score,
            }
            mlflow.log_metrics(metrics)

        elif metric_artifact.regression_metrics:
            raise NotImplementedError("Regression metrics logging is not implemented.")
        else:
            raise ValueError("Provided metric artifact contains no valid metrics.")
    

    def train_model(self, X_train, y_train, X_test, y_test):
        """
        Trains a set of machine learning models, evaluates them, selects the best model, 
        and saves it along with the data preprocessor.

        Args:
            X_train (np.ndarray): Training feature data.
            y_train (np.ndarray): Training target labels.
            X_test (np.ndarray): Test feature data.
            y_test (np.ndarray): Test target labels.

        Returns:
            ModelTrainerArtifact: Contains the trained model path and evaluation metrics.

        Raises:
            ValueError: If model type is not supported.
            MLPipelineException: If any error occurs during model training.
        """
        # Define supported models and their hyperparameters for classification tasks
        if self.model_trainer_config.model_type == 'classification':
            models = {
                    "Random Forest": RandomForestClassifier(verbose=1),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                    "Logistic Regression": LogisticRegression(verbose=1),
                    "AdaBoost": AdaBoostClassifier(),
                }
            
            # List of hyper-parameters for tuning
            params={
                "Decision Tree": {
                    'criterion':['gini', 'entropy', 'log_loss'],
                },
                "Random Forest":{
                    'n_estimators': [8,16,32,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Logistic Regression":{},
                "AdaBoost":{
                    'learning_rate':[.1,.01,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }
        elif self.model_trainer_config.model_type == 'regression':
            NotImplementedError
        else:
            raise ValueError('Selected ML Model Type [{}] is not correct!'.format(self.model_trainer_config.model_type))

        # Evaluate all models with given hyperparameters and select the best one
        model_report: dict=evaluate_models(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            models=models,
            params=params    
        )

        # Identify the best-performing model based on evaluation score
        best_model_name, best_model_score = max(model_report.items(), key=lambda item: item[1])
        best_model = models[best_model_name]

        # Generate predictions using the best model for both train and test sets
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
        # Compute evaluation metrics for classification/regression
        if self.model_trainer_config.model_type == 'classification':
            # Evaluation metrics on train
            train_set_metrics = get_classification_scores(y_true=y_train, y_pred=y_train_pred)
            trainset_model_metric_artifact = ModelMetricArtifact(classification_metrics=train_set_metrics)
            # Evaluation metrics on test
            test_set_metrics = get_classification_scores(y_true=y_test, y_pred=y_test_pred)
            testset_model_metric_artifact = ModelMetricArtifact(classification_metrics=test_set_metrics) 
        elif self.model_trainer_config.model_type == 'regression':
            NotImplementedError
        else:
            raise ValueError('Selected ML Model Type [{}] is not correct!'.format(self.model_trainer_config.model_type))

        # Input example: does not anonymize data
        input_examlpe = X_train[0: 5]

        # Track expriment with MLFLOW
        self.track_mlflow(
            best_model=best_model,
            train_metrics = trainset_model_metric_artifact,
            test_metrics = testset_model_metric_artifact,
            input_example = input_examlpe
        )

        # Load the data preprocessor used during feature transformation
        preprocessor = load_object(file_path=self.data_transform_artifact.transformed_object_file_path)

        # Ensure the directory exists before saving the trained model
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        # Create a pipeline object containing both preprocessor and trained model
        final_ml_model = MLModel(preprocessor=preprocessor, model=best_model) 

        # Save the final ML model to disk in its experiment folder
        save_object(
            file_path=self.model_trainer_config.trained_model_file_path, 
            obj=final_ml_model
        )

        # Save the mode and preprocssor in 'final_model'
        final_model_dir_path = os.path.dirname(self.model_trainer_config.final_trained_model_file_path)
        os.makedirs(final_model_dir_path, exist_ok=True)
        save_object(
            file_path=self.model_trainer_config.final_trained_model_file_path, 
            obj=final_ml_model
        )

        # Package model path and evaluation results into an artifact for downstream use
        model_training_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=trainset_model_metric_artifact,
            test_metric_artifact=testset_model_metric_artifact
        )

        logging.info("Model training artifact: {}".format(model_training_artifact))
        return model_training_artifact

    def initiate_model_trainer(self)->ModelTrainerArtifact:
        """
        Initiates the model training process by loading the transformed data,
        training the model, and returning the training artifact.

        Returns:
            ModelTrainerArtifact: The artifact containing the trained model and its evaluation metrics.

        Raises:
            MLPipelineException: If any error occurs during model training initialization.
        """
        try:
            # Retrieve file paths of transformed training and testing datasets
            train_file_path = self.data_transform_artifact.transformed_train_file_path
            test_file_path = self.data_transform_artifact.transformed_test_file_path

            # Load transformed data from NumPy array files
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            # Split the arrays into features (X) and labels (y)
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            # Train the model and generate the model training artifact
            model_training_artifact = self.train_model(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test
            )

            return model_training_artifact

        except Exception as e:
            raise MLPipelineException(e)