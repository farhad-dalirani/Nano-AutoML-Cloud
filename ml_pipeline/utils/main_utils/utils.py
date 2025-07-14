import os
import sys
import pickle
import yaml
import numpy as np
from typing import Dict

from sklearn.metrics import r2_score, f1_score
from sklearn.model_selection import GridSearchCV

from ml_pipeline.logging.logger import logging
from ml_pipeline.exception.exception import MLPipelineException
from ml_pipeline.constants.training_pipeline import SCHEMA_DIR


def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its contents as a dictionary.

    Args:
        file_path (str): The path to the YAML file to be read.

    Returns:
        dict: Parsed contents of the YAML file.

    Raises:
        MLPipelineException: If the file cannot be read or parsed.
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise MLPipelineException(error_message=e)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Writes the given content to a YAML file at the specified path.

    If `replace` is True and the file already exists, it will be deleted before writing.
    The directory path will be created if it does not exist.

    Args:
        file_path (str): The path where the YAML file will be written.
        content (object): The content to serialize and write to the YAML file.
        replace (bool, optional): If True, replaces the existing file if it exists. Defaults to False.

    Raises:
        MLPipelineException: If any error occurs during the file operation or YAML dumping.
    """
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)

    except Exception as e:
        raise MLPipelineException(e)


def save_object(file_path: str, obj: object) -> None:
    """
    Serializes and saves a Python object to the specified file path using the pickle module.

    Args:
        file_path (str): The full path (including file name) where the object should be saved.
        obj (object): The Python object to serialize and save.

    Raises:
        MLPipelineException: If an error occurs during directory creation or object serialization.
    """
    try:
        logging.info("Entered the save_oject method of MainUtils class.")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_oject method of MainUtils class.")
    except Exception as e:
        raise MLPipelineException(e)


def load_object(file_path: str) -> object:
    """
    Loads a Python object from a pickle file.

    Args:
        file_path (str): The path to the pickle file to be loaded.

    Returns:
        object: The Python object loaded from the file.

    Raises:
        MLPipelineException: If the file does not exist or an error occurs during loading.
    """
    try:
        if not os.path.exists(file_path):
            raise Exception("The file {} does not exist.".format(file_path))
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise MLPipelineException(e)


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Saves a NumPy array to a specified file path in binary `.npy` format.

    Args:
        file_path (str): The full path (including file name) where the array should be saved.
        array (np.array): The NumPy array to be saved.

    Raises:
        MLPipelineException: If there is any issue during directory creation or file writing.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise MLPipelineException(e)


def load_numpy_array_data(file_path: str) -> np.array:
    """
    Loads a NumPy array from a `.npy` file.

    Args:
        file_path (str): The path to the `.npy` file containing the NumPy array.

    Returns:
        np.array: The NumPy array loaded from the file.

    Raises:
        MLPipelineException: If the file does not exist or an error occurs during loading.
    """
    try:
        if not os.path.exists(file_path):
            raise Exception("The file {} does not exist.".format(file_path))
        return np.load(file_path)
    except Exception as e:
        raise MLPipelineException(e)


def validate_schema_config(config: dict) -> bool:
    """
    Validates the structure and constraints of a data schema YAML configuration dictionary
    used for a machine learning project pipeline.

    This function checks the following:
    1. All required top-level keys are present.
    2. The 'columns' key contains a list of single-key dictionaries with type 'int64'.
    3. The 'target_column' exists in the defined columns.
    4. If data transformation for the target class is enabled, the mapping must be provided as a dictionary.
    5. All columns in 'numerical_columns' exist in the 'columns' list and do not include the target column.
    6. All columns in 'categorical_columns' exist in the 'columns' list and do not include the target column.
    7. The 'task_type' is either 'classification' or 'regression'.

    Parameters:
    ----------
    config : dict
        A dictionary loaded from a YAML file containing project configuration.

    Returns:
    -------
    bool
        Returns True if all checks pass. Raises ValueError or TypeError if any check fails.

    Raises:
    ------
    ValueError:
        If any required key is missing, the format of columns is invalid,
        or column constraints are violated.

    TypeError:
        If column types or target class mappings are not in the expected data types.
    """
    required_keys = [
        "DB_name",
        "DB_collection_name",
        "task_type",
        "columns",
        "target_column",
        "DATA_TRANSFORMATION_ENABLE_TARGET_CLASS_MAPPING",
        "DATA_TRANSFORMATION_TARGET_CLASS_MAPPING",
        "numerical_columns",
        "categorical_columns",
        "ignore_columns",
    ]
    try:
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in schema file: '{key}'")

        # Validate 'columns'
        if not isinstance(config["columns"], list):
            raise TypeError("'columns' must be a list of single-key dictionaries")

        column_dict = {}
        for item in config["columns"]:
            if not isinstance(item, dict) or len(item) != 1:
                raise ValueError(f"Invalid column format: {item}")
            col_name, col_type = list(item.items())[0]
            column_dict[col_name] = col_type

        # Validate 'target_column' exists in columns
        target_column = config["target_column"]
        if target_column not in column_dict:
            raise ValueError(f"target_column '{target_column}' not found in columns")

        # If transformation is enabled, mapping must be present
        if config["DATA_TRANSFORMATION_ENABLE_TARGET_CLASS_MAPPING"]:
            if not isinstance(config["DATA_TRANSFORMATION_TARGET_CLASS_MAPPING"], dict):
                raise TypeError(
                    "DATA_TRANSFORMATION_TARGET_CLASS_MAPPING must be a dictionary"
                )

        # Validate numerical_columns exist and do not include target_column
        numerical_columns = config.get("numerical_columns")
        if not isinstance(numerical_columns, list):
            numerical_columns = []
        for col in numerical_columns:
            if col not in column_dict:
                raise ValueError(f"Numerical column '{col}' not found in columns")
            if col == target_column:
                raise ValueError(
                    f"target_column '{target_column}' should not be in numerical_columns"
                )

        # Validate categorical_columns exist and do not include target_column
        categorical_columns = config.get("categorical_columns")
        if not isinstance(categorical_columns, list):
            categorical_columns = []
        for col in categorical_columns:
            if col not in column_dict:
                raise ValueError(f"Categorical column '{col}' not found in columns")
            if col == target_column:
                raise ValueError(
                    f"target_column '{target_column}' should not be in categorical_columns"
                )

        if len(numerical_columns) == 0 and len(categorical_columns) == 0:
            raise ValueError(
                f"Both numerical_columns and categorical_columns can not be empty."
            )

        # Validate task_type
        if config["task_type"] not in {"classification", "regression"}:
            raise ValueError(
                "task_type must be either 'classification' or 'regression'"
            )
    except Exception as e:
        raise MLPipelineException(e)

    return True


def read_schema_file(schema_filepath: str) -> dict:
    """
    Reads a YAML schema file, validate its content structure, and returns its contents as a dictionary.

    Parameters:
    ----------
    schema_filepath : str
        The full path to the YAML (.yaml or .yml) schema file.

    Returns:
    -------
    dict
        Parsed contents of the YAML file.

    Raises:
    ------
    Exception
        If the file cannot be read or parsed, or if the file extension is unsupported.
    """
    if schema_filepath.endswith((".yaml", ".yml")):
        try:
            schema = read_yaml_file(file_path=schema_filepath)
            validate_schema_config(config=schema)
            return schema
        except Exception as e:
            raise Exception(f"Failed to read or parse '{schema_filepath}': {str(e)}")
    else:
        raise Exception(
            f"Unsupported file extension for '{schema_filepath}'. Expected a .yaml or .yml file."
        )


def get_dataset_schema_mapping() -> Dict[str, str]:
    """
    Scans the schema directory for YAML/YML files and maps each dataset
    (identified by 'DB_collection_name') to its corresponding schema filename.

    Returns:
        Dict[str, str]: Dictionary mapping dataset names (DB_collection_name)
                        to schema filenames.

    Raises:
        Exception: If any schema file is unreadable or missing the required field.
    """
    collection_names = {}

    for filename in os.listdir(SCHEMA_DIR):
        if filename.endswith((".yaml", ".yml")):
            filepath = os.path.join(SCHEMA_DIR, filename)
            try:
                content = read_schema_file(schema_filepath=filepath)
                collection_name = content.get("DB_collection_name")
                if collection_name:
                    collection_names[collection_name] = filename
            except Exception as e:
                raise Exception(f"Failed to read or parse '{filename}': {str(e)}")

    return collection_names


def evaluate_models(
    X_train, y_train, X_test, y_test, models, params, task_type="classification"
):
    """
    Evaluates multiple machine learning models using GridSearchCV and returns a performance report.

    This function performs hyperparameter tuning using GridSearchCV for each model,
    fits the best estimator to the training data, and evaluates the performance on test data.
    It supports both classification and regression tasks.

    Parameters:
    ----------
    X_train : array-like or pandas.DataFrame
        Feature matrix for training.

    y_train : array-like or pandas.Series
        Target vector for training.

    X_test : array-like or pandas.DataFrame
        Feature matrix for testing.

    y_test : array-like or pandas.Series
        Target vector for testing.

    models : dict
        A dictionary of model name and model object pairs. Example: {"RandomForest": RandomForestClassifier()}.

    params : dict
        A dictionary where keys are model names and values are dictionaries of hyperparameters
        to be tuned via GridSearchCV. Example: {"RandomForest": {"n_estimators": [100, 200]}}.

    task_type : str, default="classification"
        The type of task to perform. Supported values are:
        - "classification": Uses weighted F1 score for evaluation.
        - "regression": Uses R² score for evaluation.

    Returns:
    -------
    report : dict
        A dictionary containing model names as keys and their respective performance scores as values.

    Raises:
    ------
    MLPipelineException
        If any error occurs during model evaluation or hyperparameter tuning.

    ValueError
        If an unsupported task_type is provided.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = params[model_name]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            if task_type == "regression":
                test_model_score = r2_score(y_test, y_test_pred)
            elif task_type == "classification":
                test_model_score = f1_score(y_test, y_test_pred, average="weighted")
            else:
                raise ValueError(f"Unsupported task_type: {task_type}")

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise MLPipelineException(e)
