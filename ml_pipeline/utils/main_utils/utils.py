import os
import sys
import pickle
import yaml
import numpy as np
from typing import Dict

from sklearn.metrics import r2_score, f1_score
from sklearn.model_selection import GridSearchCV

from ml_pipeline.exception.exception import MLPipelineException
from ml_pipeline.logging.logger import logging


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
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise MLPipelineException(error_message=e)
    

def write_yaml_file(file_path:str, content: object, replace: bool = False) -> None:
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


def get_dataset_schema_mapping(schema_dir: str) -> Dict[str, str]:
    """
    Scans the schema directory for YAML/YML files and maps each dataset
    (identified by 'DB_collection_name') to its corresponding schema filename.

    Args:
        schema_dir (str): Path to the directory containing schema files.

    Returns:
        Dict[str, str]: Dictionary mapping dataset names (DB_collection_name)
                        to schema filenames.

    Raises:
        Exception: If any schema file is unreadable or missing the required field.
    """
    collection_names = {}

    for filename in os.listdir(schema_dir):
        if filename.endswith(('.yaml', '.yml')):
            filepath = os.path.join(schema_dir, filename)
            try:
                content = read_yaml_file(file_path=filepath)
                collection_name = content.get("DB_collection_name")
                if collection_name:
                    collection_names[collection_name] = filename
            except Exception as e:
                raise Exception(
                    f"Failed to read or parse '{filename}': {str(e)}"
                )

    return collection_names


def evaluate_models(X_train, y_train, X_test, y_test, models, params, task_type="classification"):
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
                test_model_score = f1_score(y_test, y_test_pred, average='weighted')
            else:
                raise ValueError(f"Unsupported task_type: {task_type}")

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise MLPipelineException(e)