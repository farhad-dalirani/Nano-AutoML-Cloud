import os
import sys
import pickle
import yaml
import numpy as np
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
