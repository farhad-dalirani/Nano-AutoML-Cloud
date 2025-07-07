import os
import pandas as pd

from ml_pipeline.logging.logger import logging
from ml_pipeline.exception.exception import MLPipelineException
from ml_pipeline.pipeline.training_pipeline import TrainingPipeline
from ml_pipeline.utils.main_utils.utils import load_object
from ml_pipeline.utils.ml_utils.model.estimator import MLModel 

from ml_pipeline.constants.training_pipeline import FINAL_MODEL_DIR, MODEL_FILE_NAME

def batch_data_prediction(df_input_data: pd.DataFrame):
    """
    Perform batch predictions using a pre-trained ML model.

    This function loads a trained machine learning model and uses it to make predictions
    on the provided input DataFrame. If the input DataFrame is empty, it returns an empty list.

    Parameters:
    ----------
    df_input_data : pd.DataFrame
        A pandas DataFrame containing the input features for prediction.

    Returns:
    -------
    list or array-like
        A list or array of model predictions. Returns an empty list if input is empty.

    Raises:
    ------
    MLPipelineException
        If any error occurs during model loading or prediction.
    """
    try:
        # Return empty if input data is empty
        if df_input_data.empty:
            return []
        
        # Load ML model and its corresponding data transformer
        ml_model: MLModel = load_object(file_path=os.path.join(FINAL_MODEL_DIR, MODEL_FILE_NAME))

        # ML model output for data
        output = ml_model.predict(x=df_input_data)
        
        return output
    except Exception as e:
        MLPipelineException(e)