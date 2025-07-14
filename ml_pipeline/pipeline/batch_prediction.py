import os
import pandas as pd

from ml_pipeline.utils.ml_utils.model.estimator import MLModel 
from ml_pipeline.constants.training_pipeline import FINAL_MODEL_DIR, MODEL_FILE_NAME
from ml_pipeline.exception.exception import MLPipelineException
from ml_pipeline.utils.main_utils.utils import load_object, read_schema_file

def batch_data_prediction(df_input_data: pd.DataFrame, schema_file_path: str):
    """
    Perform batch predictions using a pre-trained ML model.

    This function loads a trained machine learning model and uses it to make predictions
    on the provided input DataFrame. If the input DataFrame is empty, it returns an empty list.

    Parameters:
    ----------
    df_input_data : pd.DataFrame
        A pandas DataFrame containing the input features for prediction.

    schema_file_path (str): Path to the schema file (.yaml or .yml) containing
                                dataset column definitions, target column, and task type.
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
        
        # Validate schema file
        if not schema_file_path.endswith((".yaml", ".yml")):
            raise ValueError(f"Schema file must end with .yaml or .yml, but got: {schema_file_path}")  
        # Check the schema file exists
        if not os.path.exists(schema_file_path):
            raise FileNotFoundError(f"Schema file not found at: {schema_file_path}")
        # Open schema file
        schema = read_schema_file(schema_filepath=schema_file_path)
        dataset_name = schema.get("DB_collection_name")
    
        # Load ML model and its corresponding data transformer
        ml_model: MLModel = load_object(file_path=os.path.join(FINAL_MODEL_DIR, dataset_name, MODEL_FILE_NAME))
        
        # ML model output for data
        output = ml_model.predict(x=df_input_data)
        
        return output
    except Exception as e:
        raise MLPipelineException(e)