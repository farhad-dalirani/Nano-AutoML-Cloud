import os
import fastapi
from typing import Dict, List
import certifi
import pymongo
import pandas as pd

from fastapi import FastAPI, File, UploadFile, Request, status, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run
from starlette.responses import RedirectResponse

from ml_pipeline.logging.logger import logging
from ml_pipeline.exception.exception import MLPipelineException
from ml_pipeline.pipeline.training_pipeline import TrainingPipeline
from ml_pipeline.pipeline.batch_prediction import batch_data_prediction
from ml_pipeline.utils.main_utils.utils import read_yaml_file
from ml_pipeline.constants.training_pipeline import SCHEMA_DIR
from ml_pipeline.utils.main_utils.utils import get_dataset_schema_mapping

from dotenv import load_dotenv
load_dotenv()

ca = certifi.where()
mongo_db_url = os.getenv("MONGO_DB_URL")

client = pymongo.MongoClient(host=mongo_db_url, tlsCAFile=ca)

templates = Jinja2Templates(directory="./templates")

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get('/', tags=['authentication'], status_code=status.HTTP_200_OK)
async def index():
    """
    Redirects the root URL to the API documentation page.

    Returns:
        RedirectResponse: Redirects to '/docs' for Swagger UI API docs.
    """
    return RedirectResponse(url='/docs')

@app.get('/dataset-names', status_code=status.HTTP_200_OK)
async def get_dataset_names() -> Dict[str, List[str]]:
    """
    FastAPI endpoint to return a list of available dataset names.

    These dataset names are extracted from schema files located in the schema directory.
    Each name corresponds to a 'DB_collection_name' defined in a YAML/YML schema file.

    Returns:
        dict: A dictionary with a single key 'datasets' that maps to a list of dataset names.

              Example:
              {
                  "datasets": ["bike_sharing_daily", "phishing_sites"]
              }

    Raises:
        HTTPException (500): If any schema file is corrupted or missing required fields.
    """
    try:
        collection_map = get_dataset_schema_mapping(SCHEMA_DIR)
        return {"datasets": list(collection_map.keys())}
    except Exception as e:
        logging.error(f"Error retrieving dataset names: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dataset names due to schema file issues."
        )
    
@app.get('/train/{database_name}', status_code=status.HTTP_200_OK)
async def run_train_pipeline(database_name: str):
    """
    Endpoint to trigger the training pipeline for a given database (dataset).

    The server scans the schema directory for all available schema files and checks
    if any contain the given `database_name` as the value of `DB_collection_name`.

    If found, the corresponding schema file is used to initiate the training pipeline.

    Args:
        database_name (str): The name of the MongoDB collection (dataset) as defined
                             in the `DB_collection_name` field of a schema file.

    Returns:
        dict: A message confirming successful training, or an error if not found or failed.

    Raises:
        HTTPException (404): If the provided database name does not match any known schema.
        HTTPException (500): If the training pipeline fails to run.
    """
    try:
        # Step 1: Load map of {DB_collection_name: schema_filename}
        dataset_mapping = get_dataset_schema_mapping(SCHEMA_DIR)

        # Step 2: Check if requested DB name exists
        if database_name not in dataset_mapping:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Database name '{database_name}' not found in schema definitions."
            )

        # Step 3: Use matching schema to run pipeline
        schema_file_path = os.path.join(SCHEMA_DIR, dataset_mapping[database_name])
        train_pipeline = TrainingPipeline(schema_file_path=schema_file_path)
        train_pipeline.run()

        return {"message": f"Training pipeline successfully executed for database '{database_name}'."}

    except HTTPException as http_err:
        raise http_err  # propagate 404

    except Exception as e:
        logging.error(f"Training failed for '{database_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Training pipeline execution failed due to internal error."
        )
    
@app.post('/predict', status_code=status.HTTP_200_OK)
async def predict(request: Request, file: UploadFile=File(...)):  
    """
    Performs batch prediction on an uploaded CSV file.

    Args:
        request (Request): FastAPI request object.
        file (UploadFile): Uploaded CSV file containing input data for prediction.

    Raises:
        HTTPException: If the uploaded file is not a CSV.
        HTTPException: If batch prediction process fails.

    Returns:
        TemplateResponse: Renders an HTML page displaying the input data
                          along with predicted results in a table.
    """
    if not file.filename.endswith(".csv"):
        logging.error("Batch prediction: Only .csv files are accepted")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .csv files are accepted"
        )
    
    try:
        # Read data from file
        df_input = pd.read_csv(file.file)

        # Check with schema

        # Predict with ML model
        y_pred = batch_data_prediction(df_input_data=df_input)

        # Concat the prediction to input
        df_input['predicted_column'] = y_pred

        # Convert the dataframe to HTML and put it in an HTML page
        html_table = df_input.to_html(classes='table table-striped')
        return templates.TemplateResponse("table.html", {"request": request, "table": html_table})

    except Exception as e:
        logging.error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Batch prediction was failed."
        )

if __name__ == '__main__':
    app_run(app=app, host="localhost", port=8000)