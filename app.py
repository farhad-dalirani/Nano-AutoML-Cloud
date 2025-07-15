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
from ml_pipeline.constants.training_pipeline import SCHEMA_DIR
from ml_pipeline.pipeline.training_pipeline import TrainingPipeline
from ml_pipeline.pipeline.batch_prediction import batch_data_prediction
from ml_pipeline.utils.main_utils.utils import get_dataset_schema_mapping

from dotenv import load_dotenv

# URL of MongoDB database that contains datasets
mongo_db_url = os.getenv("MONGO_DB_URL")

# Optional fallback for local development only
if mongo_db_url is None:
    from dotenv import load_dotenv
    load_dotenv()
    mongo_db_url = os.getenv("MONGO_DB_URL")

ca = certifi.where()
client = pymongo.MongoClient(host=mongo_db_url, tlsCAFile=ca)

templates = Jinja2Templates(directory="./templates")

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["authentication"], status_code=status.HTTP_200_OK)
async def index():
    """
    Redirects the root URL to the API documentation page.

    Returns:
        RedirectResponse: Redirects to '/docs' for Swagger UI API docs.
    """
    return RedirectResponse(url="/docs")


@app.get("/dataset-names", status_code=status.HTTP_200_OK)
async def get_dataset_names() -> Dict[str, List[str]]:
    """
    FastAPI endpoint to return a list of available dataset names.

    These dataset names are extracted from schema files located in the schema directory.
    Each name corresponds to a 'DB_collection_name' defined in a YAML/YML schema file.

    Returns:
        dict: A dictionary with a single key 'datasets' that maps to a list of dataset names.

    Raises:
        HTTPException (500): If any schema file is corrupted or missing required fields.
    """
    try:
        collection_map = get_dataset_schema_mapping()
        return {"datasets": list(collection_map.keys())}
    except Exception as e:
        logging.error(f"Error retrieving dataset names: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dataset names due to schema file issues.",
        )


@app.get("/train/{database_name}", status_code=status.HTTP_200_OK)
async def run_train_pipeline(database_name: str):
    """
    Triggers the model training pipeline for a specified dataset.

    This endpoint initiates the training process using a schema configuration associated
    with the given `database_name`. The server looks for a matching entry in the schema
    directory and uses it to configure and execute the training pipeline.

    Args:
        database_name (str): The name of the MongoDB collection (dataset), which must match
                            the `DB_collection_name` defined in one of the schema files.

    Returns:
        dict: A success message indicating the training pipeline ran successfully.

    Raises:
        HTTPException:
            - 404 NOT FOUND: If no schema is found that matches the provided database name.
            - 500 INTERNAL SERVER ERROR: If the training pipeline fails to execute due to an internal error.
    """
    try:
        # Step 1: Load map of {DB_collection_name: schema_filename}
        dataset_mapping = get_dataset_schema_mapping()

        # Step 2: Check if requested DB name exists
        if database_name not in dataset_mapping:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Database name '{database_name}' not found in schema definitions.",
            )

        # Step 3: Use matching schema to run pipeline
        schema_file_path = os.path.join(SCHEMA_DIR, dataset_mapping[database_name])
        train_pipeline = TrainingPipeline(schema_file_path=schema_file_path)
        train_pipeline.run()

        return {
            "message": f"Training pipeline successfully executed for database '{database_name}'."
        }

    except HTTPException as http_err:
        raise http_err  # propagate 404

    except Exception as e:
        logging.error(f"Training failed for '{database_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Training pipeline execution failed due to internal error.",
        )


@app.post("/predict/", status_code=status.HTTP_200_OK)
async def predict(request: Request, database_name: str, file: UploadFile = File(...)):
    """
    Handles batch prediction using a machine learning model on data provided in a CSV file.

    This endpoint accepts a CSV file and a database name, validates the inputs, loads the
    appropriate schema for the selected database, performs predictions using a pre-trained
    machine learning model, and returns an HTML page displaying the original input data
    with an added column for predicted results.

    Args:
        request (Request): FastAPI request object used for rendering templates.
        database_name (str): Name of the target database, used to select the appropriate schema and model.
        file (UploadFile): A CSV file containing input data to be used for batch prediction.

    Raises:
        HTTPException:
            - 400 BAD REQUEST if the uploaded file is not a CSV.
            - 404 NOT FOUND if the provided database name is not defined in the schema mapping.
            - 500 INTERNAL SERVER ERROR if any unexpected error occurs during the prediction process.

    Returns:
        TemplateResponse: An HTML page rendering the input data along with predicted values
                        in a formatted table.
    """
    if not file.filename.endswith(".csv"):
        logging.error("Batch prediction: Only .csv files are accepted")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .csv files are accepted",
        )

    try:
        # Load map of {DB_collection_name: schema_filename}
        dataset_mapping = get_dataset_schema_mapping()
        # Check if requested DB name exists
        if database_name not in dataset_mapping:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Database name '{database_name}' not found in schema definitions.",
            )

        # Read data from file
        df_input = pd.read_csv(file.file)

        # Predict with ML model
        y_pred = batch_data_prediction(
            df_input_data=df_input,
            schema_file_path=os.path.join(SCHEMA_DIR, dataset_mapping[database_name]),
        )

        # Concat the prediction to input
        df_input["predicted_column"] = y_pred

        # Convert the dataframe to HTML and put it in an HTML page
        html_table = df_input.to_html(classes="table table-striped")
        return templates.TemplateResponse(
            "table.html", {"request": request, "table": html_table}
        )

    except Exception as e:
        logging.error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction was failed.",
        )


if __name__ == "__main__":
    # Starts the Uvicorn ASGI server with the FastAPI app,
    # binding it to all available network interfaces (0.0.0.0)
    # so it can be accessed externally (e.g., from outside a Docker container or EC2 instance).
    app_run(app=app, host="0.0.0.0", port=8000)
