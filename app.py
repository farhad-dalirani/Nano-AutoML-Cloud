import os
import fastapi
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
from ml_pipeline.utils.main_utils.utils import load_object


from ml_pipeline.constants.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from ml_pipeline.constants.training_pipeline import DATA_INGESTION_DATABASE_NAME

from dotenv import load_dotenv
load_dotenv()

ca = certifi.where()
mongo_db_url = os.getenv("MONGO_DB_URL")

client = pymongo.MongoClient(host=mongo_db_url, tlsCAFile=ca)

database = client[DATA_INGESTION_DATABASE_NAME]
collection = client[DATA_INGESTION_COLLECTION_NAME]

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

@app.get('/train', status_code=status.HTTP_200_OK)
async def run_train_pipeline():
    """
    Endpoint to trigger the training pipeline for the machine learning model.

    This endpoint initiates the training process by creating and running
    a TrainingPipeline instance. If the training fails, it raises a 500 HTTP error.

    Raises:
        HTTPException: If the training pipeline fails to run.

    Returns:
        None: On successful training, returns an empty response with status 200.
    """
    try:
        train_pipeline=TrainingPipeline()
        train_pipeline.run()
    except Exception as e:
        logging.error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Initiating train pipeline was failed."
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