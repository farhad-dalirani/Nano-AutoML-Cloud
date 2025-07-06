import os
import fastapi
import certifi
import pymongo
import pandas as pd

from fastapi import FastAPI, File, UploadFile, Request, status, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run as app_run
from starlette.responses import RedirectResponse


from ml_pipeline.logging.logger import logging
from ml_pipeline.exception.exception import MLPipelineException
from ml_pipeline.pipeline.training_pipeline import TrainingPipeline
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
    return RedirectResponse(url='/docs')

@app.get('/train', status_code=status.HTTP_200_OK)
async def run_train_pipeline():
    try:
        train_pipeline=TrainingPipeline()
        train_pipeline.run()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Initiating train pipeline was failed."
        )
    
if __name__ == '__main__':
    app_run(app=app, host="localhost", port=8000)