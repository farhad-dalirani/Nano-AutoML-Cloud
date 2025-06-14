# End-to-End MLOps Pipeline Framework


## Overview
<img src="README-assets/end-to-end-training-pipeline.png" alt="End-to-end-Machine-Learning-Data-Science-Training-Pipeline" style="width:100%;"/>


### Features


## Installation and Usage

1. (Optional) Create and activate a virtual environment:
    
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2. Install the required dependencies:
    
    ```
    pip install -r requirements.txt
    ```

3. Configure the environment by completing the required fields in the `.env` file, using `.env-example` as a reference.


4. Initialize the database by running `push_data.py` once to insert the target dataset into MongoDB. Once this is done, as long as the data remains in the dataset, you don't need to run it again for subsequent runs:

    ```
    python3 push_data.py
    ```

## Contents of `.env`

- Link to your MongoDB, something similar to this
```
MONGO_DB_URL="mongodb+srv://<USER>:<password>@cluster0.yrugr0p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
```
- 

## Dataset

Mohammad, R. & McCluskey, L. (2012). Phishing Websites [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51W2X.