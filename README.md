# Nano-AutoML-Cloud: An End-to-End, Cloud-Based, Production-Ready Machine Learning Pipeline
**Nano AutoML Cloud** is a compact, end-to-end machine learning pipeline designed for tabular data, supporting both classification and regression tasks. The project is production-ready and optimized for deployment on **cloud platforms** such as AWS. It adheres to modern **MLOps practices** and **CI/CD** standards, ensuring scalability, maintainability, and seamless integration into real-world systems. It is easy to expand, making it a flexible solution for diverse machine learning applications.

<img src="README-assets/end-to-end-training-pipeline.png" alt="Nano-AutoML End-to-end-Machine-Learning-Data-Science-Training-Pipeline" style="width:100%;"/>


## 🔧 Features and Components

- **Modular, Expandable Pipeline**: Includes clearly defined components—**Data Ingestion**, **Data Validation**, **Data Transformation**, **Model Training**, **Model Evaluation**, and **Model Deployment**—designed for easy customization and scalability.
- **Cloud-Native Deployment**: Built-in **FastAPI** endpoints enable real-time and batch prediction, as well as training, directly on cloud platforms like **AWS**.
- **Artifact Management**: Each pipeline component stores its output artifacts (e.g., models, metrics, transformed data) in **AWS S3**, ensuring traceability and reproducibility.
- **Centralized Logging**: Comprehensive logs are stored in AWS for monitoring, debugging, and auditing.
- **Experiment Tracking**: Integrated with **MLflow** for tracking experiments, model versions, and performance metrics in a cloud-based setup.
- **Multi-Dataset Support**: Capable of handling multiple datasets concurrently for both training and batch inference tasks.
- **MLOps & CI/CD Compliance**: Follows best practices in **MLOps** and modern **CI/CD pipelines** to ensure robust, maintainable, and automated workflows.
- **Well-Documented**: Clear and thorough documentation to support ease of use, customization, and onboarding for new users or teams.


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

## Data and Schema Files


## Contents of `.env`

- Link to your MongoDB, something similar to this
```
MONGO_DB_URL="mongodb+srv://<USER>:<password>@cluster0.yrugr0p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
```
- 

## Example Datasets and Results

Mohammad, R. & McCluskey, L. (2012). Phishing Websites [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51W2X.