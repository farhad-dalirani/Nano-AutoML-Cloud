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
- **Automated Model Selection**: Trains and evaluates multiple different kind of models using hyperparameter search, selecting the best-performing model automatically.
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

3. Create a MongoDB database and add its URL to the `.env` file. You can use a free MongoDB hosting service from the [MongoDB official website](https://www.mongodb.com/). Upload your tabular dataset as a collection. The `push_data.py` script can upload three example datasets to your database. Run the script once to insert the data: `python3 push_data.py`. You should not run it again unless the data is removed. If you're using your own dataset instead of the examples, create a new schema based on the ones in the `data_schema` folder.

4. Install `AWS CLI`, see [AWS instructions](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

5. Configure the environment by completing the required fields in the `.env` file, using `.env-example` as a reference and following instructions.

6. Configure AWS:
    - Create an AWS user via `IAM` in the AWS dashboard with the appropriate permissions. A general — but not recommended — choice is `AdministratorAccess`.
    - Once the user is created, go to the user's settings, then under `Security Credentials`, create a `CLI access key`.
    - According to the provided `.env` file, enter the Access Key and Secret Access Key from the previous step into your .env file. It is also recommended to run `aws configure` in the terminal and use the same Access Key and Secret Access Key to initialize the AWS CLI. Also add them to Github Secrets.
    - Create an S3 bucket Via `S3` in AWS dashboard, and put its name into `.env` file according to provided example.
    - Create an `AWS ECR` to privately host the dockerfile. Then, accoding to section `.env and Github Secret`, add the necessary secrets to your GitHub repository.
    - In the AWS panel, go to `EC2`and create an Ubuntu instance. Then, choose an appropriate instance type, such as  `t2.medium`. Afterwards, from the EC2 panel, connect to the running instance's terminal. Once connected, enter the following commands in the terminal, which, generally speaking, is the proper way to install Docker on an EC2 instance:
      ```
      sudo apt-get update -y
      sudo apt-get upgrade
      curl -fsSL https://get.docker.com -o get-docker.sh
      sudo sh get-docker.sh
      sudo usermod -aG docker ubuntu
      newgrp docker
      ```
    - Then go in to Github repository, Setting, Actions, Runners. Create a Linux `Runner`. It will give some setup commands, copy and past in EC2 terminal. If it asked `Enter the name of runner: [press Enter for ip-...-..-..-..]` enter `self-hosted`. Also, later, if you find that the action runner has stopped on your instance, you can manually run it by
      ```
       cd ~/actions-runner
       ./run.sh 
      ```
    - Furthermore, through the your AWS EC2 instance panel, navigate to the Security, Security Group, then to Edit Inbound Rules, and make sure to add a rule that allows inbound traffic on port 8000:
      ```
      | Type       | Protocol | Port Range | Source    |
      | ---------- | -------- | ---------- | --------- |
      | Custom TCP | TCP      | 8000       | 0.0.0.0/0 |

      ```

7. Go to your EC2 instance URL, something like `http://ec2-..-...-...-....compute-1.amazonaws.com:8000/docs`, to view the API endpoints for training and batch prediction on datasets. With the current settings, make sure to use `http`.


## .env and Github Secret

- Structure of `.env`:
```
MONGO_DB_URL="fill here"

AWS_ACCESS_KEY_ID="fill here"
AWS_SECRET_ACCESS_KEY="fill here"
AWS_DEFAULT_REGION = "fill here"

AWS_S3_BUCKET_NAME="fill here"

AWS_ECR_LOGIN_URI="fill here"
ECR_REPOSITORY_NAME="fill here"
```

- Add these secrets to Github, obtain the values from the `Installation and Usage` sections:
```
MONGO_DB_URL
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
AWS_DEFAULT_REGION
AWS_S3_BUCKET_NAME
AWS_ECR_LOGIN_URI
ECR_REPOSITORY_NAME
```


## Example Datasets and Results

Mohammad, R. & McCluskey, L. (2012). Phishing Websites [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51W2X.

## Code Structure
