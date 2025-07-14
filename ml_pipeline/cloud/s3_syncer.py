import os
import subprocess
import logging
from dotenv import load_dotenv

# Load variables from .env into the environment
load_dotenv()


class S3Sync:
    def __init__(self):
        # Your check part:
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

        if not aws_access_key_id or not aws_secret_access_key or not aws_region:
            raise EnvironmentError(
                "AWS credentials not found in environment, add them to `.env` file."
            )

    def sync_folder_to_s3(self, folder, aws_bucket_url):
        """
        Syncs a local folder to an AWS S3 bucket using the AWS CLI.

        Args:
            folder (str): Path to the local folder to upload.
            aws_bucket_url (str): S3 bucket URL (e.g., s3://my-bucket/folder).

        Logs:
            Success or error messages using the logging module.
        """
        try:
            subprocess.run(["aws", "s3", "sync", folder, aws_bucket_url], check=True)
            logging.info(f"Successfully synced {folder} to {aws_bucket_url}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error syncing to S3: {e}")

    def sync_folder_from_s3(self, folder, aws_bucket_url):
        """
        Syncs the contents of an AWS S3 bucket to a local folder using the AWS CLI.

        Args:
            folder (str): Path to the local folder where files will be downloaded.
            aws_bucket_url (str): S3 bucket URL (e.g., s3://my-bucket/folder).

        Logs:
            Success or error messages using the logging module.
        """
        try:
            subprocess.run(["aws", "s3", "sync", aws_bucket_url, folder], check=True)
            logging.info(f"Successfully synced {aws_bucket_url} to {folder}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error syncing from S3: {e}")
