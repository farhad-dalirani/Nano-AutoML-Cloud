# Use the official Python 3.10 slim image as the base image (lightweight Debian-based image)
FROM python:3.10-slim-buster

# Set the working directory inside the container to /app
WORKDIR /app

# Copy all files from the current directory on the host to the /app directory in the container
COPY . /app

# Update the package list and install AWS CLI tool (used to interact with AWS services)
RUN apt update -y && apt install awscli -y

# Update package list again (to ensure latest listings) and install Python dependencies from requirements.txt
RUN apt-get update && pip install -r requirements.txt

# Define the command to run the application using Python 3
CMD ["python3", "app.py"]
