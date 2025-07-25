# Use the official Python 3.10 slim image as the base image (lightweight Debian-based image)
FROM python:3.10-slim-bookworm

# Set the working directory inside the container to /app
WORKDIR /app

# Copy all files from the current directory on the host to the /app directory in the container
COPY . /app

# Update the package list and install AWS CLI tool (used to interact with AWS services)
# Update package list again (to ensure latest listings) and install Python dependencies from requirements.txt
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        awscli \
        supervisor && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt

# Copy supervisord config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Set supervisor as the container entrypoint
CMD ["/usr/bin/supervisord"]