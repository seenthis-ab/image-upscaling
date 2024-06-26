# Use an official lightweight Python image.
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get clean && apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-dev

# Copy the requirements file and install Python dependencies.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code
COPY api/ ./api

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Specify the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]