# Use the official Python base image
FROM python:3.11-slim

# Set environment variables to prevent Python 
# from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app

# Expose the port used by the web server
EXPOSE 8000

# Command to run the application
CMD ["fastapi", "run", "inference-service.py"]
