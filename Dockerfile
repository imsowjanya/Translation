# Use the official Python image as the base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app
#new commit changes
# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3  install -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose port 8080 to the outside world
EXPOSE 4545

# Run the Flask application
CMD ["python3", "app1.py"]
