# Use official Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create embeddings while building the image
# RUN python newCreateEmbedding.py

# Expose a port (optional, for web apps)
EXPOSE 8080

# Command to run your application (adjust if needed)
CMD ["python", "main.py"]
