# Production Dockerfile for SPACEIND Green Building Digital Twin Backend
# Compatible with Render, Railway, Fly.io, and AWS/GCP containers

FROM python:3.11-slim

# Install system libraries required by OpenCV (libgl, libglib) and OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend source code and YOLO weights
COPY . .

# Render, Railway & Cloud Run inject $PORT automatically
ENV PORT=8000
ENV HOST=0.0.0.0
EXPOSE 8000

# Start FastAPI server
CMD ["python", "server.py"]
