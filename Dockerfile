# Dockerfile — AE-PD Denoising Research Container
# Base: CUDA 12.1 + PyTorch 2.x + all research dependencies
# Usage:
#   docker build -t ae-pd-denoising:latest .
#   docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results \
#              -p 5000:5000 -p 8080:8080 ae-pd-denoising:latest

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    libsndfile1 libgomp1 git curl wget \
    nodejs npm \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Copy requirements first (layer cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Build React frontend if it exists
RUN if [ -d "dashboard/frontend" ]; then \
    cd dashboard/frontend && npm ci && npm run build; \
    fi

# Create directories that are mounted as volumes
RUN mkdir -p data/raw/qlin data/test_sets results checkpoints logs mlruns

# Healthcheck: verify CUDA and imports
HEALTHCHECK --interval=60s --timeout=30s --start-period=30s \
    CMD python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || exit 1

# Expose MLflow UI and dashboard
EXPOSE 5000 8080

# Default: run dashboard + mlflow together
CMD ["sh", "-c", "mlflow ui --host 0.0.0.0 --port 5000 & python3 dashboard/app.py"]
