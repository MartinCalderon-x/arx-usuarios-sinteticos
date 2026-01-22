# ML Service Dockerfile
# =====================
# Multi-stage build for optimized image size

# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch CPU version (smaller than GPU version)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install DeepGaze from GitHub for latest version
RUN pip install --no-cache-dir git+https://github.com/matthias-k/DeepGaze.git


# Stage 2: Runtime image
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime dependencies only
# Note: libgl1-mesa-glx was replaced by libgl1 in Debian Trixie
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY app/ ./app/

# Create models directory and download centerbias
RUN mkdir -p /app/models && \
    curl -L -o /app/models/centerbias_mit1003.npy \
    https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ML_DEVICE=cpu
ENV ML_DEEPGAZE_MODEL=deepgaze3
ENV ML_MODEL_CACHE_DIR=/app/models

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/health', timeout=5).raise_for_status()"

# Run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
