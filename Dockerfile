# ============================================
# Usuarios Sintéticos - Production Dockerfile
# ============================================
# Builds the FastAPI backend from product/backend/
# Frontend can be deployed separately or served via CDN
# ============================================

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install backend dependencies
COPY product/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY product/backend/app ./app

# Environment
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

# Run FastAPI with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
