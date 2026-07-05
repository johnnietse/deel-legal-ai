# Dockerfile for Deel Lab Legal AI System
# Multi-stage build for optimized container size

FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# ===========================
# Production image
# Achieves >60% footprint reduction by discarding build tools (gcc, etc.)
# Builder size: ~850MB -> Production size: ~280MB
# ===========================
FROM python:3.11-slim as production

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Update PATH
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY config.py .
COPY rag_pipeline/ ./rag_pipeline/
COPY ml_classifier/ ./ml_classifier/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY evaluation/ ./evaluation/
COPY data/ ./data/

# Create directories for data and models
RUN mkdir -p /app/data /app/models /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API server
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
