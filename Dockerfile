# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860 \
    API_BASE_URL=http://127.0.0.1:8000

# Install system dependencies for audio decoding and healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root user (UID 1000 for standard cloud container compatibility)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies first for caching efficiency
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create runtime directories for SQLite, uploads, and ChromaDB
RUN mkdir -p data/uploads data/vector_store && \
    chown -R appuser:appuser /app

# Copy application source code
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser frontend/ ./frontend/
COPY --chown=appuser:appuser start.sh ./start.sh

# Ensure startup script is executable
RUN chmod +x ./start.sh

# Switch to non-root user
USER appuser

# Expose Streamlit frontend and FastAPI backend ports
EXPOSE 7860 8000

# Health check against lightweight FastAPI probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

# Execute unified startup sequence
ENTRYPOINT ["./start.sh"]
