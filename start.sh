#!/bin/bash
set -e

# Port configuration with fallbacks
BACKEND_PORT=8000
FRONTEND_PORT=${PORT:-7860}

echo "=== Launching AI-Powered Interview Coach ==="
echo "Starting FastAPI backend on port ${BACKEND_PORT}..."
uvicorn app.main:app --host 0.0.0.0 --port ${BACKEND_PORT} &

# Wait for backend health check
echo "Waiting for backend to become healthy..."
until curl -s http://127.0.0.1:${BACKEND_PORT}/health | grep -q "healthy"; do
    sleep 0.5
done
echo "Backend is healthy!"

echo "Starting Streamlit frontend on port ${FRONTEND_PORT}..."
exec streamlit run frontend/app.py --server.port ${FRONTEND_PORT} --server.address 0.0.0.0
