FROM python:3.10-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Expose ports for app and metrics
EXPOSE 5000 8000

# Environment variables with defaults (can be overridden in Kubernetes)
# NOTE: MAX_MEMORY_MB is NOT set here - it MUST be provided via ConfigMap
#       to ensure it matches your container's memory limit (see configmap.yaml)
ENV APP_HOST=0.0.0.0 \
    APP_PORT=5000 \
    METRICS_PORT=8000 \
    DEBUG=false \
    DEFAULT_SLEEP_TIME=0.3 \
    GUNICORN_WORKERS=3 \
    GUNICORN_THREADS=4 \
    GUNICORN_TIMEOUT=120

# Run the application via Gunicorn in production mode.
# Workers and threads configurable via env vars for different training scenarios:
# - Conservative (low memory): --env GUNICORN_WORKERS=2 --env GUNICORN_THREADS=2
# - Balanced (default): 3 workers, 4 threads (can handle ~12 concurrent requests)
# - Aggressive (high memory): --env GUNICORN_WORKERS=4 --env GUNICORN_THREADS=6
CMD gunicorn \
    -w ${GUNICORN_WORKERS} \
    -k gthread \
    --threads ${GUNICORN_THREADS} \
    --timeout ${GUNICORN_TIMEOUT} \
    -b 0.0.0.0:5000 \
    --access-logfile - \
    --error-logfile - \
    main:app
