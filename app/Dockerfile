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
ENV APP_HOST=0.0.0.0 \
    APP_PORT=5000 \
    METRICS_PORT=8000 \
    DEBUG=false \
    DEFAULT_SLEEP_TIME=0.3 \
    MAX_MEMORY_MB=100

# Run the application via Gunicorn in production mode. Use 2 workers by default; tune based on memory limits.
CMD ["gunicorn", "-w", "2", "-k", "gthread", "-b", "0.0.0.0:5000", "main:app"]
