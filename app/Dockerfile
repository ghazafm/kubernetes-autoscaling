FROM python:3.10-slim

# Alternative Dockerfile using jemalloc for superior memory management
# jemalloc is more aggressive about releasing memory to OS compared to glibc malloc
# Use this if you need the absolute best memory management

WORKDIR /app

# Install jemalloc and other dependencies
# jemalloc provides better memory allocation/deallocation patterns
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libjemalloc2 \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Expose ports for app and metrics
EXPOSE 5000 8000

# Environment variables
ENV APP_HOST=0.0.0.0 \
    APP_PORT=5000 \
    METRICS_PORT=8000 \
    DEBUG=false \
    DEFAULT_SLEEP_TIME=0.3 \
    GUNICORN_WORKERS=1 \
    GUNICORN_THREADS=4 \
    GUNICORN_TIMEOUT=120

# ========================================
# jemalloc Configuration
# ========================================

# Preload jemalloc to replace glibc malloc
# This makes Python use jemalloc for all memory allocations
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

# jemalloc tuning for better memory release
# dirty_decay_ms: Time (ms) before returning dirty pages to OS (default: 10000)
# muzzy_decay_ms: Time (ms) before returning muzzy pages to OS (default: 10000)
# Lower values = more aggressive memory release, but more CPU overhead
ENV MALLOC_CONF="dirty_decay_ms:5000,muzzy_decay_ms:5000,narenas:2"

# Python memory optimization
ENV PYTHONMALLOC=malloc

# Memory management features
ENV USE_MALLOC_TRIM=true \
    MALLOC_TRIM_INTERVAL=30 \
    AGGRESSIVE_GC=true

# ========================================
# Gunicorn Configuration (optimized for jemalloc)
# ========================================

# With jemalloc, we can be more aggressive with worker recycling
# as memory release is more predictable

CMD gunicorn \
    -w ${GUNICORN_WORKERS} \
    -k gthread \
    --threads ${GUNICORN_THREADS} \
    --timeout ${GUNICORN_TIMEOUT} \
    --max-requests 500 \
    --max-requests-jitter 50 \
    --worker-tmp-dir /dev/shm \
    -b 0.0.0.0:5000 \
    --access-logfile - \
    --error-logfile - \
    main:app