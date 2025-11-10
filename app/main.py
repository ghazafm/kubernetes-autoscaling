import hashlib
import logging
import math
import os
import time

from flask import Flask, jsonify, request
from rl_autoscale import enable_metrics

app = Flask(__name__)

# Configuration from environment variables
METRICS_PORT = int(os.environ.get("METRICS_PORT", "8000"))
APP_HOST = os.environ.get("APP_HOST", "127.0.0.1")
APP_PORT = int(os.environ.get("APP_PORT", "5000"))
DEBUG_MODE = os.environ.get("DEBUG", "false").lower() == "true"
DEFAULT_SLEEP_TIME = float(os.environ.get("DEFAULT_SLEEP_TIME", "0.3"))
MAX_MEMORY_MB = int(os.environ.get("MAX_MEMORY_MB", "100"))

# Enable RL autoscaling metrics
enable_metrics(app, port=METRICS_PORT)


@app.route("/api")
def hello():
    """Basic API endpoint."""
    time.sleep(DEFAULT_SLEEP_TIME)
    return "Hello!"


@app.route("/api/cpu")
def cpu_intensive():
    """
    CPU-intensive endpoint that performs complex mathematical calculations.
    This simulates high CPU usage for testing autoscaling.
    """
    iterations = request.args.get("iterations", default=1000000, type=int)

    result = 0
    for i in range(iterations):
        result += math.sqrt(i) * math.sin(i) * math.cos(i)
        if i % 10000 == 0:
            hashlib.sha256(str(i).encode()).hexdigest()

    return jsonify(
        {
            "status": "success",
            "message": "CPU-intensive task completed",
            "iterations": iterations,
            "result": result,
        }
    )


@app.route("/api/memory")
def memory_intensive():
    """
    Memory-intensive endpoint that allocates large amounts of memory.
    This simulates high memory usage for testing autoscaling.
    """
    size_mb = request.args.get("size_mb", default=50, type=int)

    # Cap maximum allocation based on environment config
    safe_mb = min(max(0, size_mb), MAX_MEMORY_MB)

    # Allocate a contiguous block of bytes instead of a list of Python ints.
    # Python ints are much larger than 8 bytes (object overhead), so
    # list(range(...)) will consume far more memory than requested and can
    # easily OOM the container under concurrent requests.
    try:
        allocated = bytearray(safe_mb * 1024 * 1024)
        # Touch the allocation sparsely to ensure memory is actually committed
        if safe_mb > 0:
            step = max(1, (safe_mb * 1024 * 1024) // 1024)
            for i in range(0, len(allocated), step):
                allocated[i] = 0

        # Use a tiny sample to return so we don't keep large objects around
        sample_sum = len(allocated) if allocated is not None else 0

        # Free memory promptly
        del allocated

        return jsonify(
            {
                "status": "success",
                "message": "Memory-intensive task completed",
                "allocated_mb": safe_mb,
                "sample_sum": sample_sum,
            }
        )
    except MemoryError:
        # If the allocation fails, return a clear error so load tests can record it
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Memory allocation failed (MemoryError)",
                    "requested_mb": size_mb,
                    "capped_mb": safe_mb,
                }
            ),
            503,
        )


@app.route("/health")
def health():
    """Health check endpoint for Kubernetes probes."""
    return jsonify({"status": "healthy", "service": "flask-app"}), 200


@app.route("/ready")
def ready():
    """Readiness probe endpoint for Kubernetes."""
    return jsonify({"status": "ready"}), 200


if __name__ == "__main__":
    logging.info(f"🚀 Starting Flask app on {APP_HOST}:{APP_PORT}")
    logging.info(f"📊 Metrics available on port {METRICS_PORT}")
    logging.info(f"🐛 Debug mode: {DEBUG_MODE}")
    app.run(host=APP_HOST, port=APP_PORT, debug=DEBUG_MODE)
