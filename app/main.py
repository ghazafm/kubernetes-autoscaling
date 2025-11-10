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
MAX_CPU_ITERATIONS = int(os.environ.get("MAX_CPU_ITERATIONS", "2000000"))

# Enable RL autoscaling metrics
enable_metrics(app, port=METRICS_PORT)
MEMORY_PRESSURE_CACHE_LIMIT = 5
MEMORY_PRESSURE_CACHE_CLEAN_THRESHOLD = 3

# In-memory cache to optionally hold allocations longer for RL training
# (simulates memory not being freed immediately under load)
_memory_pressure_cache = []


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

    Query parameters:
    - iterations: Number of iterations to run (capped by MAX_CPU_ITERATIONS)

    Returns:
    - 200: Request completed successfully
    - 503: Requested iterations exceed system capacity (triggers RL agent error
    detection)
    """
    iterations = request.args.get("iterations", default=1000000, type=int)

    # Check if request exceeds capacity BEFORE processing
    # Return 503 so RL agent observes this as a capacity error
    if iterations > MAX_CPU_ITERATIONS:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Requested CPU iterations exceed system capacity",
                    "requested_iterations": iterations,
                    "max_allowed_iterations": MAX_CPU_ITERATIONS,
                    "reason": "cpu_capacity_limit",
                }
            ),
            503,
        )

    # Cap to safe range (handles negative values)
    safe_iterations = min(max(0, iterations), MAX_CPU_ITERATIONS)

    result = 0
    for i in range(safe_iterations):
        result += math.sqrt(i) * math.sin(i) * math.cos(i)
        if i % 10000 == 0:
            hashlib.sha256(str(i).encode()).hexdigest()

    return jsonify(
        {
            "status": "success",
            "message": "CPU-intensive task completed",
            "iterations": safe_iterations,
            "result": result,
        }
    )


@app.route("/api/memory")
def memory_intensive():
    """
    Memory-intensive endpoint that allocates large amounts of memory.
    This simulates high memory usage for testing autoscaling.

    Query parameters:
    - size_mb: Memory to allocate in MB (capped by MAX_MEMORY_MB)
    - hold_seconds: Optional. Hold allocation for N seconds (for RL training)
                    This simulates sustained memory pressure across concurrent requests
    """
    size_mb = request.args.get("size_mb", default=50, type=int)
    hold_seconds = request.args.get("hold_seconds", default=0, type=float)

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

        # Optional: hold memory for sustained pressure (useful for RL training)
        # This allows concurrent requests to accumulate and push container memory higher
        if hold_seconds > 0:
            # Limit hold time to prevent indefinite blocking
            safe_hold = min(hold_seconds, 30.0)
            time.sleep(safe_hold)

            # Optionally cache allocation temporarily to prevent immediate GC
            # (simulates real-world scenarios where memory isn't freed instantly)
            if (
                len(_memory_pressure_cache) < MEMORY_PRESSURE_CACHE_LIMIT
            ):  # Limit cache size
                _memory_pressure_cache.append(allocated)
                # Cleanup old entries
                if len(_memory_pressure_cache) > MEMORY_PRESSURE_CACHE_CLEAN_THRESHOLD:
                    _memory_pressure_cache.pop(0)

        # Use a tiny sample to return so we don't keep large objects around
        sample_sum = len(allocated) if allocated is not None else 0

        # Free memory promptly (unless cached above)
        if allocated not in _memory_pressure_cache:
            del allocated

        return jsonify(
            {
                "status": "success",
                "message": "Memory-intensive task completed",
                "allocated_mb": safe_mb,
                "held_seconds": hold_seconds if hold_seconds > 0 else 0,
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
