import hashlib
import math
import os
import time

from flask import Flask, jsonify, request
from rl_autoscale import enable_metrics

app = Flask(__name__)

# Configuration from environment variables
METRICS_PORT = int(os.environ.get("METRICS_PORT", "8000"))
APP_HOST = os.environ.get("APP_HOST", "0.0.0.0")
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
    size_mb = min(size_mb, MAX_MEMORY_MB)

    elements = (size_mb * 1024 * 1024) // 8

    large_list = list(range(elements))

    large_dict = {i: f"value_{i}" * 10 for i in range(min(elements // 100, 10000))}

    total = sum(large_list[::1000])

    del large_list
    del large_dict

    return jsonify(
        {
            "status": "success",
            "message": "Memory-intensive task completed",
            "allocated_mb": size_mb,
            "sample_sum": total,
        }
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
    print(f"🚀 Starting Flask app on {APP_HOST}:{APP_PORT}")
    print(f"📊 Metrics available on port {METRICS_PORT}")
    print(f"🐛 Debug mode: {DEBUG_MODE}")
    app.run(host=APP_HOST, port=APP_PORT, debug=DEBUG_MODE)
