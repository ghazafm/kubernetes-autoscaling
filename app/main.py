import hashlib
import math
import time

from flask import Flask, jsonify, request
from prometheus_client import Counter, Histogram, start_http_server

app = Flask(__name__)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Request latency in seconds",
    ["method", "path"],
)
REQUEST_COUNT = Counter(
    "http_requests_total", "Total requests", ["method", "path", "http_status"]
)


@app.before_request
def before_request():
    request.start_time = time.time()


@app.after_request
def after_request(response):
    latency = time.time() - request.start_time
    REQUEST_LATENCY.labels(request.method, request.path).observe(latency)
    REQUEST_COUNT.labels(request.method, request.path, response.status_code).inc()
    return response


@app.route("/api")
def hello():
    time.sleep(0.3)
    return "Hello!"


@app.route("/api/cpu")
def cpu_intensive():
    """
    CPU-intensive endpoint that performs complex mathematical calculations.
    This simulates high CPU usage for testing autoscaling.
    """
    iterations = request.args.get("iterations", default=1000000, type=int)

    # Perform CPU-intensive operations
    result = 0
    for i in range(iterations):
        # Complex mathematical operations
        result += math.sqrt(i) * math.sin(i) * math.cos(i)
        if i % 10000 == 0:
            # Add some hash computation
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

    # Cap maximum allocation to prevent OOMKill
    size_mb = min(size_mb, 100)  # Maximum 100MB per request

    # Allocate memory - create a large list
    # Each element is approximately 8 bytes (integer), so we calculate elements needed
    elements = (size_mb * 1024 * 1024) // 8

    # Create large data structures
    large_list = list(range(elements))

    # Create some additional data structures to increase memory usage
    large_dict = {i: f"value_{i}" * 10 for i in range(min(elements // 100, 10000))}

    # Calculate some statistics to ensure the data is used
    total = sum(large_list[::1000])  # Sample every 1000th element to avoid timeout

    # Force garbage collection to release memory faster
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


if __name__ == "__main__":
    start_http_server(8000)  # expose metrics on :8000
    app.run(host="0.0.0.0", port=5000)  # noqa: S104
