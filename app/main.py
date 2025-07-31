import time

from flask import Flask, request
from prometheus_client import Counter, Histogram, start_http_server

app = Flask(__name__)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
)
REQUEST_COUNT = Counter(
    "http_requests_total", "Total requests", ["method", "endpoint", "http_status"]
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


if __name__ == "__main__":
    start_http_server(8000)  # expose metrics on :8000
    app.run(host="0.0.0.0", port=5000)
