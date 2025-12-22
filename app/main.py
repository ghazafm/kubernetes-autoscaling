# main.py
import contextlib
import gc
import hashlib
import logging
import math
import os
import threading
import time
from typing import Any, Dict, List

from flask import Flask, jsonify, request
from rl_autoscale import enable_metrics

app = Flask(__name__)

# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -----------------------
# Config
# -----------------------
METRICS_PORT = int(os.environ.get("METRICS_PORT", "8000"))
APP_HOST = os.environ.get("APP_HOST", "127.0.0.1")
APP_PORT = int(os.environ.get("APP_PORT", "5000"))
DEBUG_MODE = os.environ.get("DEBUG", "false").lower() == "true"

DEFAULT_SLEEP_TIME = float(os.environ.get("DEFAULT_SLEEP_TIME", "0.1"))
MAX_MEMORY_MB = int(os.environ.get("MAX_MEMORY_MB", "100"))
MAX_CPU_ITERATIONS = int(os.environ.get("MAX_CPU_ITERATIONS", "500000"))

# Cache behavior for sustained memory pressure
MEMORY_PRESSURE_CACHE_LIMIT = int(os.environ.get("MEMORY_PRESSURE_CACHE_LIMIT", "5"))
CACHE_ENTRY_TTL = float(
    os.environ.get("CACHE_ENTRY_TTL", "300")
)  # seconds; 0 disables TTL eviction
CLEAN_CHECK_INTERVAL = float(os.environ.get("CLEAN_CHECK_INTERVAL", "10"))

# Optional idle eviction: if no requests for this long, clear cache (0 disables)
CACHE_IDLE_SECONDS = float(os.environ.get("CACHE_IDLE_SECONDS", "60"))

# -----------------------
# Metrics
# -----------------------
# IMPORTANT NOTE:
# If rl_autoscale.enable_metrics() starts a separate HTTP server bound to METRICS_PORT,
# then running multiple Gunicorn workers will cause "address already in use".
# If that happens for you, set GUNICORN_WORKERS=1 OR
# change rl_autoscale to expose /metrics
# on the same Flask/Gunicorn port instead.
enable_metrics(app, port=METRICS_PORT)

# -----------------------
# Shared state (per-process)
# -----------------------
app.config["_last_request_time"] = time.time()
_memory_pressure_cache: List[Dict[str, Any]] = []

_cleaner_started = False
_cleaner_lock = threading.Lock()


def _now() -> float:
    return time.time()


EXCLUDED_IDLE_PATHS = {"/health", "/ready", "/metrics"}


@app.before_request
def mark_request_and_start_cleaner():
    global _cleaner_started  # noqa: PLW0603

    # Only treat "real traffic" as activity
    if request.path not in EXCLUDED_IDLE_PATHS:
        app.config["_last_request_time"] = time.time()

    if _cleaner_started:
        return

    with _cleaner_lock:
        if _cleaner_started:
            return
        threading.Thread(target=_background_cleaner, daemon=True).start()
        _cleaner_started = True
        logger.info("Background cleaner thread started (per-process).")


# -----------------------
# Routes
# -----------------------
@app.route("/api")
def hello():
    time.sleep(DEFAULT_SLEEP_TIME)
    return "Hello!"


@app.route("/api/cpu")
def cpu_intensive():
    """
    CPU-intensive endpoint.

    Query parameters:
      - iterations: int (default 100000)
    Returns:
      - 200 on success
      - 503 if requested > MAX_CPU_ITERATIONS (capacity signal)
    """
    iterations = request.args.get("iterations", default=100000, type=int)

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

    safe_iterations = min(max(0, iterations), MAX_CPU_ITERATIONS)

    result = 0.0
    for i in range(safe_iterations):
        # moderately expensive math
        result += math.sqrt(i) * math.sin(i) * math.cos(i)
        # add some hashing periodically
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
    Memory-intensive endpoint.

    Query parameters:
      - size_mb: int (default 50, capped by MAX_MEMORY_MB)
      - hold_seconds: float (default 0, capped to 30)

    Behavior:
      - Allocates bytearray(size_mb).
      - If hold_seconds > 0, holds allocation during request.
      - Optionally caches allocation after hold to simulate lingering memory.
    """
    size_mb = request.args.get("size_mb", default=50, type=int)
    hold_seconds = request.args.get("hold_seconds", default=0.0, type=float)

    safe_mb = min(max(0, size_mb), MAX_MEMORY_MB)
    safe_hold = min(max(0.0, hold_seconds), 30.0)

    try:
        allocated = bytearray(safe_mb * 1024 * 1024)

        # Touch sparsely so pages are committed
        if safe_mb > 0:
            step = max(1, len(allocated) // 1024)
            for i in range(0, len(allocated), step):
                allocated[i] = 0

        if safe_hold > 0:
            time.sleep(safe_hold)
            _maybe_cache_allocation(allocated, safe_mb)

        sample_sum = len(allocated)

        # If not cached, free promptly
        if not _is_cached_object(allocated):
            del allocated

        return jsonify(
            {
                "status": "success",
                "message": "Memory-intensive task completed",
                "allocated_mb": safe_mb,
                "held_seconds": safe_hold,
                "sample_sum": sample_sum,
                "cache_count": len(_memory_pressure_cache),
            }
        )

    except MemoryError:
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


@app.route("/clean", methods=["POST", "GET"])
def clean_cache_endpoint():
    cleaned = clean_memory_cache()
    return jsonify({"status": "cleaned", "cleared_items": cleaned}), 200


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "service": "flask-app"}), 200


@app.route("/ready")
def ready():
    return jsonify({"status": "ready"}), 200


# -----------------------
# Cache helpers
# -----------------------
def _is_cached_object(obj: Any) -> bool:
    return any(entry.get("obj") is obj for entry in _memory_pressure_cache)


def _maybe_cache_allocation(allocated: bytearray, safe_mb: int) -> None:
    """Cache allocations to simulate lingering memory after request completes."""
    if MEMORY_PRESSURE_CACHE_LIMIT <= 0:
        return

    if len(_memory_pressure_cache) < MEMORY_PRESSURE_CACHE_LIMIT:
        _memory_pressure_cache.append({"ts": _now(), "obj": allocated})
        logger.info(
            "Cached allocation: %s MB (cache_count=%d/%d)",
            safe_mb,
            len(_memory_pressure_cache),
            MEMORY_PRESSURE_CACHE_LIMIT,
        )
    else:
        logger.info(
            "Cache full: skipped caching allocation of %s MB (cache_count=%d/%d)",
            safe_mb,
            len(_memory_pressure_cache),
            MEMORY_PRESSURE_CACHE_LIMIT,
        )


def clean_memory_cache() -> int:
    """Clear cache and run GC. Returns number of cleared items."""
    count = len(_memory_pressure_cache)
    if count:
        logger.info("Cleaning memory cache: removing %d cached allocations", count)
        _memory_pressure_cache.clear()
        with contextlib.suppress(Exception):
            gc.collect()
    return count


def evict_expired_entries() -> int:
    """Evict TTL-expired entries and enforce size limit. Returns number removed."""
    if CACHE_ENTRY_TTL <= 0:
        return 0

    now = _now()
    removed = 0
    new_cache: List[Dict[str, Any]] = []

    for entry in _memory_pressure_cache:
        ts = float(entry.get("ts", 0.0))
        if (now - ts) > CACHE_ENTRY_TTL:
            removed += 1
        else:
            new_cache.append(entry)

    # Enforce limit defensively
    while len(new_cache) > MEMORY_PRESSURE_CACHE_LIMIT:
        new_cache.pop(0)
        removed += 1

    if removed:
        _memory_pressure_cache[:] = new_cache
        with contextlib.suppress(Exception):
            gc.collect()
    if not removed and _memory_pressure_cache:
        logger.info(
            "No TTL eviction yet (oldest_age=%.1fs, ttl=%.1fs)",
            time.time() - _memory_pressure_cache[0]["ts"],
            CACHE_ENTRY_TTL,
        )

    return removed


def evict_if_idle() -> int:
    """If idle too long, clear cache. Returns number removed."""
    if CACHE_IDLE_SECONDS <= 0:
        return 0
    last = float(app.config.get("_last_request_time", _now()))
    if (_now() - last) > CACHE_IDLE_SECONDS and _memory_pressure_cache:
        logger.info("Idle for %.1fs -> clearing memory cache", _now() - last)
        return clean_memory_cache()
    return 0


def _background_cleaner():
    """Periodic cache cleaner (daemon thread, per-process)."""
    logger.info(
        "Background cleaner started: ttl=%s, idle=%s, interval=%s, limit=%s",
        CACHE_ENTRY_TTL,
        CACHE_IDLE_SECONDS,
        CLEAN_CHECK_INTERVAL,
        MEMORY_PRESSURE_CACHE_LIMIT,
    )
    while True:
        try:
            removed_ttl = evict_expired_entries()
            removed_idle = evict_if_idle()
            removed = removed_ttl + removed_idle
            if removed:
                logger.info(
                    "Cleaner removed %d entries (cache_count=%d)",
                    removed,
                    len(_memory_pressure_cache),
                )
            time.sleep(CLEAN_CHECK_INTERVAL)
        except Exception:
            logger.exception("Background cleaner encountered an error")
            time.sleep(CLEAN_CHECK_INTERVAL)


# -----------------------
# Entrypoint (dev only)
# -----------------------
if __name__ == "__main__":
    logger.info("🚀 Starting Flask app on %s:%s", APP_HOST, APP_PORT)
    logger.info("📊 Metrics configured with METRICS_PORT=%s", METRICS_PORT)
    logger.info("🐛 Debug mode: %s", DEBUG_MODE)

    # Start cleaner in dev too (Gunicorn workers start via before_request)
    with _cleaner_lock:
        if not _cleaner_started:
            threading.Thread(
                target=_background_cleaner, daemon=True, name="cache-cleaner"
            ).start()

    app.run(host=APP_HOST, port=APP_PORT, debug=DEBUG_MODE)
