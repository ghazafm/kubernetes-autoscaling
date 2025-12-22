# main.py - Improved Memory Management Version
import ctypes
import fcntl
import gc
import hashlib
import logging
import math
import os
import platform
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import psutil
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

MEMORY_PRESSURE_CACHE_LIMIT = int(os.environ.get("MEMORY_PRESSURE_CACHE_LIMIT", "5"))
CACHE_ENTRY_TTL = float(os.environ.get("CACHE_ENTRY_TTL", "300"))
CLEAN_CHECK_INTERVAL = float(os.environ.get("CLEAN_CHECK_INTERVAL", "10"))
CACHE_IDLE_SECONDS = float(os.environ.get("CACHE_IDLE_SECONDS", "60"))
CACHE_DEFAULT = os.environ.get("CACHE_DEFAULT", "true").lower() == "true"

# New memory management settings
USE_MALLOC_TRIM = os.environ.get("USE_MALLOC_TRIM", "true").lower() == "true"
MALLOC_TRIM_INTERVAL = float(os.environ.get("MALLOC_TRIM_INTERVAL", "30"))
AGGRESSIVE_GC = os.environ.get("AGGRESSIVE_GC", "true").lower() == "true"
ENABLE_METRICS = os.environ.get("ENABLE_METRICS", "true").lower() == "true"

# -----------------------
# Metrics Initialization
# -----------------------
# Import tempfile and fcntl for metrics lock


METRICS_LOCK_FILE = os.environ.get(
    "METRICS_LOCK_FILE",
    str(Path(tempfile.gettempdir()) / f"rl_metrics_{os.getuid()}.lock"),
)


def _acquire_file_lock(path: str) -> bool:
    """Acquire exclusive file lock for metrics server."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        app.config["_metrics_lock_fd"] = fd
        return True
    except Exception:
        return False


def _init_metrics_early() -> None:
    """Initialize metrics before first request (Flask 3.x compatibility)."""
    if not ENABLE_METRICS:
        logger.info("Metrics disabled via ENABLE_METRICS=false")
        return

    if _acquire_file_lock(METRICS_LOCK_FILE):
        try:
            enable_metrics(app, port=METRICS_PORT)
            logger.info("✅ Metrics enabled (single process) on port %s", METRICS_PORT)
        except Exception:
            logger.exception("Failed to enable metrics")
    else:
        logger.info("Metrics lock not acquired; skipping metrics init in this process.")


# Enable metrics at module import time
_init_metrics_early()


# -----------------------
# Memory Management Utilities
# -----------------------
class MemoryManager:
    """Enhanced memory management utilities for Python applications."""

    def __init__(self):
        self.libc = None
        self.malloc_trim_available = False
        self._init_malloc_trim()

    def _init_malloc_trim(self):
        """Initialize malloc_trim if available (Linux only)."""
        if not USE_MALLOC_TRIM:
            logger.info("malloc_trim disabled via USE_MALLOC_TRIM=false")
            return

        if platform.system() != "Linux":
            logger.info("malloc_trim not available (not Linux)")
            return

        try:
            self.libc = ctypes.CDLL("libc.so.6")
            # Test if malloc_trim is available
            self.libc.malloc_trim.argtypes = [ctypes.c_int]
            self.libc.malloc_trim.restype = ctypes.c_int
            self.malloc_trim_available = True
            logger.info("✅ malloc_trim initialized successfully")
        except (OSError, AttributeError) as e:
            logger.warning(f"malloc_trim not available: {e}")
            self.malloc_trim_available = False

    def trim_memory(self) -> int:
        """
        Force glibc to release free memory back to OS.

        Returns:
            1 if successful, 0 otherwise
        """
        if not self.malloc_trim_available:
            return 0

        try:
            # malloc_trim(0) releases all free memory
            result = self.libc.malloc_trim(0)
            if result:
                logger.debug("malloc_trim successfully released memory to OS")
            return result
        except Exception as e:
            logger.error(f"malloc_trim failed: {e}")
            return 0

    def aggressive_gc(self) -> int:
        """
        Perform aggressive garbage collection across all generations.

        Returns:
            Number of objects collected
        """
        # Collect from all generations (0, 1, 2)
        collected = 0
        try:
            # Run GC on each generation explicitly
            for generation in range(3):
                n = gc.collect(generation)
                collected += n

            logger.debug(f"Aggressive GC collected {collected} objects")
            return collected
        except Exception as e:
            logger.error(f"Aggressive GC failed: {e}")
            return 0

    def full_cleanup(self) -> Dict[str, int]:
        """
        Perform full memory cleanup: aggressive GC + malloc_trim.

        Returns:
            Dictionary with cleanup statistics
        """
        stats = {"gc_collected": 0, "malloc_trim_result": 0}

        # First, run aggressive garbage collection
        if AGGRESSIVE_GC:
            stats["gc_collected"] = self.aggressive_gc()
        else:
            stats["gc_collected"] = gc.collect()

        # Then, try to release memory to OS
        if self.malloc_trim_available:
            stats["malloc_trim_result"] = self.trim_memory()

        return stats


# Global memory manager instance
memory_manager = MemoryManager()


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


# -----------------------
# Background Cleanup Tasks
# -----------------------
def _background_cleaner():
    """Periodic cache cleaner with memory management (daemon thread, per-process)."""
    logger.info(
        "Background cleaner started: ttl=%s, idle=%s, interval=%s, limit=%s",
        CACHE_ENTRY_TTL,
        CACHE_IDLE_SECONDS,
        CLEAN_CHECK_INTERVAL,
        MEMORY_PRESSURE_CACHE_LIMIT,
    )

    last_malloc_trim = _now()

    while True:
        try:
            # Evict expired and idle entries
            removed_ttl = evict_expired_entries()
            removed_idle = evict_if_idle()
            removed = removed_ttl + removed_idle

            if removed:
                logger.info(
                    "Cleaner removed %d entries (cache_count=%d)",
                    removed,
                    len(_memory_pressure_cache),
                )

            # Periodic malloc_trim to return memory to OS
            current_time = _now()
            if (current_time - last_malloc_trim) > MALLOC_TRIM_INTERVAL:
                if memory_manager.malloc_trim_available:
                    result = memory_manager.trim_memory()
                    if result:
                        logger.info("Periodic malloc_trim released memory to OS")
                last_malloc_trim = current_time

            time.sleep(CLEAN_CHECK_INTERVAL)

        except Exception:
            logger.exception("Background cleaner encountered an error")
            time.sleep(CLEAN_CHECK_INTERVAL)


def _start_cleaner_once_per_process() -> None:
    """Start the background cleaner thread (once per process)."""
    global _cleaner_started
    if _cleaner_started:
        return
    with _cleaner_lock:
        if _cleaner_started:
            return
        threading.Thread(
            target=_background_cleaner, daemon=True, name="cache-cleaner"
        ).start()
        _cleaner_started = True
        logger.info("Background cleaner thread started (per-process).")


@app.before_request
def _mark_activity_and_start_cleaner():
    """Mark request activity and ensure cleaner is running."""
    if request.path not in EXCLUDED_IDLE_PATHS:
        app.config["_last_request_time"] = time.time()
    _start_cleaner_once_per_process()


# -----------------------
# Routes
# -----------------------
@app.route("/api")
def hello():
    time.sleep(DEFAULT_SLEEP_TIME)
    return "Hello!"


@app.route("/api/cpu")
def cpu_intensive():
    """CPU-intensive endpoint with capacity limits."""
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
    """Memory-intensive endpoint with improved cleanup."""
    size_mb = request.args.get("size_mb", default=50, type=int)
    hold_seconds = request.args.get("hold_seconds", default=0.0, type=float)

    cache_flag = request.args.get("cache", default=None, type=str)
    if cache_flag is None:
        should_cache = CACHE_DEFAULT
    else:
        should_cache = cache_flag not in ("0", "false", "no")

    safe_mb = min(max(0, size_mb), MAX_MEMORY_MB)
    safe_hold = min(max(0.0, hold_seconds), 30.0)

    allocated = None
    try:
        # Allocate memory
        allocated = bytearray(safe_mb * 1024 * 1024)

        # Touch memory to ensure pages are committed
        if safe_mb > 0:
            step = max(1, len(allocated) // 1024)
            for i in range(0, len(allocated), step):
                allocated[i] = 0

        # Hold if requested
        if safe_hold > 0:
            time.sleep(safe_hold)

        # Cache or free
        if should_cache:
            _maybe_cache_allocation(allocated, safe_mb)
            sample_sum = len(allocated)
        else:
            sample_sum = len(allocated)
            # Explicitly free memory when not caching
            del allocated
            allocated = None

            # Force garbage collection if AGGRESSIVE_GC is enabled
            if AGGRESSIVE_GC:
                memory_manager.aggressive_gc()

        return jsonify(
            {
                "status": "success",
                "message": "Memory-intensive task completed",
                "allocated_mb": safe_mb,
                "held_seconds": safe_hold,
                "cached": should_cache,
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
    finally:
        # Ensure cleanup even if exception occurs
        if allocated is not None and not _is_cached_object(allocated):
            del allocated
            allocated = None


@app.route("/clean", methods=["POST", "GET"])
def clean_cache_endpoint():
    """Manually trigger cache cleanup and memory release."""
    cleaned = clean_memory_cache()
    stats = memory_manager.full_cleanup()

    return jsonify(
        {
            "status": "cleaned",
            "cleared_items": cleaned,
            "gc_collected": stats["gc_collected"],
            "malloc_trim_result": stats["malloc_trim_result"],
        }
    ), 200


@app.route("/memory/stats", methods=["GET"])
def memory_stats():
    """Get current memory statistics."""
    try:
        process = psutil.Process()
        mem_info = process.memory_info()

        return jsonify(
            {
                "rss_mb": round(mem_info.rss / 1024 / 1024, 2),
                "vms_mb": round(mem_info.vms / 1024 / 1024, 2),
                "cache_count": len(_memory_pressure_cache),
                "cache_limit": MEMORY_PRESSURE_CACHE_LIMIT,
                "gc_stats": {"count": gc.get_count(), "threshold": gc.get_threshold()},
            }
        ), 200
    except ImportError:
        return jsonify(
            {
                "error": "psutil not installed",
                "cache_count": len(_memory_pressure_cache),
                "cache_limit": MEMORY_PRESSURE_CACHE_LIMIT,
            }
        ), 200


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
    """Check if object is in cache."""
    return any(entry.get("obj") is obj for entry in _memory_pressure_cache)


def _maybe_cache_allocation(allocated: bytearray, safe_mb: int) -> None:
    """Cache allocation if space available."""
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
    """
    Clear cache and run full memory cleanup.

    Returns:
        Number of cached items cleared
    """
    count = len(_memory_pressure_cache)
    if count:
        logger.info("Cleaning memory cache: removing %d cached allocations", count)

        # Clear references
        _memory_pressure_cache.clear()

        # Aggressive memory cleanup
        stats = memory_manager.full_cleanup()
        logger.info(
            "Memory cleanup: gc_collected=%d, malloc_trim=%d",
            stats["gc_collected"],
            stats["malloc_trim_result"],
        )

    return count


def evict_expired_entries() -> int:
    """
    Evict TTL-expired entries and enforce size limit.

    Returns:
        Number of entries removed
    """
    if CACHE_ENTRY_TTL <= 0:
        return 0

    now = _now()
    removed = 0
    new_cache: List[Dict[str, Any]] = []

    for entry in _memory_pressure_cache:
        ts = float(entry.get("ts", 0.0))
        if (now - ts) > CACHE_ENTRY_TTL:
            # Explicitly delete the cached object
            obj = entry.get("obj")
            if obj is not None:
                entry["obj"] = None
                del obj
            removed += 1
        else:
            new_cache.append(entry)

    # Enforce limit defensively
    while len(new_cache) > MEMORY_PRESSURE_CACHE_LIMIT:
        entry = new_cache.pop(0)
        obj = entry.get("obj")
        if obj is not None:
            entry["obj"] = None
            del obj
        removed += 1

    if removed:
        _memory_pressure_cache[:] = new_cache

        # Run cleanup after eviction
        stats = memory_manager.full_cleanup()
        logger.info(
            "Evicted %d entries, gc_collected=%d", removed, stats["gc_collected"]
        )

    return removed


def evict_if_idle() -> int:
    """
    If idle too long, clear cache.

    Returns:
        Number of entries removed
    """
    if CACHE_IDLE_SECONDS <= 0:
        return 0

    last = float(app.config.get("_last_request_time", _now()))
    if (_now() - last) > CACHE_IDLE_SECONDS and _memory_pressure_cache:
        logger.info("Idle for %.1fs -> clearing memory cache", _now() - last)
        return clean_memory_cache()

    return 0


# -----------------------
# Entrypoint (dev only)
# -----------------------
if __name__ == "__main__":
    logger.info("🚀 Starting Flask app on %s:%s", APP_HOST, APP_PORT)
    logger.info("📊 Metrics configured with METRICS_PORT=%s", METRICS_PORT)
    logger.info("🐛 Debug mode: %s", DEBUG_MODE)
    logger.info(
        "🧹 Memory management: malloc_trim=%s, aggressive_gc=%s",
        USE_MALLOC_TRIM,
        AGGRESSIVE_GC,
    )

    _start_cleaner_once_per_process()
    app.run(host=APP_HOST, port=APP_PORT, debug=DEBUG_MODE)
