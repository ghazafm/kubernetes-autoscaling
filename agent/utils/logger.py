import logging
from datetime import datetime
from logging import Logger
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from rl import Q


def setup_logger(
    service_name: str,
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "logs",
) -> Logger:
    """
    Configure a logger with console and optional file output.

    Args:
        service_name (str): Name of the service (used for the log file name)
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file (bool): Whether to log to a file
        log_dir (str): Directory to store log files

    Returns:
        logging.Logger: Configured logger instance
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    logger = logging.getLogger(service_name)
    logger.setLevel(numeric_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    logging.getLogger("kubernetes.client.rest").setLevel(logging.WARNING)
    logging.getLogger("kubernetes").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        now = datetime.now().strftime("%Y-%m-%d-%H-%M")
        log_dir_time = log_dir + "/" + now
        if not Path(log_dir_time).exists():
            Path(log_dir_time).mkdir(parents=True, exist_ok=True)

        log_file = Path(log_dir_time) / f"{service_name}_{now}.log"

        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ---- Lightweight formatting helpers (no third-party deps) ----
def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _bar(pct: float, width: int = 12) -> str:
    """Draw a simple gauge bar with unicode blocks."""
    pct = _clamp(pct)
    filled = round(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _color(v: float, warn: float, crit: float, reverse: bool = False) -> str:
    """
    Colorize value by thresholds (green < warn < yellow < crit < red).
    reverse=True flips logic (good when 'lower is better' like response time).
    For RT, values > crit are always red (severe violations).
    """
    # ANSI colors
    GREEN, YELLOW, RED = "\033[32m", "\033[33m", "\033[31m"

    if reverse:
        ok = v <= warn
        mid = warn < v <= crit
    else:
        # For normal metrics (higher is worse)
        ok = v < warn
        mid = warn <= v < crit
        # Always red if exceeds critical threshold
        if v >= crit:
            return RED

    return GREEN if ok else (YELLOW if mid else RED)


def _fmt_pct(v: float) -> str:
    try:
        return f"{float(v):6.2f}%"
    except Exception:
        return f"{v}"


def _fmt_ms(v: float) -> str:
    # adapt for small and big RT values
    MS_TO_SECONDS_THRESHOLD = 1000.0
    try:
        v = float(v)
        if v < 1.0:  # sub-ms shown with 3 decimals
            return f"{v * 1000:6.2f}µs"
        if v < MS_TO_SECONDS_THRESHOLD:  # below 1s in ms
            return f"{v:6.2f}ms"
        # >= 1s
        return f"{v / MS_TO_SECONDS_THRESHOLD:6.2f}s"
    except Exception:
        return f"{v}"


def _safe_q_values(  # noqa: PLR0912
    agent: Q, state_key, logger: Logger
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[int]]:
    # Q-table path (for traditional Q-Learning)
    # Check if agent actually uses Q-table (not just inherits empty one from DQN)
    q_table = getattr(agent, "q_table", None)
    if (
        q_table is not None
        and len(q_table) > 0
        and getattr(agent, "agent_type", "") == "Q"
    ):
        # Convert numpy.ndarray to a hashable tuple if necessary
        if isinstance(state_key, np.ndarray):
            state_key = tuple(state_key.flatten())
        if state_key in q_table:
            q = q_table[state_key]
            max_q = float(np.max(q))
            best_idx = int(np.argmax(q))
            return q, max_q, best_idx
        # State not found in Q-table
        return None, None, None

    # DQN path (for Deep Q-Network)
    policy = getattr(agent, "policy_net", None)
    device = getattr(agent, "device", "cpu")  # Default to CPU
    if policy is not None:
        try:
            # CRITICAL: Set network to eval mode to handle batch size = 1
            # BatchNorm requires batch size > 1 in training mode
            was_training = policy.training
            policy.eval()

            with torch.no_grad():
                # Handle both numpy arrays and tuples
                if isinstance(state_key, tuple):
                    state_np = np.array(state_key, dtype=np.float32)
                elif isinstance(state_key, np.ndarray):
                    state_np = state_key.astype(np.float32)
                else:
                    state_np = np.array(state_key, dtype=np.float32)

                # Ensure state has correct shape
                if state_np.ndim == 1:
                    state_t = torch.from_numpy(state_np).unsqueeze(0)
                else:
                    state_t = torch.from_numpy(state_np)

                # Move to device if needed
                if device and device != "cpu":
                    state_t = state_t.to(device)

                q_t = policy(state_t)
                if q_t.ndim > 1:
                    q_t = q_t.squeeze(0)

                q_np = q_t.detach().cpu().numpy().astype(np.float32)
                max_q = float(q_np.max())
                best_idx = int(q_np.argmax())

                # Restore original training mode
                if was_training:
                    policy.train()

                return q_np, max_q, best_idx
        except Exception as exc:
            # Restore training mode on error
            if policy is not None and was_training:
                policy.train()

            # Log error details
            logger.debug(f"Failed to compute DQN Q-values: {exc}")
            shape_info = state_key.shape if isinstance(state_key, np.ndarray) else "N/A"
            logger.debug(f"State key type: {type(state_key)}, shape: {shape_info}")

    return None, None, None


def log_verbose_details(  # noqa: PLR0912, PLR0915
    observation: Dict[str, Any], agent: Any, verbose: bool, logger: Logger
) -> None:
    """
    Compact, high-signal CLI log with 13D state representation:
    ─ Line 1: CPU│Mem│RT│Replica%│Act│Qmax│Best + bars and colors
    ─ Line 2: Deltas (CPU/Mem/RT trends) + Time-in-State + Scaling Direction
    ─ Line 3: Load Indicators (RPS/pod, RPS delta, Error Rate) ← NEW!
    ─ Optional details on unknown state / missing Q if helpful
    """
    if not verbose:
        return

    # Pull metrics with sane defaults (cpu/mem are relative clipped percents 0..100)
    cpu = float(observation.get("cpu_usage", 0.0))  # % (relative)
    mem = float(observation.get("memory_usage", 0.0))  # % (relative)
    rt_percentage = float(observation.get("response_time", 0.0))  # percentage (0-100)
    replica_pct = float(observation.get("current_replica_pct", 0.0))  # % of range
    act = observation.get("last_action", 0)  # 0-99
    iter_no = observation.get("iteration")  # optional

    # Delta metrics (trends)
    cpu_delta = float(observation.get("cpu_delta", 0.0))
    mem_delta = float(observation.get("memory_delta", 0.0))
    rt_delta = float(observation.get("rt_delta", 0.0))

    # Stability and direction
    time_in_state = float(observation.get("time_in_state", 0.0))  # 0-1
    scaling_direction = float(observation.get("scaling_direction", 0.5))  # 0/0.5/1

    # NEW: Load indicators (scale-independent)
    rps_per_pod = float(observation.get("rps_per_pod", 0.0))  # RPS per pod
    rps_delta = float(observation.get("rps_delta", 0.0))  # Change in RPS/pod
    error_rate = float(observation.get("error_rate", 0.0))  # Error % (0-10%)

    # NEW: Relative violation indicators (0..1) and in-band booleans
    cpu_dist = float(observation.get("cpu_dist", 0.0))
    memory_dist = float(observation.get("memory_dist", 0.0))
    cpu_in_band = bool(observation.get("cpu_in_band", 0.0))
    memory_in_band = bool(observation.get("memory_in_band", 0.0))

    # Bars and colors
    cpu_col = _color(cpu, warn=70, crit=90)  # higher is worse
    mem_col = _color(mem, warn=75, crit=90)  # higher is worse
    # RT can exceed 100%, so use higher thresholds for color coding
    rt_col = _color(rt_percentage, warn=80, crit=100, reverse=False)

    cpu_bar = _bar(cpu)
    mem_bar = _bar(mem)
    # For RT bar, clamp display to 200% max so severe violations are visible
    rt_bar = _bar(min(rt_percentage, 200.0), width=12)
    replica_bar = _bar(replica_pct)

    # Q-values (works for both Q and DQN if available)
    state_key = agent.get_state_key(observation)
    q_vals, qmax, best_idx = _safe_q_values(agent, state_key, logger)

    # Epsilon (exploration rate) for DQN agents
    epsilon = getattr(agent, "epsilon", None)

    RESET = "\033[0m"
    CYAN = "\033[36m"
    BLUE = "\033[34m"

    # === LINE 1: Core Metrics ===
    hdr = f"▶ Iter {iter_no:02d} " if isinstance(iter_no, int) else "▶ "
    cpu_str = f"{cpu_col}CPU {_fmt_pct(cpu)} {cpu_bar}{RESET}"
    mem_str = f"{mem_col}MEM {_fmt_pct(mem)} {mem_bar}{RESET}"
    rt_str = f"{rt_col}RT {rt_percentage:6.1f}% {rt_bar}{RESET}"
    replica_str = f"{CYAN}REP {replica_pct:6.1f}% {replica_bar}{RESET}"
    act_str = f"ACT {int(act):3d}"

    # Prepare band status strings (human friendly)
    cpu_band_status = "IN" if cpu_in_band else f"OUT ({cpu_dist * 100:.0f}% off)"
    mem_band_status = "IN" if memory_in_band else f"OUT ({memory_dist * 100:.0f}% off)"

    if qmax is not None and best_idx is not None:
        q_str = f"Qmax {qmax:+.3f}"
        best_s = f"Best {best_idx:3d}"
    else:
        q_str, best_s = "Qmax  n/a", "Best n/a"

    # Add epsilon to line 1 for DQN agents
    if epsilon is not None:
        # Color epsilon: green (low=exploit), yellow (mid), red (high=explore)
        eps_col = _color(epsilon * 100, warn=30, crit=60, reverse=True)
        eps_str = f"{eps_col}ε {epsilon:.3f}{RESET}"
        logger.info(f"{hdr}| {act_str} | {q_str} | {best_s} | {eps_str}")
        logger.info(
            f"{' ' * len(hdr)}| {cpu_str} | {mem_str} | {rt_str} | {replica_str}"
        )
        # Small human-friendly band status on the next line
        logger.info(
            f"{' ' * len(hdr)}| CPU band: {cpu_band_status} | MEM band: {mem_band_status}"  # noqa: E501
        )
    else:
        logger.info(
            f"{' ' * len(hdr)}| {cpu_str} | {mem_str} | {rt_str} | "
            f"{replica_str} | {act_str} | {q_str} | {best_s}"
        )
        logger.info(
            f"{' ' * len(hdr)}| CPU band: {cpu_band_status} | MEM band: {mem_band_status}"  # noqa: E501
        )

    # === LINE 2: Deltas, Stability, Direction ===
    # Color deltas: green if decreasing (good for CPU/Mem/RT), red if increasing
    def _delta_color(delta: float) -> str:
        """Green for negative (decreasing), red for positive (increasing)"""
        if abs(delta) < 1.0:
            return "\033[90m"  # Gray for near-zero
        return "\033[32m" if delta < 0 else "\033[31m"  # Green/Red

    cpu_d_col = _delta_color(cpu_delta)
    mem_d_col = _delta_color(mem_delta)
    rt_d_col = _delta_color(rt_delta)

    cpu_d_str = f"{cpu_d_col}ΔCPU {cpu_delta:+6.1f}{RESET}"
    mem_d_str = f"{mem_d_col}ΔMEM {mem_delta:+6.1f}{RESET}"
    rt_d_str = f"{rt_d_col}ΔRT {rt_delta:+6.1f}{RESET}"

    # Time in state: show as percentage with simple indicator
    time_pct = time_in_state * 100.0
    time_str = f"{BLUE}Time {time_pct:5.1f}%{RESET}"

    # Scaling direction: visual indicator
    SCALE_UP_THRESHOLD = 0.6
    SCALE_DOWN_THRESHOLD = 0.4

    if scaling_direction > SCALE_UP_THRESHOLD:
        dir_symbol = "↑"
        dir_color = "\033[32m"  # Green
        dir_text = "UP"
    elif scaling_direction < SCALE_DOWN_THRESHOLD:
        dir_symbol = "↓"
        dir_color = "\033[33m"  # Yellow
        dir_text = "DN"
    else:
        dir_symbol = "="
        dir_color = "\033[90m"  # Gray
        dir_text = "=="

    dir_str = f"{dir_color}{dir_symbol}{dir_text}{RESET}"

    logger.info(
        f"{' ' * len(hdr)}| {cpu_d_str} | {mem_d_str} | {rt_d_str} | "
        f"{time_str} | Dir {dir_str}"
    )

    # === LINE 3: Load Indicators ===
    # RPS per pod: Show with bar (0-10 RPS typical, scale to 0-100% for bar)
    rps_normalized = min(rps_per_pod * 10.0, 100.0)  # Assume 10 RPS = 100%
    rps_bar = _bar(rps_normalized, width=10)

    # Color RPS based on load: green (low), yellow (medium), red (high)
    rps_col = _color(rps_normalized, warn=60, crit=85)
    rps_str = f"{rps_col}RPS/pod {rps_per_pod:5.2f} {rps_bar}{RESET}"

    # RPS Delta: green if decreasing, red if increasing
    rps_d_col = _delta_color(rps_delta)
    rps_d_str = f"{rps_d_col}ΔRPS {rps_delta:+5.2f}{RESET}"

    # Error Rate: Critical metric - red if > 1%, yellow if > 0.1%
    ERROR_WARN_THRESHOLD = 0.1
    ERROR_CRIT_THRESHOLD = 1.0
    if error_rate >= ERROR_CRIT_THRESHOLD:
        err_col = "\033[31m"  # Red
        err_symbol = "🔴"
    elif error_rate >= ERROR_WARN_THRESHOLD:
        err_col = "\033[33m"  # Yellow
        err_symbol = "⚠️ "
    else:
        err_col = "\033[32m"  # Green
        err_symbol = "✓ "

    # Show error rate with bar (0-10% range)
    err_bar = _bar(min(error_rate * 10.0, 100.0), width=10)
    err_str = f"{err_col}{err_symbol}ERR {error_rate:5.2f}% {err_bar}{RESET}"

    # Also print the band severity inline for quick diagnosis
    band_info = (
        f"CPU band={'IN' if cpu_in_band else 'OUT'}({cpu_dist:.2f}) "
        f"MEM band={'IN' if memory_in_band else 'OUT'}({memory_dist:.2f})"
    )
    logger.info(f"{' ' * len(hdr)}| {rps_str} | {rps_d_str} | {err_str} | {band_info}")

    # === DEBUG: Q-values ===
    if q_vals is None:
        logger.debug("  (state unseen or DQN/Torch unavailable; skipping Q table dump)")
    else:
        # Show only when debugging at very high verbosity
        pass
