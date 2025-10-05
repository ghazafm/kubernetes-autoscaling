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

    logger = logging.getLogger()
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
    """
    # ANSI colors
    GREEN, YELLOW, RED = "\033[32m", "\033[33m", "\033[31m"
    ok = v <= warn if reverse else v < warn
    mid = (warn < v <= crit) if reverse else (warn <= v < crit)
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


def _safe_q_values(
    agent: Q, state_key: Tuple, logger: Logger
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[int]]:
    # Q-table path (for traditional Q-Learning)
    if getattr(agent, "q_table", None) is not None:
        # Convert numpy.ndarray to a hashable tuple if necessary
        if isinstance(state_key, np.ndarray):
            state_key = tuple(state_key.flatten())
        if state_key in agent.q_table:
            q = agent.q_table[state_key]
            max_q = float(np.max(q))
            best_idx = int(np.argmax(q))
            return q, max_q, best_idx
        # State not found in Q-table
        return None, None, None

    # DQN path (for Deep Q-Network)
    policy = getattr(agent, "policy_net", None)
    device = getattr(agent, "device", None)
    if policy is not None:
        try:
            with torch.no_grad():
                # Handle both numpy arrays and tuples
                if isinstance(state_key, tuple):
                    state_np = np.array(state_key, dtype=np.float32)
                else:
                    state_np = np.array(state_key, dtype=np.float32)

                state_t = torch.from_numpy(state_np).unsqueeze(0)
                if device and device != "cpu":
                    state_t = state_t.to(device)

                q_t = policy(state_t).squeeze(0)
                q_np = q_t.detach().cpu().numpy().astype(np.float32)
                max_q = float(q_np.max())
                best_idx = int(q_np.argmax())
                return q_np, max_q, best_idx
        except Exception as exc:
            logger.debug(f"Failed to compute DQN Q-values: {exc}")

    return None, None, None


def log_verbose_details(
    observation: Dict[str, Any], agent: Any, verbose: bool, logger: Logger
) -> None:
    """
    Compact, high-signal CLI log:
    ─ Summary line (one-liner): CPU│Mem│RT│Act│Qmax│Best + tiny bars and colors
    ─ Optional details on unknown state / missing Q if helpful
    """
    if not verbose:
        return

    # Pull metrics with sane defaults
    cpu = float(observation.get("cpu_usage", 0.0))  # %
    mem = float(observation.get("memory_usage", 0.0))  # %
    rt = float(observation.get("response_time", 0.0))  # ms (assumed)
    act = observation.get("last_action", 0)  # 0-100 (your convention)
    iter_no = observation.get("iteration")  # optional

    # Bars and colors
    cpu_col = _color(cpu, warn=70, crit=90)  # higher is worse
    mem_col = _color(mem, warn=75, crit=90)  # higher is worse
    rt_col = _color(
        rt, warn=200, crit=500, reverse=True
    )  # lower is better ⇒ reverse thresholds

    cpu_bar = _bar(cpu)
    mem_bar = _bar(mem)

    # Q-values (works for both Q and DQN if available)
    state_key = agent.get_state_key(observation)
    q_vals, qmax, best_idx = _safe_q_values(agent, state_key, logger)

    RESET = "\033[0m"
    hdr = f"▶ Iter {iter_no:02d} " if isinstance(iter_no, int) else "▶ "
    cpu_str = f"{cpu_col}CPU {_fmt_pct(cpu)} {cpu_bar}{RESET}"
    mem_str = f"{mem_col}MEM {_fmt_pct(mem)} {mem_bar}{RESET}"
    rt_str = f"{rt_col}RT {_fmt_ms(rt)}{RESET}"
    act_str = f"ACT {int(act):3d}"

    if qmax is not None and best_idx is not None:
        q_str = f"Qmax {qmax:+.3f}"
        best_s = f"Best {best_idx + 1:3d}"
    else:
        q_str, best_s = "Qmax  n/a", "Best  n/a"

    logger.info(
        f"{hdr}| {cpu_str} | {mem_str} | {rt_str} | {act_str} | {q_str} | {best_s}"
    )

    if q_vals is None:
        logger.debug("  (state unseen or DQN/Torch unavailable; skipping Q table dump)")
    else:
        # Show only when debugging at very high verbosity:

        # logger.debug(
        # f"  Q-values (first 8): {
        # np.array2string(q_vals[:8], precision=3, separator=', ')
        # }"
        # )
        pass
