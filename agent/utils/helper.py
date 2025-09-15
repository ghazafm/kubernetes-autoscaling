import ast
import atexit
import json
import logging
import signal
from typing import Any, Iterable, List, Tuple, Union

import numpy as np


def parse_cpu_value(cpu_str: str) -> float:
    """Parse CPU value from kubernetes format to cores (float)"""
    try:
        if cpu_str.endswith("m"):
            return float(cpu_str[:-1]) / 1000
        if cpu_str.endswith("n"):
            return float(cpu_str[:-1]) / 1000000000
        if cpu_str.endswith("u"):
            return float(cpu_str[:-1]) / 1000000
        return float(cpu_str)
    except (ValueError, IndexError) as e:
        logging.warning(f"Could not parse CPU value '{cpu_str}': {e}")
        return 0.0


def parse_memory_value(memory_str: str) -> float:
    """Parse memory value from kubernetes format to MB (float)"""
    try:
        if memory_str.endswith("Ki"):
            return float(memory_str[:-2]) / 1024
        if memory_str.endswith("Mi"):
            return float(memory_str[:-2])
        if memory_str.endswith("Gi"):
            return float(memory_str[:-2]) * 1024
        if memory_str.endswith("Ti"):
            return float(memory_str[:-2]) * 1024 * 1024
        return float(memory_str) / (1024 * 1024)
    except (ValueError, IndexError) as e:
        logging.warning(f"Could not parse memory value '{memory_str}': {e}")
        return 0.0


def setup_interruption_handlers(
    agent: Any,
    current_episode: list[int],
    current_iteration: list[int],
    checkpoint_dir: str,
    save_on_interrupt: bool,
    logger: Any,
) -> dict[str, Any]:
    """Setup signal handlers and atexit for graceful shutdown"""

    def _final_save(checkpoint_dir: str):
        try:
            agent.save_checkpoint(
                checkpoint_dir,
                episode=current_episode[0],
                iteration=current_iteration[0],
                prefix="final",
            )
            logger.info("✅ Final checkpoint saved on exit.")
        except Exception as e:
            logger.exception(f"Failed to save final checkpoint: {e}")

    atexit.register(_final_save, checkpoint_dir=checkpoint_dir)

    stop_requested = {"flag": False}

    def _handle_sigint(signum, frame):
        stop_requested["flag"] = True
        logger.warning(
            "⚠️  Ctrl+C detected. Will checkpoint and stop at next safe point..."
        )

    if save_on_interrupt:
        signal.signal(signal.SIGINT, _handle_sigint)

    return stop_requested


def log_verbose_details(observation, agent, verbose, logger):
    """Log detailed observation and Q-value information if verbose mode is enabled"""
    if not verbose:
        return

    logger.info("  🔍 Observation:")
    logger.info(f"     CPU: {observation.get('cpu_usage', 0):.1f}%")
    logger.info(f"     Memory: {observation.get('memory_usage', 0):.1f}%")
    logger.info(f"     Response Time: {observation.get('response_time', 0):.1f}ms")
    logger.info(f"     Last Action: {observation.get('last_action', 'N/A')}")

    state_key = agent.get_state_key(observation)
    logger.info(f"  🗝️  State Key: {state_key}")

    if state_key in agent.q_table:
        q_values = agent.q_table[state_key]
        max_q = np.max(q_values)
        best_action = np.argmax(q_values)
        logger.info(f"  🧠 Q-Values: Max={max_q:.3f}, Best Action={best_action}")

    logger.info("----------------------------------------")


def normalize_endpoints(
    endpoints: Union[str, Iterable, None],
    default: Iterable[Tuple[str, str]] = (("/", "GET"), ("/docs", "GET")),
) -> List[Tuple[str, str]]:
    """
    Terima berbagai bentuk:
      - JSON string: [["/a","GET"],["/b","POST"]]
      - Python literal string: [("/a","GET"), ("/b","POST")]
      - List[tuple]: [("/a","GET"), ...]
      - List[str]: ["/a", "/b"]   -> method default "GET"
      - String tunggal: "/a"      -> method default "GET"
    Return: List[Tuple[str, str]]
    """
    if endpoints is None:
        endpoints = default

    # sudah iterable tuple/list?
    if isinstance(endpoints, (list, tuple)):
        out: List[Tuple[str, str]] = []
        for item in endpoints:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                ep = str(item[0])
                method = str(item[1]) if len(item) > 1 else "GET"
                out.append((ep, method))
            elif isinstance(item, str):
                out.append((item, "GET"))
        return out

    # string -> coba JSON dulu
    if isinstance(endpoints, str):
        s = endpoints.strip()
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(s)
                return normalize_endpoints(parsed, default)
            except Exception:
                print("Failed to parse endpoints JSON:", s)  # noqa: T201
        # fallback: anggap string tunggal path
        return [(s, "GET")]

    # fallback default
    return normalize_endpoints(default, default)
