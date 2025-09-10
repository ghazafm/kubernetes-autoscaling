import atexit
import logging
import signal

import numpy as np


def parse_cpu_value(cpu_str):
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


def parse_memory_value(memory_str):
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
    agent, current_episode, current_iteration, checkpoint_dir, save_on_interrupt, logger
):
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
