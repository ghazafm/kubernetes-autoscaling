import ast
import json
from logging import Logger
from typing import Iterable, List, Tuple, Union

import numpy as np
import torch
from rl import Q


def parse_cpu_value(cpu_str: str, logger: Logger) -> float:
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
        logger.warning(f"Could not parse CPU value '{cpu_str}': {e}")
        return 0.0


def parse_memory_value(memory_str: str, logger: Logger) -> float:
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
        logger.warning(f"Could not parse memory value '{memory_str}': {e}")
        return 0.0


def log_verbose_details(
    observation: dict, agent: Q, verbose: bool, logger: Logger
) -> None:
    """Log detailed observation and Q-value information if verbose mode is enabled"""
    if not verbose:
        return

    logger.debug(observation)
    logger.info("  🔍 Observation:")
    cpu_usage = observation.get("cpu_usage", 0.0)
    logger.info(f"     CPU: {cpu_usage:.3f}%")  # Already in %

    memory_usage = observation.get("memory_usage", 0.0)
    logger.info(f"     Memory: {memory_usage:.3f}%")  # Already in %

    response_time = observation.get("response_time", 0.0)
    logger.info(f"     Response Time: {response_time:.3f}%")  # Already in %

    last_action = observation.get("last_action", 0.0)
    logger.info(f"     Last Action: {last_action}")  # Already raw 0-99

    state_key = agent.get_state_key(observation)
    logger.info(f"  🗝️  State Key: {state_key}")

    if agent.__class__.__name__ == "DQN":
        logger.info("  🧠 Mode: DQN (neural network)")
        try:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state_key).unsqueeze(0).to(agent.device)
                q_values = agent.policy_net(state_tensor).squeeze(0)
                max_q = torch.max(q_values).item()
                best_action = (
                    torch.argmax(q_values).item() + 1
                )  # Convert to 1-100 range
                logger.info(
                    f"  🧠 Q-Values: Max={max_q:.3f}, Best Action={best_action}"
                )
        except Exception as e:
            logger.info(f"  🧠 Could not compute Q-values: {e}")
    # For Q-table mode, state_key is a tuple
    elif state_key in agent.q_table:
        q_values = agent.q_table[state_key]
        max_q = np.max(q_values)
        best_action = np.argmax(q_values) + 1  # Convert to 1-100 range
        logger.info(f"  🧠 Q-Values: Max={max_q:.3f}, Best Action={best_action}")
    else:
        logger.info("  🧠 State not in Q-table yet")

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
