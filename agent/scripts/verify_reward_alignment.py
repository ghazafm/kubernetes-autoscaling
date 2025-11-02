"""Verify that generator reward matches KubernetesEnv._calculate_reward

Usage:
    python scripts/verify_reward_alignment.py

This script loads the generator's _env_reward implementation and calls the
KubernetesEnv._calculate_reward method (without running full init) for the same
random states to ensure both functions compute the same reward given the same
hyperparameters.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import numpy as np

# Load project root so imports work
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Dynamically import the scripts/generate_training_csv.py module to access _env_reward
spec = importlib.util.spec_from_file_location(
    "generate_training_csv",
    str(ROOT / "scripts" / "generate_training_csv.py"),
)
module = importlib.util.module_from_spec(spec)
loader = spec.loader
assert loader is not None
loader.exec_module(module)
_env_reward = module._env_reward

# Import KubernetesEnv class
from environment import KubernetesEnv


class Dummy:
    pass


def compare_once(params: dict) -> tuple[float, float]:
    """Generate one random state, compute reward by both implementations."""
    cpu = random.uniform(0.0, 100.0)
    mem = random.uniform(0.0, 100.0)
    resp = random.uniform(0.0, params["max_response_time"] * 2.0)
    replica = random.randint(params["min_replicas"], params["max_replicas"])

    # env generator reward expects cpu/mem/resp in percentages as done by generator
    reward_gen = _env_reward(
        cpu,
        mem,
        resp,
        replica,
        min_replicas=params["min_replicas"],
        max_replicas=params["max_replicas"],
        min_cpu=params["min_cpu"],
        min_memory=params["min_memory"],
        max_cpu=params["max_cpu"],
        max_memory=params["max_memory"],
        max_response_time=params["max_response_time"],
        response_time_weight=params["response_time_weight"],
        cpu_memory_weight=params["cpu_memory_weight"],
        cost_weight=params["cost_weight"],
    )

    # Build dummy object with attributes expected by KubernetesEnv._calculate_reward
    d = Dummy()
    d.response_time = resp
    d.max_response_time = params["max_response_time"]
    d.cpu_usage = cpu
    d.memory_usage = mem
    d.min_cpu = params["min_cpu"]
    d.max_cpu = params["max_cpu"]
    d.min_memory = params["min_memory"]
    d.max_memory = params["max_memory"]
    d.response_time_weight = params["response_time_weight"]
    d.cpu_memory_weight = params["cpu_memory_weight"]
    d.cost_weight = params["cost_weight"]
    d.replica_state = replica
    d.min_replicas = params["min_replicas"]
    d.range_replicas = max(1, params["max_replicas"] - params["min_replicas"])

    reward_env = KubernetesEnv._calculate_reward(d)

    return reward_gen, reward_env


def main() -> None:
    params = {
        "min_replicas": 1,
        "max_replicas": 50,
        "min_cpu": 20.0,
        "max_cpu": 90.0,
        "min_memory": 20.0,
        "max_memory": 90.0,
        "max_response_time": 100.0,
        "response_time_weight": 1.0,
        "cpu_memory_weight": 0.5,
        "cost_weight": 0.3,
    }

    random.seed(42)
    mismatches = 0
    for i in range(200):
        r_gen, r_env = compare_once(params)
        if not np.isclose(r_gen, r_env, atol=1e-6):
            print(f"Mismatch #{i}: gen={r_gen:.6f}, env={r_env:.6f}")
            mismatches += 1

    if mismatches == 0:
        print("All checks passed: generator reward matches environment reward.")
    else:
        print(f"Found {mismatches} mismatches out of 200 samples.")


if __name__ == "__main__":
    main()
