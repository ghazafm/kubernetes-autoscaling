"""Generate a synthetic CSV compatible with agent/offline_train.py

Usage:
    python scripts/generate_training_csv.py --out data/generated_training.csv --rows 1000

The generated CSV columns:
  cpu_usage,memory_usage,response_time,replica,action,reward,next_state,done

next_state is a Python-dict-like string (single quotes) so the training script can ast.literal_eval it.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path


def _env_reward(
    cpu: float,
    mem: float,
    resp: float,
    replica_state: int,
    *,
    min_replicas: int = 1,
    max_replicas: int = 50,
    min_cpu: float = 20.0,
    min_memory: float = 20.0,
    max_cpu: float = 90.0,
    max_memory: float = 90.0,
    max_response_time: float = 100.0,
    response_time_weight: float = 1.0,
    cpu_memory_weight: float = 0.5,
    cost_weight: float = 0.3,
) -> float:
    # follow KubernetesEnv._calculate_reward()
    range_replicas = max(1, max_replicas - min_replicas)
    response_time_percentage = (resp / max_response_time) * 100.0

    if cpu < min_cpu:
        cpu_pen = (min_cpu - cpu) / min_cpu
    elif cpu > max_cpu:
        cpu_pen = (cpu - max_cpu) / (100 - max_cpu)
    else:
        cpu_pen = 0.0

    if mem < min_memory:
        mem_pen = (min_memory - mem) / min_memory
    elif mem > max_memory:
        mem_pen = (mem - max_memory) / (100 - max_memory)
    else:
        mem_pen = 0.0

    resp_pen = min(
        response_time_weight, max(0.0, (response_time_percentage - 100.0) / 100.0)
    )

    cpu_mem_pen = cpu_memory_weight * (cpu_pen + mem_pen)

    cost_pen = cost_weight * (replica_state - min_replicas) / range_replicas

    reward = 1.0 - resp_pen - cpu_mem_pen - cost_pen
    return float(max(min(reward, 1.0), -1.0))


def build_row(
    i: int,
    cpu: float,
    mem: float,
    resp: float,
    replica: int,
    next_cpu: float,
    next_mem: float,
    next_resp: float,
    next_replica: int,
    *,
    min_replicas: int = 1,
    max_replicas: int = 50,
    min_cpu: float = 20.0,
    max_cpu: float = 90.0,
    min_memory: float = 20.0,
    max_memory: float = 90.0,
    max_response_time: float = 100.0,
    response_time_weight: float = 1.0,
    cpu_memory_weight: float = 0.5,
    cost_weight: float = 0.3,
) -> dict:
    # compute action (0-99) that would map to next_replica in KubernetesEnv.step()
    range_replicas = max(1, max_replicas - min_replicas)
    desired_pct = (next_replica - min_replicas) / range_replicas
    action = int(round(desired_pct * 99.0))
    action = max(0, min(99, action))

    # state values as percentages (cpu/mem already in 0-100 range here)
    cpu_pct = round(float(cpu), 2)
    mem_pct = round(float(mem), 2)
    resp_pct = round((float(resp) / 100.0) * 100.0, 2)  # max_response_time default 100

    # compute reward using environment logic
    reward = _env_reward(
        cpu_pct,
        mem_pct,
        resp_pct,
        replica,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
    )

    # compute discrete action (0-99) that would map to next_replica
    next_desired_pct = (next_replica - min_replicas) / range_replicas
    next_action = int(round(next_desired_pct * 99.0))
    next_action = max(0, min(99, next_action))

    next_state = {
        "cpu_usage": round(float(next_cpu), 2),
        "memory_usage": round(float(next_mem), 2),
        "response_time": round((float(next_resp) / max_response_time) * 100.0, 2),
        # last_action inside next_state should be the discrete action (0-99)
        "last_action": int(next_action),
    }

    # Map current replica count to discrete last_action (0-99) so offline
    # training that expects `replica` to be last_action remains consistent.
    current_pct = (replica - min_replicas) / range_replicas
    last_action_for_current = int(round(current_pct * 99.0))
    last_action_for_current = max(0, min(99, last_action_for_current))

    # compute reward using passed-in hyperparameters to ensure alignment
    reward = _env_reward(
        cpu_pct,
        mem_pct,
        resp_pct,
        replica,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        min_cpu=min_cpu,
        min_memory=min_memory,
        max_cpu=max_cpu,
        max_memory=max_memory,
        max_response_time=max_response_time,
        response_time_weight=response_time_weight,
        cpu_memory_weight=cpu_memory_weight,
        cost_weight=cost_weight,
    )

    return {
        "cpu_usage": f"{cpu_pct:.2f}",
        "memory_usage": f"{mem_pct:.2f}",
        "response_time": f"{resp_pct:.2f}",
        # `replica` column intentionally stores the agent-facing last_action (0-99)
        "replica": str(int(last_action_for_current)),
        # keep actual replica count available for clarity
        "replica_count": str(int(replica)),
        "action_pct": f"{desired_pct:.4f}",
        "action": str(int(action)),
        "reward": f"{reward:.4f}",
        "next_state": repr(next_state),
        # done handled by caller loop
        "done": "False",
    }


def generate(
    out: Path,
    rows: int = 1000,
    seed: int = 42,
    *,
    min_replicas: int = 1,
    max_replicas: int = 50,
    min_cpu: float = 20.0,
    max_cpu: float = 90.0,
    min_memory: float = 20.0,
    max_memory: float = 90.0,
    max_response_time: float = 100.0,
    response_time_weight: float = 1.0,
    cpu_memory_weight: float = 0.5,
    cost_weight: float = 0.3,
) -> None:
    random.seed(seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cpu_usage",
        "memory_usage",
        "response_time",
        # `replica` column stores the last_action (discrete 0-99) used by the agent
        "replica",
        # actual replica count for clarity
        "replica_count",
        # action as discrete 0-99 is already in `action`; action_pct is 0.0-1.0
        "action_pct",
        "action",
        "reward",
        "next_state",
        "done",
    ]

    # Start with some initial values
    cpu = 30.0 + random.random() * 20.0
    mem = 40.0 + random.random() * 30.0
    # replica_count is the current replica count (used to simulate next_resp etc.);
    # we initialize it to 1 (min_replicas) — the column written to CSV named
    # `replica` will contain the agent's last_action (0-99), not the replica count.
    replica = 1
    resp = 50.0 + random.random() * 20.0

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(rows):
            # simulate small fluctuations
            next_cpu = cpu + random.uniform(-5.0, 5.0) + math.sin(i * 0.03) * 2.0
            next_mem = mem + random.uniform(-3.0, 3.0) + math.cos(i * 0.02) * 1.5
            # response_time tends to go up with cpu and down with more replicas
            next_resp = (
                resp + (next_cpu - cpu) * 0.5 - (replica - 1) * random.uniform(0.2, 0.8)
            )

            # Occasionally change replica count (scale up/down)
            if random.random() < 0.07:
                next_replica = max(1, replica + random.choice([-1, 1]))
            else:
                next_replica = replica

            row = build_row(
                i,
                cpu,
                mem,
                resp,
                replica,
                next_cpu,
                next_mem,
                next_resp,
                next_replica,
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                min_cpu=min_cpu,
                max_cpu=max_cpu,
                min_memory=min_memory,
                max_memory=max_memory,
                max_response_time=max_response_time,
                response_time_weight=response_time_weight,
                cpu_memory_weight=cpu_memory_weight,
                cost_weight=cost_weight,
            )
            # mark last row as done
            if i == rows - 1:
                row["done"] = "True"
            else:
                row["done"] = "False"

            writer.writerow(row)

            # advance state
            cpu, mem, resp, replica = next_cpu, next_mem, next_resp, next_replica


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data/generated_training.csv")
    p.add_argument("--rows", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-replicas", type=int, default=1)
    p.add_argument("--max-replicas", type=int, default=50)
    p.add_argument("--min-cpu", type=float, default=20.0)
    p.add_argument("--max-cpu", type=float, default=90.0)
    p.add_argument("--min-memory", type=float, default=20.0)
    p.add_argument("--max-memory", type=float, default=90.0)
    p.add_argument("--max-response-time", type=float, default=100.0)
    p.add_argument("--response-time-weight", type=float, default=1.0)
    p.add_argument("--cpu-memory-weight", type=float, default=0.5)
    p.add_argument("--cost-weight", type=float, default=0.3)
    args = p.parse_args()
    generate(
        Path(args.out),
        rows=args.rows,
        seed=args.seed,
        min_replicas=args.min_replicas,
        max_replicas=args.max_replicas,
        min_cpu=args.min_cpu,
        max_cpu=args.max_cpu,
        min_memory=args.min_memory,
        max_memory=args.max_memory,
        max_response_time=args.max_response_time,
        response_time_weight=args.response_time_weight,
        cpu_memory_weight=args.cpu_memory_weight,
        cost_weight=args.cost_weight,
    )
