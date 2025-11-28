"""Generate a synthetic CSV compatible with agent/offline_train.py

Usage:
    python scripts/generate_training_csv.py --out data/generated_training.csv
    --rows 1000

The generated CSV columns:
  cpu_usage,memory_usage,response_time,replica,action,reward,next_state,done

next_state is a Python-dict-like string (single quotes) so the training script
can ast.literal_eval it.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / ".env")


def _calculate_rps_per_pod(
    cpu: float, mem: float, replica: int, *, max_replicas: int = 50
) -> float:
    """Calculate realistic RPS per pod based on system load and replica count.

    Logic:
    - Higher CPU/Memory → More requests being processed → Higher RPS
    - More replicas → Load distributed → Lower RPS per pod
    - Adds realistic noise and spike patterns
    """
    # Base RPS depends on resource utilization (0-1 normalized)
    load_factor = (min(cpu, 100.0) / 100.0 + min(mem, 100.0) / 100.0) / 2.0
    base_rps = 0.5 + load_factor * 9.0  # 0.5-9.5 RPS range

    # Adjust by replica count (more pods = distributed load)
    replica_factor = max(1.0, replica / 10.0)
    rps_per_pod = base_rps / replica_factor

    # Add realistic noise and occasional spikes
    noise = random.uniform(-0.3, 0.3)
    spike = random.uniform(0, 2.0) if random.random() < 0.05 else 0  # 5% spike chance
    rps_per_pod += noise + spike

    # Clamp to realistic range (0.1-15 RPS per pod)
    return max(0.1, min(15.0, rps_per_pod))


def _calculate_error_rate(
    cpu: float,
    mem: float,
    resp: float,
    replica: int,
    rps_per_pod: float,
    *,
    max_cpu: float = 90.0,
    max_memory: float = 90.0,
    max_response_time: float = 100.0,
) -> float:
    """Calculate realistic error rate based on system stress.

    Logic:
    - Errors spike when resources are exhausted (CPU/MEM > 90%)
    - High response time → timeouts → errors
    - High RPS per pod + few replicas → overload → errors
    - Exponential growth (cascading failures)
    """
    # Resource pressure (0-1 when exceeding thresholds)
    cpu_stress = max(0, (cpu - max_cpu) / 10.0)
    mem_stress = max(0, (mem - max_memory) / 10.0)

    # Response time stress (SLA violations)
    rt_stress = max(0, (resp - max_response_time) / max_response_time)

    # Overload stress (high RPS per pod indicates insufficient capacity)
    overload_stress = max(0, (rps_per_pod - 8.0) / 8.0)  # Stress when > 8 RPS/pod

    # Combine stress factors (weighted by severity)
    total_stress = (
        cpu_stress * 1.5  # CPU exhaustion is critical
        + mem_stress * 1.2  # Memory exhaustion causes OOM
        + rt_stress * 1.0  # RT violations indicate problems
        + overload_stress * 0.8  # Overload leads to errors
    )

    # Error rate increases exponentially (cascading failures)
    error_rate = (total_stress**2) * 3.0  # 0-10% range

    # Baseline error rate (even healthy systems have occasional errors)
    baseline = random.uniform(0, 0.05)  # 0-0.05% noise
    error_rate += baseline

    # Occasional error spikes (network issues, deployment, etc.)
    if random.random() < 0.03:  # 3% chance of error spike
        error_rate += random.uniform(0.5, 2.0)

    # Clamp to 0-10% range
    return max(0.0, min(10.0, error_rate))


def _env_reward(
    cpu: float,
    mem: float,
    resp: float,
    replica_state: int,
    error_rate: float,
    *,
    min_replicas: int = 1,
    max_replicas: int = 50,
    min_cpu: float = 20.0,
    min_memory: float = 20.0,
    max_cpu: float = 90.0,
    max_memory: float = 90.0,
    max_response_time: float = 100.0,
    response_time_weight: float = 1.5,
    error_rate_weight: float = 1.0,
    cpu_memory_weight: float = 0.5,
    cost_weight: float = 0.3,
) -> float:
    # Mirror environment._calculate_reward() from KubernetesEnv
    def _cpu_mem_penalty(
        value: float, low: float, high: float, min_tol_pct: float = 0.01
    ) -> float:
        # Inside the allowed band => no penalty
        if low <= value <= high:
            return 0.0

        # Distance outside the band
        distance = low - value if value < low else value - high

        bandwidth = max(high - low, 1e-6)
        min_tol = max(min_tol_pct * bandwidth, 1e-6)

        normalized = distance / (bandwidth + min_tol)
        penalty = min(1.0, normalized * normalized)
        return float(penalty)

    range_replicas = max(1, max_replicas - min_replicas)
    response_time_percentage = (resp / max_response_time) * 100.0
    response_time_percentage = min(response_time_percentage, 1000.0)

    cpu_pen = _cpu_mem_penalty(cpu, min_cpu, max_cpu)
    mem_pen = _cpu_mem_penalty(mem, min_memory, max_memory)

    RESPONSE_TIME_HIGH_THRESHOLD = 80.0
    RESPONSE_TIME_VIOLATION_THRESHOLD = 100.0
    MAX_RESPONSE_PENALTY = 2.0

    if response_time_percentage <= RESPONSE_TIME_HIGH_THRESHOLD:
        resp_pen = 0.0
    elif response_time_percentage <= RESPONSE_TIME_VIOLATION_THRESHOLD:
        resp_pen = (response_time_percentage - RESPONSE_TIME_HIGH_THRESHOLD) / (
            RESPONSE_TIME_VIOLATION_THRESHOLD - RESPONSE_TIME_HIGH_THRESHOLD
        )
    else:
        over = (
            response_time_percentage - RESPONSE_TIME_VIOLATION_THRESHOLD
        ) / RESPONSE_TIME_VIOLATION_THRESHOLD
        resp_pen = 1.0 + over

    resp_pen = max(0.0, min(resp_pen, MAX_RESPONSE_PENALTY))

    ERROR_RATE_MAX = 100.0
    error_pen = min(max(error_rate, 0.0) / ERROR_RATE_MAX, 1.0)

    weighted_resp_pen = response_time_weight * resp_pen
    weighted_error_pen = error_rate_weight * error_pen
    weighted_cpu_mem_pen = cpu_memory_weight * (cpu_pen + mem_pen)
    # cost_pen uses last action percent; caller should pass replica_state as last_action (0-99)
    weighted_cost_pen = cost_weight * (replica_state / 100.0)

    total_penalty = (
        weighted_resp_pen
        + weighted_error_pen
        + weighted_cpu_mem_pen
        + weighted_cost_pen
    )

    max_possible_penalty = (
        response_time_weight + error_rate_weight + cpu_memory_weight * 2.0 + cost_weight
    )

    normalized_penalty = min(total_penalty / max_possible_penalty, 1.0)

    reward = 1.0 - 2.0 * normalized_penalty

    # Stability penalty: if this step had an applied scaling delta, callers can subtract
    # stability_penalty externally. Generator will subtract it when action caused replica change.
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
    response_time_weight: float = 1.5,
    error_rate_weight: float = 1.0,
    cpu_memory_weight: float = 0.5,
    cost_weight: float = 0.3,
) -> dict:
    # compute action (0-99) that would map to next_replica in KubernetesEnv.step()
    range_replicas = max(1, max_replicas - min_replicas)
    desired_pct = (next_replica - min_replicas) / range_replicas
    action = round(desired_pct * 99.0)
    action = max(0, min(99, action))

    # state values as percentages (cpu/mem already in 0-100 range here)
    cpu_pct = round(float(cpu), 2)
    mem_pct = round(float(mem), 2)
    # Convert response time (in ms) to percentage of max_response_time for observation
    resp_pct = round((float(resp) / max_response_time) * 100.0, 2)

    # Calculate current RPS per pod and error rate BEFORE reward calculation
    current_rps_per_pod = _calculate_rps_per_pod(
        cpu, mem, replica, max_replicas=max_replicas
    )

    current_error_rate = _calculate_error_rate(
        cpu,
        mem,
        resp,
        replica,
        current_rps_per_pod,
        max_cpu=max_cpu,
        max_memory=max_memory,
        max_response_time=max_response_time,
    )

    # compute reward using environment logic (pass raw milliseconds for resp)
    reward = _env_reward(
        cpu_pct,
        mem_pct,
        resp,  # Pass raw milliseconds, _env_reward will convert to percentage
        replica,
        current_error_rate,  # NEW: Pass error_rate for penalty calculation
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        min_cpu=min_cpu,
        max_cpu=max_cpu,
        min_memory=min_memory,
        max_memory=max_memory,
        max_response_time=max_response_time,
        response_time_weight=response_time_weight,
        error_rate_weight=error_rate_weight,  # NEW: Pass error_rate_weight
        cpu_memory_weight=cpu_memory_weight,
        cost_weight=cost_weight,
    )

    # compute discrete action (0-99) that would map to next_replica
    next_desired_pct = (next_replica - min_replicas) / range_replicas
    next_action = round(next_desired_pct * 99.0)
    next_action = max(0, min(99, next_action))

    # Calculate current replica percentage
    current_replica_pct = (replica - min_replicas) / range_replicas * 100.0
    next_replica_pct = (next_replica - min_replicas) / range_replicas * 100.0

    # Calculate deltas (change from current to next)
    cpu_delta = next_cpu - cpu
    mem_delta = next_mem - mem
    resp_pct_current = (resp / max_response_time) * 100.0
    resp_pct_next = (next_resp / max_response_time) * 100.0
    rt_delta = resp_pct_next - resp_pct_current

    # Calculate scaling direction (0=down, 0.5=same, 1=up)
    if next_replica > replica:
        scaling_direction = 1.0
    elif next_replica < replica:
        scaling_direction = 0.0
    else:
        scaling_direction = 0.5

    # Time in state (normalized, simulate staying at replica for a few steps)
    time_in_state = random.uniform(0.0, 1.0)

    # NEW: Calculate RPS per pod for current and next state
    current_rps_per_pod = _calculate_rps_per_pod(
        cpu, mem, replica, max_replicas=max_replicas
    )
    next_rps_per_pod = _calculate_rps_per_pod(
        next_cpu, next_mem, next_replica, max_replicas=max_replicas
    )

    # NEW: Calculate RPS delta
    rps_delta = next_rps_per_pod - current_rps_per_pod

    # NEW: Calculate error rate for next state only (current already calculated above for reward)
    next_error_rate = _calculate_error_rate(
        next_cpu,
        next_mem,
        next_resp,
        next_replica,
        next_rps_per_pod,
        max_cpu=max_cpu,
        max_memory=max_memory,
        max_response_time=max_response_time,
    )

    next_state = {
        "cpu_usage": round(float(next_cpu), 2),
        "memory_usage": round(float(next_mem), 2),
        "response_time": round((float(next_resp) / max_response_time) * 100.0, 2),
        "current_replica_pct": round(next_replica_pct, 2),
        "last_action": int(next_action),
        "cpu_delta": round(cpu_delta, 2),
        "memory_delta": round(mem_delta, 2),
        "rt_delta": round(rt_delta, 2),
        "time_in_state": round(time_in_state, 4),
        "scaling_direction": scaling_direction,
        "rps_per_pod": round(next_rps_per_pod, 2),  # NEW
        "rps_delta": round(rps_delta, 2),  # NEW
        "error_rate": round(next_error_rate, 2),  # NEW
    }

    # Map current replica count to discrete last_action (0-99) so offline
    # training that expects `replica` to be last_action remains consistent.
    current_pct = (replica - min_replicas) / range_replicas
    last_action_for_current = round(current_pct * 99.0)
    last_action_for_current = max(0, min(99, last_action_for_current))

    return {
        "cpu_usage": f"{cpu_pct:.2f}",
        "memory_usage": f"{mem_pct:.2f}",
        "response_time": f"{resp_pct:.2f}",
        "current_replica_pct": f"{current_replica_pct:.2f}",
        # `replica` column intentionally stores the agent-facing last_action (0-99)
        "replica": str(int(last_action_for_current)),
        # keep actual replica count available for clarity
        "replica_count": str(int(replica)),
        "cpu_delta": f"{cpu_delta:.2f}",
        "memory_delta": f"{mem_delta:.2f}",
        "rt_delta": f"{rt_delta:.2f}",
        "time_in_state": f"{time_in_state:.4f}",
        "scaling_direction": f"{scaling_direction:.1f}",
        # NEW: Load indicators (scale-independent)
        "rps_per_pod": f"{current_rps_per_pod:.2f}",
        "rps_delta": f"{rps_delta:.2f}",
        "error_rate": f"{current_error_rate:.2f}",
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
    response_time_weight: float = 1.5,
    error_rate_weight: float = 1.0,
    cpu_memory_weight: float = 0.5,
    cost_weight: float = 0.3,
) -> None:
    random.seed(seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cpu_usage",
        "memory_usage",
        "response_time",
        "current_replica_pct",
        # `replica` column stores the last_action (discrete 0-99) used by the agent
        "replica",
        # actual replica count for clarity
        "replica_count",
        "cpu_delta",
        "memory_delta",
        "rt_delta",
        "time_in_state",
        "scaling_direction",
        # NEW: Load indicators (scale-independent)
        "rps_per_pod",
        "rps_delta",
        "error_rate",
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
    replica = min_replicas
    resp = 30.0 + random.random() * 40.0  # Start with reasonable response time

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(rows):
            # simulate small fluctuations with proper clamping
            next_cpu = cpu + random.uniform(-5.0, 5.0) + math.sin(i * 0.03) * 2.0
            next_cpu = max(
                min_cpu - 10, min(max_cpu + 10, next_cpu)
            )  # Allow some overshoot

            next_mem = mem + random.uniform(-3.0, 3.0) + math.cos(i * 0.02) * 1.5
            next_mem = max(
                min_memory - 10, min(max_memory + 10, next_mem)
            )  # Allow some overshoot

            # response_time calculation with proper bounds
            # Higher CPU/memory and lower replicas = higher response time
            cpu_pressure = max(0, (next_cpu - max_cpu) / 100.0)
            mem_pressure = max(0, (next_mem - max_memory) / 100.0)
            replica_factor = max(
                1.0, replica / 10.0
            )  # More replicas = lower response time

            base_resp = 20.0 + cpu_pressure * 30.0 + mem_pressure * 20.0
            next_resp = (base_resp / replica_factor) + random.uniform(-5.0, 5.0)
            next_resp = max(
                5.0, min(max_response_time * 1.5, next_resp)
            )  # Keep response time reasonable

            # Occasionally change replica count (scale up/down) with bounds checking
            if random.random() < 0.07:
                next_replica = max(
                    min_replicas, min(max_replicas, replica + random.choice([-1, 1]))
                )
            else:
                next_replica = replica

            # Clamp values to realistic ranges before building row
            cpu = max(0.0, min(100.0, cpu))
            mem = max(0.0, min(100.0, mem))
            resp = max(0.0, resp)
            next_cpu = max(0.0, min(100.0, next_cpu))
            next_mem = max(0.0, min(100.0, next_mem))
            next_resp = max(0.0, next_resp)

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
                error_rate_weight=error_rate_weight,
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
    # Load defaults from .env file
    default_min_replicas = int(os.getenv("MIN_REPLICAS", "1"))
    default_max_replicas = int(os.getenv("MAX_REPLICAS", "50"))
    default_min_cpu = float(os.getenv("MIN_CPU", "20.0"))
    default_max_cpu = float(os.getenv("MAX_CPU", "90.0"))
    default_min_memory = float(os.getenv("MIN_MEMORY", "20.0"))
    default_max_memory = float(os.getenv("MAX_MEMORY", "90.0"))
    default_max_response_time = float(os.getenv("MAX_RESPONSE_TIME", "100.0"))
    default_response_time_weight = float(os.getenv("RESPONSE_TIME_WEIGHT", "1.5"))
    default_error_rate_weight = float(os.getenv("ERROR_RATE_WEIGHT", "1.0"))
    default_cpu_memory_weight = float(os.getenv("CPU_MEMORY_WEIGHT", "0.5"))
    default_cost_weight = float(os.getenv("COST_WEIGHT", "0.15"))

    p = argparse.ArgumentParser(
        description="Generate synthetic training data aligned with .env configuration"
    )
    p.add_argument("--out", type=str, default="data/generated_training.csv")
    p.add_argument("--rows", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-replicas", type=int, default=default_min_replicas)
    p.add_argument("--max-replicas", type=int, default=default_max_replicas)
    p.add_argument("--min-cpu", type=float, default=default_min_cpu)
    p.add_argument("--max-cpu", type=float, default=default_max_cpu)
    p.add_argument("--min-memory", type=float, default=default_min_memory)
    p.add_argument("--max-memory", type=float, default=default_max_memory)
    p.add_argument("--max-response-time", type=float, default=default_max_response_time)
    p.add_argument(
        "--response-time-weight", type=float, default=default_response_time_weight
    )
    p.add_argument("--error-rate-weight", type=float, default=default_error_rate_weight)
    p.add_argument("--cpu-memory-weight", type=float, default=default_cpu_memory_weight)
    p.add_argument("--cost-weight", type=float, default=default_cost_weight)
    args = p.parse_args()

    print("Generating training data with configuration from .env:")
    print(f"  MIN_REPLICAS={args.min_replicas}, MAX_REPLICAS={args.max_replicas}")
    print(f"  MIN_CPU={args.min_cpu}%, MAX_CPU={args.max_cpu}%")
    print(f"  MIN_MEMORY={args.min_memory}%, MAX_MEMORY={args.max_memory}%")
    print(f"  MAX_RESPONSE_TIME={args.max_response_time}ms")
    print(
        f"  Weights: RT={args.response_time_weight}, "
        f"ERROR={args.error_rate_weight}, "
        f"CPU/MEM={args.cpu_memory_weight}, "
        f"COST={args.cost_weight}"
    )
    print(f"  Generating {args.rows} rows → {args.out}")

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
        error_rate_weight=args.error_rate_weight,
        cpu_memory_weight=args.cpu_memory_weight,
        cost_weight=args.cost_weight,
    )
    print("✅ Training data generated successfully!")
