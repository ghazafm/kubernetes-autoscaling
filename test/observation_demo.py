"""
Simple demo script to show how cpu_dist and memory_dist are computed
using the new edge-relative formula implemented in the environment.

This script mirrors only the observation-distance logic so it can run
without Kubernetes/Prometheus dependencies.
"""

from __future__ import annotations

EPSILON = 1e-6


def compute_rel_and_dist(value: float, min_v: float, max_v: float) -> dict:
    """Compute rel_pct_clipped, in_band and distance using same rules
    as `KubernetesEnv._get_observation` after the recent change.

    Returns a dict with keys: rel_pct_clipped, in_band, dist
    """
    # relative position inside band (0..100 clipped)
    band = max(max_v - min_v, EPSILON)
    raw_rel = (value - min_v) / band
    rel_pct = raw_rel * 100.0
    rel_pct_clipped = min(max(rel_pct, 0.0), 100.0)

    in_band = 1.0 if (min_v <= value <= max_v) else 0.0

    if in_band:
        dist = 0.0
    elif value < min_v:
        denom = max(min_v, EPSILON)
        dist = min(1.0, (min_v - value) / denom)
    else:
        denom = max(100.0 - max_v, EPSILON)
        dist = min(1.0, (value - max_v) / denom)

    return {
        "rel_pct_clipped": rel_pct_clipped,
        "in_band": in_band,
        "dist": dist,
    }


def print_example(cpu_value: float, min_cpu: float, max_cpu: float) -> None:
    res = compute_rel_and_dist(cpu_value, min_cpu, max_cpu)
    print(
        f"value={cpu_value:5.1f}  min={min_cpu:5.1f}  max={max_cpu:5.1f}  "
        f"rel_pct={res['rel_pct_clipped']:6.2f}  in_band={int(res['in_band'])}  "
        f"dist={res['dist']:5.3f}"
    )


def main() -> None:
    print("Demonstration of edge-relative cpu_dist / memory_dist")
    print("(format: value, min, max -> rel_pct, in_band, dist)")
    print("-")

    # Examples from the user
    print("User examples:")
    print_example(5.0, 10.0, 90.0)  # min=10, value=5 -> dist 0.5
    print_example(10.0, 20.0, 90.0)  # min=20, value=10 -> dist 0.5
    print_example(5.0, 20.0, 90.0)  # min=20, value=5 -> dist 0.75

    print("-")
    # Some edge cases
    print("Edge cases:")
    print_example(0.0, 0.0, 90.0)  # min=0, below-min should be handled
    print_example(100.0, 10.0, 100.0)  # max=100, above-max handled
    print_example(95.0, 10.0, 90.0)  # slightly above max

    print("-")
    # Table of values for min variations
    print("Table: value=5 for different mins (max fixed at 90):")
    for m in (5, 10, 15, 20, 25, 50):
        print_example(5.0, float(m), 90.0)


if __name__ == "__main__":
    main()
