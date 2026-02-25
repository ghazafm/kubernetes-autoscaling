from logging import Logger
from typing import Dict, Optional


class Fuzzy:
    def __init__(self, logger: Optional[Logger] = None, max_replicas: int = 12):
        def _trapezoidal(x, a, b, c, d):
            if x < a or x > d:
                return 0.0
            elif b <= x <= c:
                return 1.0
            elif a < x < b:
                return (x - a) / (b - a) if (b - a) != 0 else 0.0
            else:
                return (d - x) / (d - c) if (d - c) != 0 else 0.0

        # All metrics use 0-100% scale for consistency
        # 3 membership levels: low, medium, high
        self.memberships = {
            "cpu_usage": {
                "low": lambda x: _trapezoidal(x, 0, 0, 20, 40),
                "medium": lambda x: _trapezoidal(x, 30, 45, 55, 70),
                "high": lambda x: _trapezoidal(x, 60, 80, 100, 100),
            },
            "memory_usage": {
                "low": lambda x: _trapezoidal(x, 0, 0, 20, 40),
                "medium": lambda x: _trapezoidal(x, 30, 45, 55, 70),
                "high": lambda x: _trapezoidal(x, 60, 80, 100, 100),
            },
            "response_time": {
                # Scale 0-300%; high reaches full membership at 100% and stays until 300%
                "low": lambda x: _trapezoidal(x, 0, 0, 20, 40),
                "medium": lambda x: _trapezoidal(x, 30, 45, 55, 70),
                "high": lambda x: _trapezoidal(x, 60, 100, 300, 300),
            },
            "last_action": {
                "low": lambda x: _trapezoidal(x, 0, 0, 20, 40),
                "medium": lambda x: _trapezoidal(x, 30, 45, 55, 70),
                "high": lambda x: _trapezoidal(x, 60, 80, 100, 100),
            },
            # Delta metrics: raw delta in percentage points (-100..100)
            # mapped to 0-100 via (delta + 100) / 2 before membership evaluation
            "delta_cpu": {
                "decreasing": lambda x: _trapezoidal(x, 0, 0, 30, 45),
                "stable": lambda x: _trapezoidal(x, 35, 45, 55, 65),
                "increasing": lambda x: _trapezoidal(x, 55, 70, 100, 100),
            },
            "delta_memory": {
                "decreasing": lambda x: _trapezoidal(x, 0, 0, 30, 45),
                "stable": lambda x: _trapezoidal(x, 35, 45, 55, 65),
                "increasing": lambda x: _trapezoidal(x, 55, 70, 100, 100),
            },
            "delta_response_time": {
                "decreasing": lambda x: _trapezoidal(x, 0, 0, 30, 45),
                "stable": lambda x: _trapezoidal(x, 35, 45, 55, 65),
                "increasing": lambda x: _trapezoidal(x, 55, 70, 100, 100),
            },
        }

        self.max_replicas = max_replicas
        self.logger = logger or Logger(__name__)

    def fuzzify(self, obs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        delta_metrics = {"delta_cpu", "delta_memory", "delta_response_time"}
        fuzzy_state = {}
        for metric, value in obs.items():
            if metric in self.memberships:
                if metric == "last_action":
                    # Normalize last_action (0..99) to 0-100% scale
                    fuzz_value = (value / 99.0) * 100.0
                elif metric in delta_metrics:
                    # Normalize delta to 0-100 scale, then clip so extreme deltas
                    # stay at the correct boundary membership (0 = fully decreasing,
                    # 100 = fully increasing)
                    fuzz_value = max(0.0, min(100.0, (value + 100.0) / 2.0))
                else:
                    fuzz_value = value
                fuzzy_state[metric] = {
                    label: fn(fuzz_value)
                    for label, fn in self.memberships[metric].items()
                }

        return fuzzy_state
