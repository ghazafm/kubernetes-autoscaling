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
                "low": lambda x: _trapezoidal(x, 0, 0, 20, 40),
                "medium": lambda x: _trapezoidal(x, 30, 45, 55, 70),
                "high": lambda x: _trapezoidal(x, 60, 80, 100, 100),
            },
            "last_action": {
                "low": lambda x: _trapezoidal(x, 0, 0, 20, 40),
                "medium": lambda x: _trapezoidal(x, 30, 45, 55, 70),
                "high": lambda x: _trapezoidal(x, 60, 80, 100, 100),
            },
        }

        self.max_replicas = max_replicas
        self.logger = logger or Logger(__name__)

    def fuzzify(self, obs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        fuzzy_state = {}
        for metric, value in obs.items():
            if metric in self.memberships:
                # Normalize last_action (0..99) to 0-100% scale
                fuzz_value = (value / 99.0) * 100.0 if metric == "last_action" else value
                fuzzy_state[metric] = {
                    label: fn(fuzz_value)
                    for label, fn in self.memberships[metric].items()
                }

        return fuzzy_state
