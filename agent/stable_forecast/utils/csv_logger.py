import csv
from datetime import datetime
from pathlib import Path
from typing import Optional


class TransitionLogger:
    """
    Logs RL transitions (s, a, r, s', done) to CSV for offline training.
    UPDATED: Supports 7-dimensional observation space (includes Deltas).
    """

    COLUMNS = [
        "timestamp",
        "episode",
        "step",
        # Current observation (state)
        "obs_action",
        "obs_cpu",
        "obs_memory",
        "obs_response_time",
        "obs_cpu_delta",  # NEW
        "obs_memory_delta",  # NEW
        "obs_rt_delta",  # NEW
        # Action taken
        "action",
        # Reward
        "reward",
        # Next observation (next state)
        "next_obs_action",
        "next_obs_cpu",
        "next_obs_memory",
        "next_obs_response_time",
        "next_obs_cpu_delta",  # NEW
        "next_obs_memory_delta",  # NEW
        "next_obs_rt_delta",  # NEW
        # Done flags
        "terminated",
        "truncated",
        # Info fields (raw metrics)
        "cpu",
        "memory",
        "response_time",
        "replicas",
    ]

    def __init__(
        self,
        log_dir: str = "data",
        prefix: str = "transitions",
        enabled: bool = True,
    ):
        self.enabled = enabled
        if not enabled:
            return

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.filepath = self.log_dir / f"{prefix}.csv"

        self.episode = 0
        self.step = 0
        self.last_obs = None

        # Create CSV file with headers
        with self.filepath.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writeheader()

    def on_reset(self, obs, info: dict):
        if not self.enabled:
            return

        self.episode += 1
        self.step = 0
        self.last_obs = obs.copy()

    def log_transition(
        self,
        obs,
        action: int,
        reward: float,
        next_obs,
        terminated: bool,
        truncated: bool,
        info: dict,
    ):
        if not self.enabled:
            return

        self.step += 1

        row = {
            "timestamp": datetime.now().isoformat(),
            "episode": self.episode,
            "step": self.step,
            # Current observation (Index 0-6)
            "obs_action": float(obs[0]),
            "obs_cpu": float(obs[1]),
            "obs_memory": float(obs[2]),
            "obs_response_time": float(obs[3]),
            "obs_cpu_delta": float(obs[4]),  # NEW
            "obs_memory_delta": float(obs[5]),  # NEW
            "obs_rt_delta": float(obs[6]),  # NEW
            # Action
            "action": action,
            # Reward
            "reward": reward,
            # Next observation (Index 0-6)
            "next_obs_action": float(next_obs[0]),
            "next_obs_cpu": float(next_obs[1]),
            "next_obs_memory": float(next_obs[2]),
            "next_obs_response_time": float(next_obs[3]),
            "next_obs_cpu_delta": float(next_obs[4]),  # NEW
            "next_obs_memory_delta": float(next_obs[5]),  # NEW
            "next_obs_rt_delta": float(next_obs[6]),  # NEW
            # Done flags
            "terminated": terminated,
            "truncated": truncated,
            # Info fields
            "cpu": info.get("cpu", 0.0),
            "memory": info.get("memory", 0.0),
            "response_time": info.get("response_time", 0.0),
            "replicas": info.get("replicas", 0),
        }

        with self.filepath.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writerow(row)

        self.last_obs = next_obs.copy()

    def get_filepath(self) -> Optional[Path]:
        """Return the path to the CSV file."""
        return self.filepath if self.enabled else None

    def get_stats(self) -> dict:
        """Return logging statistics."""
        if not self.enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "filepath": str(self.filepath),
            "episodes": self.episode,
            "total_steps": self._count_rows(),
        }

    def _count_rows(self) -> int:
        """Count total rows in CSV (excluding header)."""
        try:
            with self.filepath.open() as f:
                return sum(1 for _ in f) - 1  # Subtract header
        except Exception:
            return 0
