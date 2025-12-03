import csv
from datetime import datetime
from pathlib import Path
from typing import Optional


class TransitionLogger:
    """
    Logs RL transitions (s, a, r, s', done) to CSV for offline training.

    CSV columns:
    - timestamp: When the transition occurred
    - episode: Episode number
    - step: Step within episode
    - obs_action, obs_cpu_relative, obs_memory_relative, obs_cpu_distance,
      obs_memory_distance, obs_response_time: Current state (6D)
    - action: Action taken (0-99)
    - reward: Reward received
    - next_obs_*: Next state (6D)
    - terminated: Whether episode ended naturally
    - truncated: Whether episode was truncated
    - info fields: cpu, memory, response_time, replicas, etc.
    """

    COLUMNS = [
        "timestamp",
        "episode",
        "step",
        # Current observation (state)
        "obs_action",
        "obs_cpu_relative",
        "obs_memory_relative",
        "obs_cpu_distance",
        "obs_memory_distance",
        "obs_response_time",
        # Action taken
        "action",
        # Reward
        "reward",
        # Next observation (next state)
        "next_obs_action",
        "next_obs_cpu_relative",
        "next_obs_memory_relative",
        "next_obs_cpu_distance",
        "next_obs_memory_distance",
        "next_obs_response_time",
        # Done flags
        "terminated",
        "truncated",
        # Info fields (raw metrics)
        "cpu",
        "memory",
        "response_time",
        "replicas",
        "cpu_relative",
        "memory_relative",
        "cpu_distance",
        "memory_distance",
    ]

    def __init__(
        self,
        log_dir: str = "data",
        prefix: str = "transitions",
        enabled: bool = True,
    ):
        """
        Initialize the transition logger.

        Args:
            log_dir: Directory to save CSV files
            prefix: Prefix for the CSV filename
            enabled: Whether logging is enabled
        """
        self.enabled = enabled
        if not enabled:
            return

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.filepath = self.log_dir / f"{prefix}_{timestamp}.csv"

        self.episode = 0
        self.step = 0
        self.last_obs = None

        # Create CSV file with headers
        with self.filepath.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writeheader()

    def on_reset(self, obs, info: dict):
        """Called when environment resets. Records initial observation."""
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
        """
        Log a single transition to CSV.

        Args:
            obs: Current observation (before action)
            action: Action taken
            reward: Reward received
            next_obs: Next observation (after action)
            terminated: Whether episode ended naturally
            truncated: Whether episode was truncated
            info: Additional info dict from environment
        """
        if not self.enabled:
            return

        self.step += 1

        row = {
            "timestamp": datetime.now().isoformat(),
            "episode": self.episode,
            "step": self.step,
            # Current observation
            "obs_action": float(obs[0]),
            "obs_cpu_relative": float(obs[1]),
            "obs_memory_relative": float(obs[2]),
            "obs_cpu_distance": float(obs[3]),
            "obs_memory_distance": float(obs[4]),
            "obs_response_time": float(obs[5]),
            # Action
            "action": action,
            # Reward
            "reward": reward,
            # Next observation
            "next_obs_action": float(next_obs[0]),
            "next_obs_cpu_relative": float(next_obs[1]),
            "next_obs_memory_relative": float(next_obs[2]),
            "next_obs_cpu_distance": float(next_obs[3]),
            "next_obs_memory_distance": float(next_obs[4]),
            "next_obs_response_time": float(next_obs[5]),
            # Done flags
            "terminated": terminated,
            "truncated": truncated,
            # Info fields
            "cpu": info.get("cpu", 0.0),
            "memory": info.get("memory", 0.0),
            "response_time": info.get("response_time", 0.0),
            "replicas": info.get("replicas", 0),
            "cpu_relative": info.get("cpu_relative", 0.0),
            "memory_relative": info.get("memory_relative", 0.0),
            "cpu_distance": info.get("cpu_distance", 0.0),
            "memory_distance": info.get("memory_distance", 0.0),
        }

        with self.filepath.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writerow(row)

        # Update last observation for next transition
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
