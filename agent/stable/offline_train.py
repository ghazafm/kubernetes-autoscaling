import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from environment import KubernetesEnv
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from utils import setup_logger

load_dotenv()


class OfflineEnv(Env):
    """Dummy environment for offline training - only defines spaces."""

    def __init__(
        self,
        weight_response_time: float = 1.0,
        weight_cost: float = 1.0,
        max_cpu: float = 80.0,
        min_cpu: float = 20.0,
        max_memory: float = 60.0,
        min_memory: float = 20.0,
    ):
        self.action_space = Discrete(100)
        self.observation_space = Box(
            low=np.array([0.0, 0.0, 0.0, -2.0, -2.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.weight_response_time = weight_response_time
        self.weight_cost = weight_cost
        self.min_cpu: float = min_cpu
        self.min_memory: float = min_memory
        self.max_cpu: float = max_cpu
        self.max_memory: float = max_memory

        self.data = None
        self._index = 0
        self.observations = np.zeros(6, dtype=np.float32)

    calculate_reward = KubernetesEnv.calculate_reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # reset dataset cursor
        self._index = 0
        if self.data is not None and len(self.data) > 0:
            row = self.data.iloc[0]
            obs = np.array(
                [
                    float(row["obs_action"]),
                    float(row["obs_cpu_relative"]),
                    float(row["obs_memory_relative"]),
                    float(row["obs_cpu_distance"]),
                    float(row["obs_memory_distance"]),
                    float(row["obs_response_time"]),
                ],
                dtype=np.float32,
            )
            self.observations = obs
            return obs, {}

        return np.zeros(6, dtype=np.float32), {}

    def step(self, action):
        """
        Return the next transition from the offline dataset (if provided).
        The env ignores the agent's action and yields the recorded next state,
        reward and done flags so training can proceed deterministically from
        existing transitions.
        """
        if self.data is None or self._index >= len(self.data):
            # no data left — return a terminal empty transition
            obs = np.zeros(6, dtype=np.float32)
            return obs, 0.0, True, False, {}

        row = self.data.iloc[self._index]

        obs = np.array(
            [
                float(row["obs_action"]),
                float(row["obs_cpu_relative"]),
                float(row["obs_memory_relative"]),
                float(row["obs_cpu_distance"]),
                float(row["obs_memory_distance"]),
                float(row["obs_response_time"]),
            ],
            dtype=np.float32,
        )

        next_obs = np.array(
            [
                float(row["next_obs_action"]),
                float(row["next_obs_cpu_relative"]),
                float(row["next_obs_memory_relative"]),
                float(row["next_obs_cpu_distance"]),
                float(row["next_obs_memory_distance"]),
                float(row["next_obs_response_time"]),
            ],
            dtype=np.float32,
        )

        reward = float(row.get("reward", 0.0))
        terminated = bool(row.get("terminated", False))
        truncated = bool(row.get("truncated", False))

        info = {
            "cpu": float(row.get("cpu", 0.0)),
            "memory": float(row.get("memory", 0.0)),
            "response_time": float(row.get("response_time", 0.0)),
            "replicas": int(row.get("replicas", 0)),
        }

        # Use previous observation (the env's last returned observation) to
        # decide whether metrics are stale, mirroring the logic in
        # `environment.KubernetesEnv.step()`.
        prev_obs = self.observations.copy()
        RT_STALE_THRESHOLD = 0.3

        missing_cpu = info["cpu"] <= 0.0
        missing_mem = info["memory"] <= 0.0
        is_broken = (missing_cpu or missing_mem) and prev_obs[5] >= RT_STALE_THRESHOLD

        if is_broken:
            recorded_action = int(obs[0] * 99)
            est_cpu, est_mem, est_rt = self.estimate_metrics(
                prev_obs=prev_obs,
                action=recorded_action,
                cpu=info["cpu"],
                memory=info["memory"],
                response_time=info["response_time"],
            )

            info["cpu"] = est_cpu
            info["memory"] = est_mem
            info["response_time"] = est_rt

            # Recompute normalized distances using the same helper as the
            # online environment so offline and online observations stay
            # consistent.
            cpu_relative, memory_relative, cpu_distance, memory_distance = (
                KubernetesEnv.calculate_distance(self, est_cpu, est_mem)
            )

            next_obs[1] = float(np.clip(cpu_relative, 0.0, 1.0))
            next_obs[2] = float(np.clip(memory_relative, 0.0, 1.0))
            next_obs[3] = float(np.clip(cpu_distance, -2.0, 2.0))
            next_obs[4] = float(np.clip(memory_distance, -2.0, 2.0))
            next_obs[5] = float(np.clip(est_rt / 100.0, 0.0, 3.0))

        # advance cursor
        self._index += 1
        self.observations = next_obs

        return next_obs, reward, terminated, truncated, info

    def calculate_distance(self, cpu: float, memory: float) -> tuple[float, float]:
        cpu_bandwidth = self.max_cpu - self.min_cpu

        if cpu < self.min_cpu:
            cpu_distance = (cpu - self.min_cpu) / cpu_bandwidth
        elif cpu > self.max_cpu:
            cpu_distance = (cpu - self.max_cpu) / cpu_bandwidth
        else:
            cpu_distance = 0.0

        cpu_relative = (cpu - self.min_cpu) / cpu_bandwidth

        memory_bandwidth = self.max_memory - self.min_memory

        if memory < self.min_memory:
            memory_distance = (memory - self.min_memory) / memory_bandwidth
        elif memory > self.max_memory:
            memory_distance = (memory - self.max_memory) / memory_bandwidth
        else:
            memory_distance = 0.0

        memory_relative = (memory - self.min_memory) / memory_bandwidth

        return cpu_relative, memory_relative, cpu_distance, memory_distance

    def estimate_metrics(
        self,
        prev_obs,
        action: int,
        cpu: float,
        memory: float,
        response_time: float,
    ) -> tuple[float, float, float]:
        prev_action = prev_obs[0]
        current_action = action / 99.0
        action_change = current_action - prev_action

        prev_cpu_rel = prev_obs[1]
        prev_mem_rel = prev_obs[2]

        cpu_bandwidth = self.max_cpu - self.min_cpu
        mem_bandwidth = self.max_memory - self.min_memory
        prev_cpu = prev_cpu_rel * cpu_bandwidth + self.min_cpu
        prev_mem = prev_mem_rel * mem_bandwidth + self.min_memory
        prev_rt = float(prev_obs[5]) * 100.0

        scale_factor = 0.5

        missing_cpu = cpu <= 0.0
        missing_mem = memory <= 0.0
        missing_rt = response_time <= 0.0

        if missing_cpu:
            cpu = prev_cpu * (1 - action_change * scale_factor)
            cpu = max(0.01, cpu)

        if missing_mem:
            memory = prev_mem * (1 - action_change * scale_factor)
            memory = max(0.01, memory)

        if missing_rt:
            response_time = prev_rt * (1 - action_change * scale_factor)
            response_time = float(np.clip(response_time, 0.01, 1000.0))

        return cpu, memory, response_time


def load_csv_data(csv_paths: list[str]) -> pd.DataFrame:
    """Load and concatenate multiple CSV files."""
    dfs = [pd.read_csv(p) for p in csv_paths]
    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    now = datetime.now().strftime("%Y-%m-%d-%H-%M")
    logger, log_dir = setup_logger(
        "offline_train", log_level=os.getenv("LOG_LEVEL", "INFO"), log_to_file=True
    )

    csv_paths_str = os.getenv("CSV_PATHS", "")
    if csv_paths_str:
        csv_paths = [p.strip() for p in csv_paths_str.split(",")]
    else:
        data_dir = Path("data")
        csv_paths = [str(p) for p in data_dir.glob("*.csv")]

    logger.info(f"Loading {len(csv_paths)} CSV files: {csv_paths}")
    df = load_csv_data(csv_paths)
    logger.info(f"Loaded {len(df)} transitions")

    # CPU/memory bounds for stale data recalculation
    min_cpu = float(os.getenv("MIN_CPU", "20.0"))
    max_cpu = float(os.getenv("MAX_CPU", "80.0"))
    min_memory = float(os.getenv("MIN_MEMORY", "20.0"))
    max_memory = float(os.getenv("MAX_MEMORY", "60.0"))

    env = OfflineEnv(
        weight_response_time=float(os.getenv("WEIGHT_RESPONSE_TIME", "1.0")),
        weight_cost=float(os.getenv("WEIGHT_COST", "1.0")),
        max_cpu=max_cpu,
        min_cpu=min_cpu,
        max_memory=max_memory,
        min_memory=min_memory,
    )
    # attach loaded dataset so env.step yields offline transitions
    env.data = df.reset_index(drop=True)

    note = os.getenv("NOTE", "offline")
    model_dir = Path(f"model/{now}_{note}")
    model_dir.mkdir(parents=True, exist_ok=True)
    iteration = int(os.getenv("ITERATION"))
    BASE_EPISODES = 10
    num_episodes = int(os.getenv("EPISODE", BASE_EPISODES))

    additional_timesteps = num_episodes * iteration
    total_timesteps = additional_timesteps

    model = DQN(
        policy="MlpPolicy",
        env=env,
        policy_kwargs={"net_arch": [256, 256, 128]},
        learning_rate=1e-4,
        gamma=0.99,
        buffer_size=100_000,
        learning_starts=iteration * 3,
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=iteration,
        exploration_fraction=0.4,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        max_grad_norm=1,
        verbose=1,
        tensorboard_log=log_dir,
        seed=42,
        device="auto",
    )

    logger.info("Populating replay buffer with stale data correction...")

    logger.info(f"Training for {iteration} gradient steps")

    checkpoint_callback = CheckpointCallback(
        save_freq=iteration * 2,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="dqn_autoscaler",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=1,
    )

    model.learn(
        total_timesteps=len(df),
        callback=checkpoint_callback,
        reset_num_timesteps=False,
        progress_bar=True,
        tb_log_name="DQN",
    )

    final_path = model_dir / "final" / "model"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(final_path)
    logger.info(f"Model saved to {final_path}")

    env.close()
