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

    def __init__(self, weight_response_time: float = 1.0, weight_cost: float = 1.0):
        self.action_space = Discrete(100)
        self.observation_space = Box(
            low=np.array([0.0, 0.0, 0.0, -2.0, -2.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.weight_response_time = weight_response_time
        self.weight_cost = weight_cost

    calculate_reward = KubernetesEnv.calculate_reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(6, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(6, dtype=np.float32), 0.0, False, False, {}


def load_csv_data(csv_paths: list[str]) -> pd.DataFrame:
    """Load and concatenate multiple CSV files."""
    dfs = [pd.read_csv(p) for p in csv_paths]
    return pd.concat(dfs, ignore_index=True)


def populate_replay_buffer(
    model: DQN,
    df: pd.DataFrame,
    env: OfflineEnv,
    min_cpu: float,
    max_cpu: float,
    min_memory: float,
    max_memory: float,
):
    """Add transitions from DataFrame to replay buffer, recalculating rewards."""
    obs_cols = [
        "obs_action",
        "obs_cpu_relative",
        "obs_memory_relative",
        "obs_cpu_distance",
        "obs_memory_distance",
        "obs_response_time",
    ]

    cpu_bandwidth = max_cpu - min_cpu
    mem_bandwidth = max_memory - min_memory

    for _, row in df.iterrows():
        obs = np.array([row[c] for c in obs_cols], dtype=np.float32)
        action = int(row["action"])
        terminated = bool(row["terminated"])
        truncated = bool(row["truncated"])
        response_time = float(row["response_time"])

        # Get raw cpu/memory from CSV
        cpu = float(row["cpu"])
        memory = float(row["memory"])

        # Stale data handling - recalculate cpu/memory if stale
        RT_STALE_THRESHOLD = 0.3
        is_stale = cpu <= 0.0 and memory <= 0.0 and obs[5] >= RT_STALE_THRESHOLD

        if is_stale:
            prev_action = obs[0]
            current_action = action / 99.0
            action_change = current_action - prev_action

            prev_cpu = obs[1] * cpu_bandwidth + min_cpu
            prev_mem = obs[2] * mem_bandwidth + min_memory

            scale_factor = 0.5
            cpu = prev_cpu * (1 - action_change * scale_factor)
            memory = prev_mem * (1 - action_change * scale_factor)

            cpu = max(0.01, cpu)
            memory = max(0.01, memory)

        # Recalculate distance with corrected cpu/memory
        if cpu < min_cpu:
            cpu_distance = (cpu - min_cpu) / cpu_bandwidth
        elif cpu > max_cpu:
            cpu_distance = (cpu - max_cpu) / cpu_bandwidth
        else:
            cpu_distance = 0.0

        if memory < min_memory:
            memory_distance = (memory - min_memory) / mem_bandwidth
        elif memory > max_memory:
            memory_distance = (memory - max_memory) / mem_bandwidth
        else:
            memory_distance = 0.0

        # Recalculate reward
        reward = env.calculate_reward(
            action=action,
            cpu_distance=cpu_distance,
            memory_distance=memory_distance,
            response_time=response_time,
        )

        # Recalculate next_obs with corrected values
        cpu_relative = np.clip((cpu - min_cpu) / cpu_bandwidth, 0.0, 1.0)
        memory_relative = np.clip((memory - min_memory) / mem_bandwidth, 0.0, 1.0)
        next_obs = np.array(
            [
                action / 99.0,
                cpu_relative,
                memory_relative,
                np.clip(cpu_distance, -2.0, 2.0),
                np.clip(memory_distance, -2.0, 2.0),
                np.clip(response_time / 100.0, 0.0, 3.0),
            ],
            dtype=np.float32,
        )

        model.replay_buffer.add(
            obs=obs,
            next_obs=next_obs,
            action=np.array([action]),
            reward=np.array([reward]),
            done=np.array([terminated or truncated]),
            infos=[{}],
        )


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

    env = OfflineEnv(
        weight_response_time=float(os.getenv("WEIGHT_RESPONSE_TIME", "1.0")),
        weight_cost=float(os.getenv("WEIGHT_COST", "1.0")),
    )

    # CPU/memory bounds for stale data recalculation
    min_cpu = float(os.getenv("MIN_CPU", "20.0"))
    max_cpu = float(os.getenv("MAX_CPU", "80.0"))
    min_memory = float(os.getenv("MIN_MEMORY", "20.0"))
    max_memory = float(os.getenv("MAX_MEMORY", "60.0"))

    note = os.getenv("NOTE", "offline")
    model_dir = Path(f"model/{now}_{note}")
    model_dir.mkdir(parents=True, exist_ok=True)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        policy_kwargs={"net_arch": [256, 256, 128]},
        learning_rate=1e-4,
        gamma=0.99,
        buffer_size=len(df) + 1000,
        learning_starts=0,
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=100,
        exploration_fraction=0.0,
        exploration_initial_eps=0.0,
        exploration_final_eps=0.0,
        verbose=1,
        tensorboard_log=log_dir,
        seed=42,
        device="auto",
    )

    logger.info("Populating replay buffer with stale data correction...")
    populate_replay_buffer(model, df, env, min_cpu, max_cpu, min_memory, max_memory)
    logger.info(f"Replay buffer size: {model.replay_buffer.size()}")

    train_steps = int(os.getenv("TRAIN_STEPS", len(df) * 10))
    logger.info(f"Training for {train_steps} gradient steps")

    checkpoint_callback = CheckpointCallback(
        save_freq=train_steps // 5,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="dqn_offline",
        save_replay_buffer=False,
        verbose=1,
    )

    model.learn(
        total_timesteps=train_steps,
        callback=checkpoint_callback,
        progress_bar=True,
        tb_log_name="DQN_offline",
    )

    final_path = model_dir / "final" / "model"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(final_path)
    logger.info(f"Model saved to {final_path}")

    env.close()
