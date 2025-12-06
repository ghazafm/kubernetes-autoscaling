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


def populate_replay_buffer(model: DQN, df: pd.DataFrame, env: OfflineEnv):
    """Add transitions from DataFrame to replay buffer, recalculating rewards."""
    obs_cols = [
        "obs_action",
        "obs_cpu_relative",
        "obs_memory_relative",
        "obs_cpu_distance",
        "obs_memory_distance",
        "obs_response_time",
    ]
    next_obs_cols = [
        "next_obs_action",
        "next_obs_cpu_relative",
        "next_obs_memory_relative",
        "next_obs_cpu_distance",
        "next_obs_memory_distance",
        "next_obs_response_time",
    ]

    for _, row in df.iterrows():
        obs = np.array([row[c] for c in obs_cols], dtype=np.float32)
        next_obs = np.array([row[c] for c in next_obs_cols], dtype=np.float32)
        action = int(row["action"])
        terminated = bool(row["terminated"])
        truncated = bool(row["truncated"])

        reward = env.calculate_reward(
            action=action,
            cpu_distance=float(row["cpu_distance"]),
            memory_distance=float(row["memory_distance"]),
            response_time=float(row["response_time"]),
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

    logger.info("Populating replay buffer...")
    populate_replay_buffer(model, df, env)
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
