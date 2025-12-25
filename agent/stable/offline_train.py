import os
import sys
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
    def __init__(
        self,
        weight_response_time: float = 1.0,
        weight_cost: float = 1.0,
        num_epochs: int = 1,
    ):
        self.action_space = Discrete(100)
        self.observation_space = Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 3.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.weight_response_time = weight_response_time
        self.weight_cost = weight_cost
        self.num_epochs = num_epochs

        self.data = None
        self._index = 0
        self._epoch = 0
        self.observations = np.zeros(4, dtype=np.float32)

    calculate_reward = KubernetesEnv.calculate_reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._index = 0
        self._epoch = 0

        if self.data is not None and len(self.data) > 0:
            row = self.data.iloc[0]
            obs = np.array(
                [
                    float(row["obs_action"]),
                    float(row["cpu"]) / 100.0,
                    float(row["memory"]) / 100.0,
                    float(row["obs_response_time"]),
                ],
                dtype=np.float32,
            )
            self.observations = obs
            return obs, {}

        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        if self.data is None or len(self.data) == 0:
            return np.zeros(4, dtype=np.float32), 0.0, True, False, {}

        if self._index >= len(self.data):
            self._epoch += 1
            self._index = 0

            if self._epoch >= self.num_epochs:
                return np.zeros(4, dtype=np.float32), 0.0, True, False, {}

            row = self.data.iloc[0]
            obs = np.array(
                [
                    float(row["obs_action"]),
                    float(row["cpu"]) / 100.0,
                    float(row["memory"]) / 100.0,
                    float(row["obs_response_time"]),
                ],
                dtype=np.float32,
            )
            self.observations = obs

        row = self.data.iloc[self._index]
        obs = self.observations.copy()
        next_obs = np.array(
            [
                float(row["next_obs_action"]),
                float(row["cpu"]) / 100.0,
                float(row["memory"]) / 100.0,
                float(row["response_time"]) / 100.0,
            ],
            dtype=np.float32,
        )
        action = int(row["action"])

        reward = KubernetesEnv.calculate_reward(
            self,
            action=action,
            response_time=float(row["response_time"]),
        )

        terminated = bool(row.get("terminated", False))
        truncated = bool(row.get("truncated", False))

        info = {
            "cpu": float(row.get("cpu", 0.0)),
            "memory": float(row.get("memory", 0.0)),
            "response_time": float(row.get("response_time", 0.0)),
            "replicas": int(row.get("replicas", 0)),
            "epoch": self._epoch,
            "index": self._index,
        }

        self._index += 1
        self.observations = next_obs

        return next_obs, reward, terminated, truncated, info


def load_csv_data(csv_paths: list[str]) -> pd.DataFrame:
    dfs = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {p}: {e}")  # noqa: T201

    if not dfs:
        raise ValueError("No CSV files could be loaded")

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

    if not csv_paths:
        logger.error(
            "No CSV files found. Please set CSV_PATHS or "
            "place CSV files in 'data' directory"
        )
        sys.exit(1)

    logger.info(f"Loading {len(csv_paths)} CSV files: {csv_paths}")
    df = load_csv_data(csv_paths)
    logger.info(f"Loaded {len(df)} transitions")

    required_columns = [
        "obs_action",
        "obs_response_time",
        "cpu",
        "memory",
        "response_time",
        "next_obs_action",
        "action",
        "replicas",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        logger.info(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    logger.info("Data statistics:")
    logger.info(f"  CPU range: {df['cpu'].min():.2f}% - {df['cpu'].max():.2f}%")
    logger.info(
        f"  Memory range: {df['memory'].min():.2f}% - {df['memory'].max():.2f}%"
    )
    logger.info(
        f"  Response time range: {df['response_time'].min():.2f}% - "
        f"{df['response_time'].max():.2f}%"
    )
    logger.info(f"  Actions: {df['action'].min()} - {df['action'].max()}")
    logger.info(
        f"  Episodes: {df['episode'].nunique() if 'episode' in df.columns else 'unknown'}"  # noqa: E501
    )

    zero_cpu = (df["cpu"] <= 0.0).sum()
    zero_mem = (df["memory"] <= 0.0).sum()
    zero_rt = (df["response_time"] <= 0.0).sum()
    if zero_cpu > 0 or zero_mem > 0 or zero_rt > 0:
        logger.warning(f"Found {zero_cpu} rows with zero/negative CPU")
        logger.warning(f"Found {zero_mem} rows with zero/negative memory")
        logger.warning(f"Found {zero_rt} rows with zero/negative response time")

    num_epochs = int(os.getenv("NUM_EPOCHS", "1"))
    target_update_freq = int(os.getenv("TARGET_UPDATE_FREQ", "100"))
    warmup_steps = int(os.getenv("WARMUP_STEPS", "1000"))

    env = OfflineEnv(
        weight_response_time=float(os.getenv("WEIGHT_RESPONSE_TIME", "1.0")),
        weight_cost=float(os.getenv("WEIGHT_COST", "1.0")),
        num_epochs=num_epochs,
    )
    env.data = df.reset_index(drop=True)

    note = os.getenv("NOTE", "offline")
    model_dir = Path(f"model/{now}_{note}")
    model_dir.mkdir(parents=True, exist_ok=True)

    dataset_size = len(df)
    total_timesteps = dataset_size * num_epochs

    adjusted_learning_starts = min(warmup_steps, dataset_size // 10)

    logger.info("=" * 80)
    logger.info("Training Configuration:")
    logger.info(f"  Dataset size: {dataset_size:,} transitions")
    logger.info(f"  Number of epochs: {num_epochs}")
    logger.info(f"  Total timesteps: {total_timesteps:,}")
    logger.info(f"  Warmup steps: {adjusted_learning_starts:,}")
    logger.info(f"  Target update frequency: {target_update_freq}")
    logger.info("=" * 80)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        policy_kwargs={"net_arch": [256, 256, 128]},
        learning_rate=1e-4,
        gamma=0.99,
        buffer_size=100_000,
        learning_starts=adjusted_learning_starts,
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=target_update_freq,
        exploration_fraction=0.4,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        max_grad_norm=1,
        verbose=1,
        tensorboard_log=log_dir,
        seed=42,
        device="auto",
    )

    logger.info("Starting offline training...")
    logger.info(f"Buffer size: {model.buffer_size}")
    logger.info(f"Learning starts after: {model.learning_starts} steps")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(target_update_freq * 2, 50000),
        save_path=str(model_dir / "checkpoints"),
        name_prefix="dqn_autoscaler",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=1,
    )

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            reset_num_timesteps=False,
            progress_bar=True,
            tb_log_name="DQN",
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    final_path = model_dir / "final" / "model"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(final_path)
    logger.info(f"Model saved to {final_path}")

    buffer_path = model_dir / "final" / "replay_buffer.pkl"
    model.save_replay_buffer(buffer_path)
    logger.info(f"Replay buffer saved to {buffer_path}")

    env.close()
    logger.info("=" * 80)
    logger.info("Offline training completed successfully!")
    logger.info(f"Trained for {num_epochs} epochs over {dataset_size:,} transitions")
    logger.info(f"Total training steps: {total_timesteps:,}")
    logger.info("=" * 80)
