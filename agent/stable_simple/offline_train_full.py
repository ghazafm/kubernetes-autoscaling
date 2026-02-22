import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from environment import calculate_reward
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import DQN
from stable_baselines3.common.type_aliases import TrainFrequencyUnit
from utils import setup_logger

load_dotenv()

OBS_LOW = np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -3.0], dtype=np.float32)
OBS_HIGH = np.array([1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0], dtype=np.float32)


class OfflineDatasetEnv(Env):
    def __init__(self):
        self.action_space = Discrete(100)
        self.observation_space = Box(low=OBS_LOW, high=OBS_HIGH, dtype=np.float32)
        self._zero_obs = np.zeros(7, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._zero_obs.copy(), {}

    def step(self, action):
        return self._zero_obs.copy(), 0.0, True, False, {}


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def to_float(value, default: float = 0.0) -> float:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return default
    return float(number)


def clip_obs(obs: np.ndarray) -> np.ndarray:
    return np.clip(obs, OBS_LOW, OBS_HIGH).astype(np.float32)


def row_to_obs(row: pd.Series, prefix: str) -> np.ndarray:
    obs = np.array(
        [
            to_float(row.get(f"{prefix}action", 0.0)),
            to_float(row.get(f"{prefix}cpu", 0.0)),
            to_float(row.get(f"{prefix}memory", 0.0)),
            to_float(row.get(f"{prefix}response_time", 0.0)),
            to_float(row.get(f"{prefix}cpu_delta", 0.0)),
            to_float(row.get(f"{prefix}memory_delta", 0.0)),
            to_float(row.get(f"{prefix}rt_delta", 0.0)),
        ],
        dtype=np.float32,
    )
    return clip_obs(obs)


def add_transition_to_buffer(model: DQN, row: pd.Series):
    obs = row_to_obs(row, prefix="obs_")
    next_obs = row_to_obs(row, prefix="next_obs_")

    action = int(np.clip(to_float(row.get("action", 0.0)), 0, 99))

    if "reward" in row and not pd.isna(row["reward"]):
        reward = float(row["reward"])
    else:
        reward, _ = calculate_reward(
            action=action,
            response_time=to_float(row.get("response_time", 0.0)),
        )

    terminated = to_bool(row.get("terminated", False))
    truncated = to_bool(row.get("truncated", False))
    done = terminated or truncated

    model.replay_buffer.add(
        obs=obs.reshape(1, -1),
        next_obs=next_obs.reshape(1, -1),
        action=np.array([[action]], dtype=np.int64),
        reward=np.array([reward], dtype=np.float32),
        done=np.array([done], dtype=np.float32),
        infos=[{"TimeLimit.truncated": bool(truncated)}],
    )


if __name__ == "__main__":
    now = datetime.now().strftime("%Y-%m-%d-%H-%M")
    logger, log_dir = setup_logger(
        "offline_train", log_level=os.getenv("LOG_LEVEL", "INFO"), log_to_file=True
    )

    csv_paths_env = os.getenv("CSV_PATHS", "")
    if csv_paths_env:
        csv_paths = [path.strip() for path in csv_paths_env.split(",") if path.strip()]
    else:
        csv_paths = [str(path) for path in Path("data").glob("*.csv")]

    if not csv_paths:
        logger.error("No CSV files found. Set CSV_PATHS or place CSVs in 'data'.")
        sys.exit(1)

    df = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)

    required_columns = [
        "obs_action",
        "obs_cpu",
        "obs_memory",
        "obs_response_time",
        "obs_cpu_delta",
        "obs_memory_delta",
        "obs_rt_delta",
        "next_obs_action",
        "next_obs_cpu",
        "next_obs_memory",
        "next_obs_response_time",
        "next_obs_cpu_delta",
        "next_obs_memory_delta",
        "next_obs_rt_delta",
        "action",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        sys.exit(1)
    if "reward" not in df.columns and "response_time" not in df.columns:
        logger.error("Need 'response_time' column to recompute reward")
        sys.exit(1)

    dataset_size = len(df)
    num_epochs = max(1, int(os.getenv("EPOCHS", "1")))
    default_total_timesteps = dataset_size * num_epochs
    total_timesteps = max(
        1,
        int(
            os.getenv(
                "TOTAL_TIMESTEPS",
                os.getenv("GRADIENT_STEPS", str(default_total_timesteps)),
            )
        ),
    )

    note = os.getenv("NOTE", "offline")
    model_dir = Path(f"model/{now}_{note}")
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = OfflineDatasetEnv()
    model = DQN(policy="MlpPolicy", env=env, seed=1, tensorboard_log=log_dir)

    if model.train_freq.unit != TrainFrequencyUnit.STEP:
        logger.error(
            "Offline loop currently supports step-based train_freq only. "
            f"Found train_freq={model.train_freq}."
        )
        sys.exit(1)

    total_timesteps, callback = model._setup_learn(
        total_timesteps=total_timesteps,
        callback=None,
        reset_num_timesteps=True,
        tb_log_name="offline_dqn",
        progress_bar=False,
    )
    callback.on_training_start(locals(), globals())

    if dataset_size > int(model.buffer_size):
        logger.warning(
            "Dataset size exceeds replay capacity. Old transitions will be overwritten:"
            f" dataset={dataset_size:,}, buffer={int(model.buffer_size):,}"
        )

    logger.info("Prefilling replay buffer from CSV transitions...")
    inserted = 0
    for _, row in df.iterrows():
        add_transition_to_buffer(model, row)
        inserted += 1
    logger.info(f"Replay buffer filled with {inserted:,} transitions")

    if inserted < int(model.batch_size):
        logger.error(
            "Not enough transitions for one batch: "
            f"inserted={inserted}, batch_size={int(model.batch_size)}"
        )
        sys.exit(1)

    checkpoint_freq = max(int(model.target_update_interval) * 2, 50000)
    train_freq = int(model.train_freq.frequency)
    grad_steps = int(model.gradient_steps)
    if grad_steps < 0:
        grad_steps = train_freq

    update_calls = 0
    for step in range(1, total_timesteps + 1):
        model.num_timesteps += 1

        callback.update_locals(locals())
        if not callback.on_step():
            break

        model._update_current_progress_remaining(
            model.num_timesteps, model._total_timesteps
        )
        model._on_step()

        if step % train_freq == 0 and model.num_timesteps > model.learning_starts:  # noqa: SIM102
            if grad_steps > 0:
                model.train(gradient_steps=grad_steps, batch_size=int(model.batch_size))
                update_calls += 1

        if step % checkpoint_freq == 0:
            checkpoint_model_path = checkpoint_dir / f"dqn_autoscaler_{step}_steps"
            checkpoint_buffer_path = (
                checkpoint_dir / f"dqn_autoscaler_replay_buffer_{step}_steps.pkl"
            )
            model.save(checkpoint_model_path)
            model.save_replay_buffer(checkpoint_buffer_path)
            logger.info(f"Checkpoint saved at step {step:,}: {checkpoint_model_path}")
            model.logger.dump(step=model.num_timesteps)

    model.logger.dump(step=model.num_timesteps)
    callback.on_training_end()

    final_path = model_dir / "final" / "model"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(final_path)
    logger.info(f"Model saved to {final_path}")

    buffer_path = model_dir / "final" / "replay_buffer.pkl"
    model.save_replay_buffer(buffer_path)
    logger.info(f"Replay buffer saved to {buffer_path}")

    env.close()
    logger.info("Offline training completed successfully")
    logger.info(
        f"Transitions={inserted:,}, timesteps={model.num_timesteps:,}, "
        f"train_calls={update_calls:,}, replay_epochs={num_epochs}"
    )
