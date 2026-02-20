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
    """
    Minimal env used only to provide action/observation spaces to SB3.
    Training data comes from CSV transitions, not from env.step rollouts.
    """

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
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


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


def prefill_replay_buffer(model: DQN, df: pd.DataFrame) -> int:
    inserted = 0
    for _, row in df.iterrows():
        add_transition_to_buffer(model, row)
        inserted += 1
    return inserted


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
        logger.info(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    if "reward" not in df.columns:
        logger.warning(
            "'reward' column not found. Reward will be recomputed from "
            "action and response_time."
        )
        if "response_time" not in df.columns:
            logger.error("Need 'response_time' column to recompute reward")
            sys.exit(1)

    logger.info("Data statistics:")
    if "cpu" in df.columns:
        logger.info(f"  CPU range: {df['cpu'].min():.2f}% - {df['cpu'].max():.2f}%")
    if "memory" in df.columns:
        logger.info(
            f"  Memory range: {df['memory'].min():.2f}% - {df['memory'].max():.2f}%"
        )
    if "response_time" in df.columns:
        logger.info(
            f"  Response time range: {df['response_time'].min():.2f}% - "
            f"{df['response_time'].max():.2f}%"
        )
    logger.info(f"  Actions: {df['action'].min()} - {df['action'].max()}")
    logger.info(
        f"  Episodes: {df['episode'].nunique() if 'episode' in df.columns else 'unknown'}"  # noqa: E501
    )

    num_epochs = max(1, int(os.getenv("EPOCHS", "1")))
    dataset_size = len(df)
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
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = OfflineDatasetEnv()
    model = DQN(policy="MlpPolicy", env=env, seed=42, tensorboard_log=log_dir)

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

    logger.info("=" * 80)
    logger.info("Offline Training Configuration:")
    logger.info(f"  Dataset transitions: {dataset_size:,}")
    logger.info(f"  Replay epochs (update multiplier): {num_epochs}")
    logger.info(f"  Replay buffer capacity (SB3 default): {int(model.buffer_size):,}")
    logger.info(f"  Total timesteps (offline schedule): {total_timesteps:,}")
    logger.info(f"  Batch size (SB3 default): {int(model.batch_size)}")
    logger.info(
        f"  Target update frequency (SB3 default): {int(model.target_update_interval)}"
    )
    logger.info(f"  Learning starts (SB3 default): {int(model.learning_starts)}")
    logger.info(f"  Train frequency (SB3 default): {model.train_freq}")
    logger.info(
        f"  Gradient steps per train call (SB3 default): {int(model.gradient_steps)}"
    )
    logger.info("=" * 80)

    if dataset_size > int(model.buffer_size):
        logger.warning(
            "Dataset size exceeds replay capacity. Old transitions will be overwritten: "
            f"dataset={dataset_size:,}, buffer={int(model.buffer_size):,}"
        )

    logger.info("Prefilling replay buffer from CSV transitions...")
    inserted = prefill_replay_buffer(model=model, df=df)
    logger.info(f"Replay buffer filled with {inserted:,} transitions")

    if inserted < int(model.batch_size):
        logger.error(
            "Not enough transitions for one batch: "
            f"inserted={inserted}, batch_size={int(model.batch_size)}"
        )
        sys.exit(1)

    checkpoint_freq = max(int(model.target_update_interval) * 2, 50000)
    logger.info("Starting offline gradient updates (no env rollout)...")

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

        if (
            step % model.train_freq.frequency == 0
            and model.num_timesteps > model.learning_starts
        ):
            grad_steps = (
                int(model.gradient_steps)
                if int(model.gradient_steps) >= 0
                else int(model.train_freq.frequency)
            )
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

        if step % 1000 == 0 or step == total_timesteps:
            model.logger.dump(step=model.num_timesteps)
            logger.info(
                f"Offline update progress: {step:,}/{total_timesteps:,} | "
                f"train_calls={update_calls:,}"
            )

    callback.on_training_end()

    final_path = model_dir / "final" / "model"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(final_path)
    logger.info(f"Model saved to {final_path}")

    buffer_path = model_dir / "final" / "replay_buffer.pkl"
    model.save_replay_buffer(buffer_path)
    logger.info(f"Replay buffer saved to {buffer_path}")

    env.close()
    logger.info("=" * 80)
    logger.info("Offline training completed successfully")
    logger.info(f"Replay transitions used: {inserted:,}")
    logger.info(f"Timesteps processed: {model.num_timesteps:,}")
    logger.info(f"Train calls executed: {update_calls:,}")
    logger.info("=" * 80)
