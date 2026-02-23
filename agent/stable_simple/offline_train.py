import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from environment import calculate_reward
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import DQN
from utils import setup_logger

load_dotenv()

OBS_LOW = np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -3.0], dtype=np.float32)
OBS_HIGH = np.array([1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0], dtype=np.float32)
OBS_KEYS = [
    "action",
    "cpu",
    "memory",
    "response_time",
    "cpu_delta",
    "memory_delta",
    "rt_delta",
]


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


def add_transition_to_buffer(model: DQN, row: pd.Series):
    obs = np.clip(
        [row.get(f"obs_{k}", 0.0) for k in OBS_KEYS], OBS_LOW, OBS_HIGH
    ).astype(np.float32)
    next_obs = np.clip(
        [row.get(f"next_obs_{k}", 0.0) for k in OBS_KEYS], OBS_LOW, OBS_HIGH
    ).astype(np.float32)

    action = int(
        np.clip(pd.to_numeric(row.get("action", 0), errors="coerce") or 0, 0, 99)
    )

    reward, _ = calculate_reward(
        action=action,
        response_time=float(
            pd.to_numeric(row.get("response_time", 0.0), errors="coerce") or 0.0
        ),
    )

    done = bool(row.get("terminated", False)) or bool(row.get("truncated", False))

    model.replay_buffer.add(
        obs=obs.reshape(1, -1),
        next_obs=next_obs.reshape(1, -1),
        action=np.array([[action]], dtype=np.int64),
        reward=np.array([reward], dtype=np.float32),
        done=np.array([done], dtype=np.float32),
        infos=[{"TimeLimit.truncated": bool(row.get("truncated", False))}],
    )


if __name__ == "__main__":
    now = datetime.now().strftime("%Y-%m-%d-%H-%M")
    logger, log_dir = setup_logger(
        "offline_train", log_level=os.getenv("LOG_LEVEL", "INFO"), log_to_file=True
    )

    csv_paths = [
        p.strip() for p in os.getenv("CSV_PATHS", "").split(",") if p.strip()
    ] or [str(p) for p in Path("data").glob("*.csv")]

    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    total_timesteps = len(df)

    model_dir = Path(f"model/{now}_{os.getenv('NOTE', 'offline')}")
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = DQN(
        policy="MlpPolicy", env=OfflineDatasetEnv(), seed=1, tensorboard_log=log_dir
    )

    total_timesteps, callback = model._setup_learn(
        total_timesteps=total_timesteps,
        callback=None,
        reset_num_timesteps=True,
        tb_log_name="offline_dqn",
    )
    callback.on_training_start(locals(), globals())

    for _, row in df.iterrows():
        add_transition_to_buffer(model, row)
    logger.info(f"Replay buffer filled with {len(df):,} transitions")

    checkpoint_freq = max(int(model.target_update_interval) * 2, 50000)
    train_freq = model.train_freq.frequency
    grad_steps = model.gradient_steps if model.gradient_steps > 0 else train_freq

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
            step % train_freq == 0
            and model.num_timesteps > model.learning_starts
            and grad_steps > 0
        ):
            model.train(gradient_steps=grad_steps, batch_size=int(model.batch_size))

        if step % checkpoint_freq == 0:
            model.save(checkpoint_dir / f"dqn_autoscaler_{step}_steps")
            model.save_replay_buffer(
                checkpoint_dir / f"dqn_autoscaler_replay_buffer_{step}_steps.pkl"
            )
            model.logger.dump(step=model.num_timesteps)

    model.logger.dump(step=model.num_timesteps)
    callback.on_training_end()

    final_path = model_dir / "final" / "model"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(final_path)
    model.save_replay_buffer(model_dir / "final" / "replay_buffer.pkl")
    logger.info("Offline training completed successfully")
    logger.info(f"Model saved to {final_path}")
