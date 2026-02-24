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
from utils import Fuzzy, setup_logger

load_dotenv()

# New fuzzy observation: 4 metrics x 3 membership labels = 12 dims
OBS_LOW = np.zeros(12, dtype=np.float32)
OBS_HIGH = np.ones(12, dtype=np.float32)


def _row_to_fuzzy_obs(row, prefix: str, fuzzy):
    """Convert a CSV row's obs or next_obs fields into the 12-dim fuzzy vector.

    Handles legacy 7-dim CSVs by reconstructing cpu/memory/response_time/last_action
    values and passing them through the project's Fuzzy.fuzzify().
    - If obs_* values look like normalized (<=1.0), scale them to 0..100 for fuzzify.
    - last_action is reconstructed to 0..99 scale.
    """
    # read raw values (may be missing)
    raw_action = float(row.get(f"{prefix}_action", 0.0))
    raw_cpu = float(row.get(f"{prefix}_cpu", 0.0))
    raw_mem = float(row.get(f"{prefix}_memory", 0.0))
    raw_rt = float(row.get(f"{prefix}_response_time", 0.0))

    # Normalize/sanity: if values look like proportions (<=1), scale to percentage
    cpu_for_fuzzy = raw_cpu * 100.0 if raw_cpu <= 1.0 else raw_cpu
    mem_for_fuzzy = raw_mem * 100.0 if raw_mem <= 1.0 else raw_mem
    rt_for_fuzzy = raw_rt * 100.0 if raw_rt <= 1.0 else raw_rt

    # last_action: CSV stores normalized action (0..1) in obs_action for legacy logs
    last_action = raw_action * 99.0 if raw_action <= 1.0 else raw_action

    fuzzy_state = fuzzy.fuzzify(
        {
            "cpu_usage": float(np.clip(cpu_for_fuzzy, 0.0, 100.0)),
            "memory_usage": float(np.clip(mem_for_fuzzy, 0.0, 100.0)),
            "response_time": float(np.clip(rt_for_fuzzy, 0.0, 100.0)),
            "last_action": float(last_action),
        }
    )

    labels = ["low", "medium", "high"]
    metrics = ["cpu_usage", "memory_usage", "response_time", "last_action"]
    flat = [fuzzy_state[m][label] for m in metrics for label in labels]
    return np.array(flat, dtype=np.float32)


class OfflineDatasetEnv(Env):
    def __init__(self):
        self.action_space = Discrete(100)
        self.observation_space = Box(low=OBS_LOW, high=OBS_HIGH, dtype=np.float32)
        self._zero_obs = np.zeros(12, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._zero_obs.copy(), {}

    def step(self, action):
        return self._zero_obs.copy(), 0.0, True, False, {}


def add_transition_to_buffer(model: DQN, row: pd.Series):
    # Build fuzzy observations (12-dim). Support legacy 7-dim CSVs by reconstructing
    # fuzzy features from obs_action, obs_cpu, obs_memory, obs_response_time.
    fuzzy = Fuzzy()
    obs = _row_to_fuzzy_obs(row, "obs", fuzzy)
    next_obs = _row_to_fuzzy_obs(row, "next_obs", fuzzy)

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
