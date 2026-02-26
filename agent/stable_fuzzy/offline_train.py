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

METRICS_LABELS = [
    ("cpu_usage", ["low", "medium", "high"]),
    ("memory_usage", ["low", "medium", "high"]),
    ("response_time", ["low", "medium", "high"]),
    ("last_action", ["low", "medium", "high"]),
    ("delta_cpu", ["decreasing", "stable", "increasing"]),
    ("delta_memory", ["decreasing", "stable", "increasing"]),
    ("delta_response_time", ["decreasing", "stable", "increasing"]),
]
OBS_DIM = sum(len(labels) for _, labels in METRICS_LABELS)  # = 21
OBS_LOW = np.zeros(OBS_DIM, dtype=np.float32)
OBS_HIGH = np.ones(OBS_DIM, dtype=np.float32)


def _row_to_fuzzy_obs(row: pd.Series, prefix: str, fuzzy: Fuzzy) -> np.ndarray:
    p = prefix

    raw_action = float(row.get(f"{p}_action", 0.0))
    raw_cpu = float(row.get(f"{p}_cpu", 0.0))
    raw_mem = float(row.get(f"{p}_memory", 0.0))
    raw_rt = float(row.get(f"{p}_response_time", 0.0))

    raw_dcpu = float(row.get(f"{p}_cpu_delta", row.get(f"{p}_delta_cpu", 0.0)))
    raw_dmem = float(row.get(f"{p}_memory_delta", row.get(f"{p}_delta_memory", 0.0)))
    raw_drt = float(row.get(f"{p}_rt_delta", row.get(f"{p}_delta_response_time", 0.0)))

    cpu_for_fuzzy = raw_cpu * 100.0 if raw_cpu <= 1.0 else raw_cpu
    mem_for_fuzzy = raw_mem * 100.0 if raw_mem <= 1.0 else raw_mem
    rt_for_fuzzy = raw_rt * 100.0

    last_action = raw_action * 99.0 if raw_action <= 1.0 else raw_action

    dcpu_for_fuzzy = raw_dcpu * 100.0 if abs(raw_dcpu) <= 1.0 else raw_dcpu
    dmem_for_fuzzy = raw_dmem * 100.0 if abs(raw_dmem) <= 1.0 else raw_dmem
    drt_for_fuzzy = raw_drt * 100.0 if abs(raw_drt) <= 3.0 else raw_drt

    # ── Fuzzify ───────────────────────────────────────────────────────────────
    fuzzy_state = fuzzy.fuzzify(
        {
            "cpu_usage": float(np.clip(cpu_for_fuzzy, 0.0, 100.0)),
            "memory_usage": float(np.clip(mem_for_fuzzy, 0.0, 100.0)),
            "response_time": float(np.clip(rt_for_fuzzy, 0.0, 300.0)),
            "last_action": float(np.clip(last_action, 0.0, 99.0)),
            "delta_cpu": float(np.clip(dcpu_for_fuzzy, -100.0, 100.0)),
            "delta_memory": float(np.clip(dmem_for_fuzzy, -100.0, 100.0)),
            "delta_response_time": float(np.clip(drt_for_fuzzy, -300.0, 300.0)),
        }
    )

    # ── Flatten in fixed order matching environment.py ────────────────────────
    flat = [fuzzy_state[m][label] for m, labels in METRICS_LABELS for label in labels]
    return np.array(flat, dtype=np.float32)


class OfflineDatasetEnv(Env):
    def __init__(self):
        self.action_space = Discrete(100)
        self.observation_space = Box(low=OBS_LOW, high=OBS_HIGH, dtype=np.float32)
        self._zero_obs = np.zeros(OBS_DIM, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._zero_obs.copy(), {}

    def step(self, action):
        return self._zero_obs.copy(), 0.0, True, False, {}


def add_transition_to_buffer(model: DQN, row: pd.Series, fuzzy: Fuzzy) -> None:

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

    logger.info(f"Loading CSVs: {csv_paths}")
    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    total_timesteps = len(df) * int(os.getenv("EPOCHS", "1"))
    logger.info(f"Total transitions: {total_timesteps:,}")
    logger.info(f"CSV columns: {list(df.columns)}")

    model_dir = Path(f"model/{now}_{os.getenv('NOTE', 'offline')}")
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = DQN(
        policy="MlpPolicy", env=OfflineDatasetEnv(), seed=1, tensorboard_log=log_dir
    )

    assert model.observation_space.shape[0] == OBS_DIM, (
        f"Obs dim mismatch: model has {model.observation_space.shape[0]} dims, "
        f"expected {OBS_DIM}. Update METRICS_LABELS or environment.py."
    )

    total_timesteps, callback = model._setup_learn(
        total_timesteps=total_timesteps,
        callback=None,
        reset_num_timesteps=True,
        tb_log_name="offline_dqn",
    )
    callback.on_training_start(locals(), globals())

    # Instantiate Fuzzy ONCE and reuse across all rows
    fuzzy = Fuzzy()

    for _, row in df.iterrows():
        add_transition_to_buffer(model, row, fuzzy)
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
