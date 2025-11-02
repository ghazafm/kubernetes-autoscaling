import argparse
import ast
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from dotenv import load_dotenv
from rl import DQN, Q

load_dotenv()


def _getenv_float(name: str, default: Optional[float] = None) -> Optional[float]:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        msg = f"Environment variable {name} must be a float: got {v!r}"
        raise ValueError(msg) from None


def _getenv_int(name: str, default: Optional[int] = None) -> Optional[int]:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        msg = f"Environment variable {name} must be an int: got {v!r}"
        raise ValueError(msg) from None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("train_with_csv")
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    logger.setLevel(level)
    return logger


def load_or_make_df(path: str | None, logger: logging.Logger) -> pd.DataFrame:
    if path:
        df = pd.read_csv(path)
        logger.info(f"Loaded data from {path} ({len(df)} rows)")
        return df

    # Small synthetic dataset if no path provided (for smoke tests)
    logger.warning("No data_path provided, creating small synthetic dataset")
    # Create a small synthetic dataset compatible with the training loop
    cpu = [20.0, 40.0, 60.0, 80.0, 50.0]
    mem = [30.0, 45.0, 70.0, 90.0, 55.0]
    resp = [10.0, 15.0, 30.0, 60.0, 25.0]
    replica = [1, 2, 3, 4, 3]

    # actions and rewards (toy values)
    actions = [1, 1, 0, -1, 0]
    rewards = [0.1, 0.2, -0.1, -0.5, 0.0]

    # next_state is represented as a dict per-row to match expected format
    next_states: list[Dict[str, Any]] = []
    for i in range(len(cpu)):
        j = min(i + 1, len(cpu) - 1)
        next_states.append(
            {
                "cpu_usage": cpu[j],
                "memory_usage": mem[j],
                "response_time": resp[j],
                "last_action": replica[j],
            }
        )

    data = {
        "cpu_usage": cpu,
        "memory_usage": mem,
        "response_time": resp,
        "replica": replica,
        "action": actions,
        "reward": rewards,
        "next_state": next_states,
        # optional terminated/done flag
        "done": [False, False, False, False, True],
    }
    return pd.DataFrame(data)


def train(
    data_path: str | None,
    episodes: int = 100,
    save_dir: str | Path = "Model",
    logger: logging.Logger | None = None,
    log_every: int = 10,
    log_every_step: int = 100,
    save_every: int = 10,
    max_steps: Optional[int] = None,
) -> None:
    start_time = time.time()
    logger = logger or make_logger()

    # data is state, action, reward, next_state [tuples]
    df = load_or_make_df(data_path, logger)
    choose_algorithm = os.getenv("ALGORITHM", "Q").upper()
    logger.info(f"Using algorithm: {choose_algorithm}")
    if choose_algorithm == "Q":
        # Use sensible defaults when env vars are not set to avoid passing None
        agent = Q(
            learning_rate=_getenv_float("LEARNING_RATE", 0.1),
            discount_factor=_getenv_float("DISCOUNT_FACTOR", 0.95),
            epsilon_start=_getenv_float("EPSILON_START", 0.1),
            epsilon_decay=_getenv_float("EPSILON_DECAY", 0.0),
            epsilon_min=_getenv_float("EPSILON_MIN", 0.01),
            created_at=start_time,
            logger=logger,
        )
    elif choose_algorithm == "DQN":
        agent = DQN(
            learning_rate=_getenv_float("LEARNING_RATE", 0.001),
            discount_factor=_getenv_float("DISCOUNT_FACTOR", 0.99),
            epsilon_start=_getenv_float("EPSILON_START", 1.0),
            epsilon_decay=_getenv_float("EPSILON_DECAY", 0.995),
            epsilon_min=_getenv_float("EPSILON_MIN", 0.01),
            device=os.getenv("DEVICE", None),
            buffer_size=_getenv_int("BUFFER_SIZE", 10000),
            batch_size=_getenv_int("BATCH_SIZE", 64),
            target_update_freq=_getenv_int("TARGET_UPDATE_FREQ", 10),
            grad_clip_norm=_getenv_float("GRAD_CLIP_NORM", 1.0),
            created_at=start_time,
            logger=logger,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {choose_algorithm}")

    # Ensure save_dir is a Path and create directories
    save_dir = Path(save_dir)
    checkpoint_dir = save_dir / "checkpoint"
    final_dir = save_dir / "final"
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training loop")
    for ep in range(1, episodes + 1):
        total_reward = 0.0
        done = False
        step = 0

        # iterate rows but optionally limit per-episode steps
        for i, data in enumerate(df.itertuples()):
            if max_steps is not None and i >= max_steps:
                break
            # Normalize observation fields (CSV may store them as strings)
            try:
                cpu_val = float(data.cpu_usage)
            except Exception:
                cpu_val = 0.0
            try:
                mem_val = float(data.memory_usage)
            except Exception:
                mem_val = 0.0
            try:
                resp_val = float(data.response_time)
            except Exception:
                resp_val = 0.0
            try:
                last_act_val = int(data.replica)
            except Exception:
                try:
                    last_act_val = int(float(data.replica))
                except Exception:
                    last_act_val = 0

            obs = {
                "cpu_usage": cpu_val,
                "memory_usage": mem_val,
                "response_time": resp_val,
                "last_action": last_act_val,
            }
            action = getattr(data, "action", 0)
            # Extract next state / reward / done with fallbacks for CSV shapes
            next_obs = getattr(data, "next_state", None)
            reward = getattr(data, "reward", 0.0)
            done = getattr(data, "done", False)

            # Normalize types coming from CSV (strings) so agents receive proper types
            try:
                action = int(action)
            except Exception:
                try:
                    action = int(float(action))
                except Exception:
                    action = 0

            try:
                reward = float(reward)
            except Exception:
                reward = 0.0

            # Normalize done/terminated to a boolean. CSV may contain 'True'/'False' strings.
            if isinstance(done, str):
                done_bool = done.strip().lower() in ("true", "1", "t", "yes")
            else:
                done_bool = bool(done)

            # next_obs may be a dict, a pandas Series, or a string (if read from CSV)
            if isinstance(next_obs, str):
                try:
                    # try to evaluate simple dict-like strings
                    parsed = ast.literal_eval(next_obs)
                    nxt = dict(parsed)
                except Exception:
                    nxt = {}
            elif hasattr(next_obs, "to_dict"):
                nxt = dict(next_obs.to_dict())
            elif isinstance(next_obs, dict):
                nxt = dict(next_obs)
            else:
                nxt = {}
            nxt["terminated"] = done_bool

            agent.update_q_table(obs, action, reward, nxt)

            obs = next_obs
            total_reward += float(reward)

            if step % log_every_step == 0 or ep == 1:
                logger.debug(
                    f"Episode {ep} Step {step}: action={action}, reward={reward:.4f}, "
                    f"total_reward={total_reward:.4f}"
                )
            step += 1

        agent.add_episode_count(1)

        if ep % log_every == 0 or ep == 1:
            avg_reward = total_reward / step if step else 0.0
            logger.info(
                f"Episode {ep}/{episodes}: total_reward={total_reward:.4f}, "
                f"avg_reward={avg_reward:.6f}, steps={step}"
            )
            # Save checkpoint
            if ep % save_every == 0 or ep == 1:
                agent.save_model(
                    checkpoint_dir / f"checkpoint_{ep}.pth", episode_count=ep
                )

    logger.info("Training complete")

    agent.save_model(final_dir / "final.pth", episode_count=episodes)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Train DQN with captured CSV using fast env")
    )
    p.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to captured CSV",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to train",
    )
    p.add_argument(
        "--save-dir",
        type=str,
        default="models/dqn_fast",
        help="Model save path",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="How often (in episodes) to log training progress",
    )
    p.add_argument(
        "--log-every-step",
        type=int,
        default=100,
        help="How often (in steps) to log training progress",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="How often (in episodes) to save checkpoints",
    )
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log = make_logger(logging.DEBUG if args.debug else logging.INFO)
    train(
        data_path=args.data_path,
        episodes=args.episodes,
        save_dir=args.save_dir,
        logger=log,
        log_every=args.log_every,
        save_every=args.save_every,
    )
