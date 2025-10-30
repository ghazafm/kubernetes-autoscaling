#!/usr/bin/env python3
"""Simple training script using the fast offline KubernetesEnv and DQN agent.

Usage:
  python agent/train_with_csv.py --data-path data/captured_metrics.csv --episodes 200

The script will load CSV (or use a small synthetic DataFrame if none provided),
create the env and DQN agent, run episodes, and save a model checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path so package imports work when running script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import logging

import pandas as pd

from agent.environment.environment_fast import KubernetesEnv
from agent.rl.dqn import DQN


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
    data = {
        "cpu_usage": [20.0, 40.0, 60.0, 80.0, 50.0],
        "memory_usage": [30.0, 45.0, 70.0, 90.0, 55.0],
        "response_time": [10.0, 15.0, 30.0, 60.0, 25.0],
        "replica": [1, 2, 3, 4, 3],
    }
    df = pd.DataFrame(data)
    return df


def train(
    data_path: str | None,
    episodes: int = 100,
    max_steps: int = 200,
    save_path: str = "models/dqn_fast.pth",
    logger: logging.Logger | None = None,
) -> None:
    logger = logger or make_logger()

    df = load_or_make_df(data_path, logger)

    env = KubernetesEnv(data_frame=df, logger=logger, iteration=max_steps)

    agent = DQN(logger=logger, device="cpu")

    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training loop")
    for ep in range(1, episodes + 1):
        obs = env.reset()
        total_reward = 0.0
        done = False
        step = 0

        while not done and step < max_steps:
            action = agent.get_action(obs)
            next_obs, reward, done, _ = env.step(action)

            # next_obs is the observation returned by env; include terminated
            nxt = dict(next_obs)
            nxt["terminated"] = done

            agent.update_q_table(obs, action, reward, nxt)

            obs = next_obs
            total_reward += reward
            step += 1

        agent.add_episode_count(1)

        if ep % 10 == 0 or ep == 1:
            logger.info(
                f"Episode {ep}/{episodes}: total_reward={total_reward:.4f}, "
                f"steps={step}"
            )
            # Save checkpoint
            agent.save_model(save_path, episode_count=ep)

    logger.info("Training complete")


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
        "--max-steps",
        type=int,
        default=200,
        help="Max steps per episode",
    )
    p.add_argument(
        "--save-path",
        type=str,
        default="models/dqn_fast.pth",
        help="Model save path",
    )
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log = make_logger(logging.DEBUG if args.debug else logging.INFO)
    train(
        data_path=args.data_path,
        episodes=args.episodes,
        max_steps=args.max_steps,
        save_path=args.save_path,
        logger=log,
    )
