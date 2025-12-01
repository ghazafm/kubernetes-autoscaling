import argparse
import ast
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from gymnasium import Env, spaces
from rl import DQN, Q

try:
    # import stable-baselines3 DQN but keep name separate to avoid collision
    from stable_baselines3 import DQN as SB3_DQN
except Exception:  # pragma: no cover - optional dependency
    SB3_DQN = None

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
    # NEW: 13D state components
    replica_pct = [0.0, 2.0, 4.1, 6.1, 4.1]  # percentage of range
    cpu_delta = [0.0, 20.0, 20.0, 20.0, -30.0]
    mem_delta = [0.0, 15.0, 25.0, 20.0, -35.0]
    rt_delta = [0.0, 5.0, 15.0, 30.0, -35.0]
    time_in_state = [0.0, 0.1, 0.2, 0.3, 0.0]
    scaling_direction = [0.5, 1.0, 1.0, 1.0, 0.0]  # same, up, up, up, down
    rps_per_pod = [5.0, 7.5, 6.0, 4.5, 5.5]  # requests per second per pod
    rps_delta = [0.0, 2.5, -1.5, -1.5, 1.0]  # change in RPS per pod
    error_rate = [0.0, 0.5, 1.2, 2.5, 1.0]  # error rate percentage

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
                "current_replica_pct": replica_pct[j],
                "last_action": replica[j],
                "cpu_delta": cpu_delta[j],
                "memory_delta": mem_delta[j],
                "rt_delta": rt_delta[j],
                "time_in_state": time_in_state[j],
                "scaling_direction": scaling_direction[j],
                "rps_per_pod": rps_per_pod[j],
                "rps_delta": rps_delta[j],
                "error_rate": error_rate[j],
            }
        )

    data = {
        "cpu_usage": cpu,
        "memory_usage": mem,
        "response_time": resp,
        "current_replica_pct": replica_pct,
        "replica": replica,
        "cpu_delta": cpu_delta,
        "memory_delta": mem_delta,
        "rt_delta": rt_delta,
        "time_in_state": time_in_state,
        "scaling_direction": scaling_direction,
        "rps_per_pod": rps_per_pod,
        "rps_delta": rps_delta,
        "error_rate": error_rate,
        "action": actions,
        "reward": rewards,
        "next_state": next_states,
        # optional terminated/done flag
        "done": [False, False, False, False, True],
    }
    return pd.DataFrame(data)


class OfflineDatasetEnv(Env):
    """A minimal Gymnasium environment that plays back transitions from a
    pandas DataFrame. This is intended for offline training with Stable-Baselines3
    where the environment returns prerecorded observations, rewards and done
    flags from a CSV. Agent actions do not influence transitions (offline data).

    Observations are converted to a fixed 13-d numpy array in the order used
    throughout this repo.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        episode_length: Optional[int] = None,
        shuffle: bool = False,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.episode_length = episode_length or len(self.df)
        self.shuffle = shuffle
        # observation shape: 13 features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(3)
        self._idx = 0

    @staticmethod
    def _row_to_obs(row: pd.Series) -> np.ndarray:
        # consistent ordering of features
        keys = [
            "cpu_usage",
            "memory_usage",
            "response_time",
            "current_replica_pct",
            "last_action",
            "cpu_delta",
            "memory_delta",
            "rt_delta",
            "time_in_state",
            "scaling_direction",
            "rps_per_pod",
            "rps_delta",
            "error_rate",
            "error_rate",
            "cpu_dist",
            "memory_dist",
            "cpu_in_band",
            "memory_in_band",
        ]
        vals = []
        for k in keys:
            try:
                v = row.get(k, 0.0)
                # Some CSVs store numbers as strings
                vals.append(float(v) if v is not None else 0.0)
            except Exception:
                vals.append(0.0)
        return np.array(vals, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
        if self.shuffle:
            # choose a random start index so episodes sample different slices
            max_start = max(1, len(self.df) - self.episode_length)
            self._idx = int(np.random.randint(0, max_start))
        else:
            self._idx = 0
        row = self.df.iloc[self._idx]
        obs = self._row_to_obs(row)
        info = {}
        return obs, info

    def step(self, action):
        # return current row's reward and next observation
        row = self.df.iloc[self._idx]
        try:
            reward = float(row.get("reward", 0.0))
        except Exception:
            reward = 0.0
        done_flag = row.get("done", False)
        if isinstance(done_flag, str):
            terminated = done_flag.strip().lower() in ("true", "1", "t", "yes")
        else:
            terminated = bool(done_flag)

        # advance index for next observation
        self._idx += 1
        truncated = False
        if self._idx >= len(self.df) or (
            self.episode_length and self._idx >= self.episode_length
        ):
            terminated = True

        if self._idx < len(self.df):
            next_row = self.df.iloc[self._idx]
            obs = self._row_to_obs(next_row)
        else:
            # return last observation if out of range
            obs = np.zeros((13,), dtype=np.float32)

        info = {}
        return obs, float(reward), bool(terminated), bool(truncated), info


def train(  # noqa: PLR0912, PLR0915
    data_path: str | None,
    episodes: int = 100,
    save_dir: str | Path = "Model",
    logger: logging.Logger | None = None,
    log_every: int = 10,
    log_every_step: int = 100,
    save_every: int = 10,
    max_steps: Optional[int] = None,
    shuffle: bool = True,
    episode_length: Optional[int] = None,
    recompute_reward: bool = False,
) -> None:
    start_time = time.time()
    logger = logger or make_logger()

    # data is state, action, reward, next_state [tuples]
    df = load_or_make_df(data_path, logger)

    # If episode_length not specified, use full dataset per episode
    if episode_length is None:
        episode_length = len(df)
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
    elif choose_algorithm in ("SB3", "SB3_DQN", "STABLE"):
        # Use Stable-Baselines3 DQN on an offline playback environment
        if SB3_DQN is None:
            raise RuntimeError(
                "stable-baselines3 is not installed; install it to use SB3 algorithm"
            )
        logger.info("Using Stable-Baselines3 DQN (offline playback)")
        env = OfflineDatasetEnv(df, episode_length=episode_length, shuffle=shuffle)
        total_timesteps = int(episodes * episode_length)
        # ensure save directories exist for SB3 artifacts
        sd = Path(save_dir)
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "checkpoint").mkdir(parents=True, exist_ok=True)
        final_dir_sb3 = sd / "final"
        final_dir_sb3.mkdir(parents=True, exist_ok=True)

        tb_log = str(sd / "tb_logs")
        model = SB3_DQN("MlpPolicy", env, verbose=1, tensorboard_log=tb_log)
        logger.info(f"Starting SB3.learn for {total_timesteps} timesteps")
        model.learn(total_timesteps=total_timesteps)
        model.save(str(final_dir_sb3 / "sb3_dqn_final"))
        logger.info("SB3 training complete; saved final model")
        return
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

        # Shuffle data each episode for better learning
        if shuffle:
            df_shuffled = df.sample(frac=1.0, random_state=ep).reset_index(drop=True)
        else:
            df_shuffled = df

        # Limit steps per episode if specified
        steps_this_episode = min(episode_length, len(df_shuffled))
        if max_steps is not None:
            steps_this_episode = min(steps_this_episode, max_steps)

        # iterate rows but optionally limit per-episode steps
        for i in range(steps_this_episode):
            data = df_shuffled.iloc[i]

            # Normalize observation fields (CSV may store them as strings)
            try:
                cpu_val = float(data["cpu_usage"])
            except Exception:
                cpu_val = 0.0
            try:
                mem_val = float(data["memory_usage"])
            except Exception:
                mem_val = 0.0
            try:
                resp_val = float(data["response_time"])
            except Exception:
                resp_val = 0.0
            try:
                last_act_val = int(data["replica"])
            except Exception:
                try:
                    last_act_val = int(float(data["replica"]))
                except Exception:
                    last_act_val = 0

            # NEW: Extract 13D state components from CSV
            try:
                current_replica_pct_val = float(data.get("current_replica_pct", 0.0))
            except Exception:
                current_replica_pct_val = 0.0

            try:
                cpu_delta_val = float(data.get("cpu_delta", 0.0))
            except Exception:
                cpu_delta_val = 0.0

            try:
                mem_delta_val = float(data.get("memory_delta", 0.0))
            except Exception:
                mem_delta_val = 0.0

            try:
                rt_delta_val = float(data.get("rt_delta", 0.0))
            except Exception:
                rt_delta_val = 0.0

            try:
                time_in_state_val = float(data.get("time_in_state", 0.0))
            except Exception:
                time_in_state_val = 0.0

            try:
                scaling_direction_val = float(data.get("scaling_direction", 0.5))
            except Exception:
                scaling_direction_val = 0.5

            try:
                rps_per_pod_val = float(data.get("rps_per_pod", 0.0))
            except Exception:
                rps_per_pod_val = 0.0

            try:
                rps_delta_val = float(data.get("rps_delta", 0.0))
            except Exception:
                rps_delta_val = 0.0

            try:
                error_rate_val = float(data.get("error_rate", 0.0))
            except Exception:
                error_rate_val = 0.0

            obs = {
                "cpu_usage": cpu_val,
                "memory_usage": mem_val,
                "response_time": resp_val,
                "current_replica_pct": current_replica_pct_val,
                "last_action": last_act_val,
                "cpu_delta": cpu_delta_val,
                "memory_delta": mem_delta_val,
                "rt_delta": rt_delta_val,
                "time_in_state": time_in_state_val,
                "scaling_direction": scaling_direction_val,
                "rps_per_pod": rps_per_pod_val,
                "rps_delta": rps_delta_val,
                "error_rate": error_rate_val,
            }
            action = data.get("action", 0)
            # Extract next state / reward / done with fallbacks for CSV shapes
            next_obs = data.get("next_state", None)
            reward = data.get("reward", 0.0)
            done = data.get("done", False)

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

            # Optionally recompute reward using canonical environment logic so
            # offline training matches online semantics exactly. This will use
            # observation fields and next_state to infer whether a scaling was
            # actually applied (used to apply stability penalty).
            if recompute_reward or os.getenv("RECOMPUTE_REWARD", "False").lower() in (
                "1",
                "true",
                "t",
                "yes",
            ):
                try:
                    # Extract fields required for reward computation
                    cpu_val_r = float(obs.get("cpu_usage", 0.0))
                    mem_val_r = float(obs.get("memory_usage", 0.0))
                    # response_time stored as percent in CSV; convert back to ms
                    max_rt = float(os.getenv("MAX_RESPONSE_TIME", "100.0"))
                    resp_pct = float(obs.get("response_time", 0.0))
                    resp_val_r = resp_pct / 100.0 * max_rt

                    # last_action stored in `replica` column as 0-99
                    last_action_pct = int(obs.get("last_action", obs.get("replica", 0)))

                    # next state's last_action is stored under nxt['last_action']
                    next_last_action = nxt.get("last_action")
                    try:
                        next_action_int = int(next_last_action)
                    except Exception:
                        try:
                            next_action_int = int(float(next_last_action))
                        except Exception:
                            next_action_int = 0

                    # error rate is present in observation or next state
                    error_rate_val = float(obs.get("error_rate", 0.0))

                    # Load weights and config from env or defaults to match generator
                    response_time_weight = _getenv_float("RESPONSE_TIME_WEIGHT", 1.5)
                    error_rate_weight = _getenv_float("ERROR_RATE_WEIGHT", 1.0)
                    cpu_memory_weight = _getenv_float("CPU_MEMORY_WEIGHT", 0.5)
                    cost_weight = _getenv_float("COST_WEIGHT", 0.3)

                    # replicate the environment reward calculation
                    def _cpu_mem_penalty(
                        value: float, low: float, high: float, min_tol_pct: float = 0.01
                    ) -> float:
                        if low <= value <= high:
                            return 0.0
                        distance = low - value if value < low else value - high
                        bandwidth = max(high - low, 1e-6)
                        min_tol = max(min_tol_pct * bandwidth, 1e-6)
                        normalized = distance / (bandwidth + min_tol)
                        return float(min(1.0, normalized * normalized))

                    min_cpu = _getenv_float("MIN_CPU", 20.0)
                    max_cpu = _getenv_float("MAX_CPU", 90.0)
                    min_memory = _getenv_float("MIN_MEMORY", 20.0)
                    max_memory = _getenv_float("MAX_MEMORY", 90.0)

                    cpu_pen_r = _cpu_mem_penalty(cpu_val_r, min_cpu, max_cpu)
                    mem_pen_r = _cpu_mem_penalty(mem_val_r, min_memory, max_memory)

                    response_time_percentage = min(
                        (resp_val_r / max_rt) * 100.0, 1000.0
                    )
                    RESPONSE_TIME_HIGH_THRESHOLD = 80.0
                    RESPONSE_TIME_VIOLATION_THRESHOLD = 100.0

                    if response_time_percentage <= RESPONSE_TIME_HIGH_THRESHOLD:
                        resp_pen_r = 0.0
                    elif response_time_percentage <= RESPONSE_TIME_VIOLATION_THRESHOLD:
                        resp_pen_r = (
                            response_time_percentage - RESPONSE_TIME_HIGH_THRESHOLD
                        ) / (
                            RESPONSE_TIME_VIOLATION_THRESHOLD
                            - RESPONSE_TIME_HIGH_THRESHOLD
                        )
                    else:
                        over = (
                            response_time_percentage - RESPONSE_TIME_VIOLATION_THRESHOLD
                        ) / RESPONSE_TIME_VIOLATION_THRESHOLD
                        resp_pen_r = 1.0 + over

                    resp_pen_r = max(0.0, min(resp_pen_r, 2.0))

                    error_pen_r = min(max(error_rate_val, 0.0) / 100.0, 1.0)

                    weighted_resp_pen = response_time_weight * resp_pen_r
                    weighted_error_pen = error_rate_weight * error_pen_r
                    weighted_cpu_mem_pen = cpu_memory_weight * (cpu_pen_r + mem_pen_r)
                    weighted_cost_pen = cost_weight * (last_action_pct / 100.0)

                    total_penalty = (
                        weighted_resp_pen
                        + weighted_error_pen
                        + weighted_cpu_mem_pen
                        + weighted_cost_pen
                    )
                    max_possible_penalty = (
                        response_time_weight
                        + error_rate_weight
                        + cpu_memory_weight * 2.0
                        + cost_weight
                    )
                    normalized_penalty = min(total_penalty / max_possible_penalty, 1.0)
                    recomputed_reward = 1.0 - 2.0 * normalized_penalty

                    try:
                        stability_penalty = (
                            _getenv_float("STABILITY_PENALTY", 0.05) or 0.0
                        )
                    except Exception:
                        stability_penalty = 0.0
                    if next_action_int - last_action_pct != 0:
                        recomputed_reward -= stability_penalty

                    try:
                        blocked_pen = float(data.get("blocked_penalty", 0.0))
                    except Exception:
                        blocked_pen = 0.0

                    reward = float(max(min(recomputed_reward, 1.0), -1.0)) - blocked_pen
                except Exception:
                    # Fall back to provided reward on any failure
                    try:
                        reward = float(reward)
                    except Exception:
                        reward = 0.0

            agent.update(obs, action, reward, nxt)

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
    p.add_argument(
        "--episode-length",
        type=int,
        default=None,
        help="Number of steps per episode (default: use full dataset)",
    )
    p.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable data shuffling between episodes",
    )
    p.add_argument(
        "--recompute-reward",
        action="store_true",
        help="Recompute reward from observations using canonical environment logic",
    )
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    p.add_argument(
        "--algorithm",
        type=str,
        default=os.getenv("ALGORITHM", "Q"),
        help="Algorithm to use: Q, DQN, SB3",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log = make_logger(logging.DEBUG if args.debug else logging.INFO)
    # allow CLI to override algorithm selection used in train()
    if getattr(args, "algorithm", None):
        os.environ["ALGORITHM"] = args.algorithm
    train(
        data_path=args.data_path,
        episodes=args.episodes,
        save_dir=args.save_dir,
        logger=log,
        log_every=args.log_every,
        save_every=args.save_every,
        episode_length=args.episode_length,
        shuffle=not args.no_shuffle,
        recompute_reward=args.recompute_reward,
    )
