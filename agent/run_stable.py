from __future__ import annotations

import argparse
import ast
import os
import statistics
from pathlib import Path

import numpy as np
from database import InfluxDB
from dotenv import load_dotenv
from environment.environment import KubernetesEnv
from stable_baselines3 import DQN
from utils import log_verbose_details, setup_logger

# canonical ordering used across repo for observations
KEYS = [
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
]


def obs_to_array(obs_raw) -> np.ndarray:
    """Convert a KubernetesEnv observation (dict or tuple) to a 1D numpy array
    matching the KEYS ordering. If obs_raw is already an array, return it.
    """
    # Some envs return (obs, info) from reset
    if isinstance(obs_raw, (tuple, list)):
        obs = obs_raw[0]
    else:
        obs = obs_raw

    if isinstance(obs, np.ndarray):
        return obs.astype(np.float32)
    if not isinstance(obs, dict):
        # fallback: try to convert directly
        try:
            return np.asarray(obs, dtype=np.float32)
        except Exception:
            return np.zeros((len(KEYS),), dtype=np.float32)

    vals = []
    for k in KEYS:
        try:
            v = obs.get(k, 0.0)
            vals.append(float(v) if v is not None else 0.0)
        except Exception:
            vals.append(0.0)
    return np.asarray(vals, dtype=np.float32)


class SB3AgentWrapper:
    """Lightweight adapter to make an SB3 model look like the repo's agent
    for logging helpers. It implements get_state_key(observation) and exposes
    epsilon (set to None) so `log_verbose_details` can be called safely.
    """

    def __init__(self, model):
        self._model = model
        self.epsilon = None

    def get_state_key(self, observation):
        # Return a numpy array (or tuple) compatible with log_verbose_details
        return obs_to_array(observation)


def _env_int(key: str, default=None):
    v = os.getenv(key)
    if v is None:
        return default
    if isinstance(v, str) and v.lower() in ("inf", "infinity"):
        return float("inf")
    try:
        return int(v)
    except ValueError:
        try:
            return int(float(v))
        except Exception:
            return default


def _env_float(key: str, default=None):
    v = os.getenv(key)
    return float(v) if v is not None else default


def _env_bool(key: str, default=False):
    return ast.literal_eval(os.getenv(key, str(default)))


load_dotenv()

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.evaluation import evaluate_policy
except Exception:  # pragma: no cover - optional dependency
    DQN = None
    evaluate_policy = None
    CheckpointCallback = None


def parse_args():
    p = argparse.ArgumentParser(description="Train/evaluate SB3 DQN on KubernetesEnv")
    p.add_argument(
        "--timesteps", type=int, default=10000, help="Total timesteps to train"
    )
    p.add_argument(
        "--save-dir", type=str, default="models/sb3_run", help="Model save directory"
    )
    p.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path to an existing SB3 model to load",
    )
    p.add_argument(
        "--eval-episodes",
        type=int,
        default=0,
        help="If >0, evaluate the loaded model for this many episodes and exit",
    )
    p.add_argument(
        "--tensorboard-log", type=str, default=None, help="Tensorboard log dir"
    )
    p.add_argument(
        "--prometheus-url",
        type=str,
        default=os.getenv("PROMETHEUS_URL", None),
        help=(
            "Prometheus base URL (overrides default). "
            "If not set, uses KubernetesEnv default"
        ),
    )
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args()


def main():
    args = parse_args()

    # Build upstream KubernetesEnv using environment variables (defaults)
    logger = setup_logger(
        "kubernetes_agent", log_level=os.getenv("LOG_LEVEL", "INFO"), log_to_file=True
    )
    if DQN is None:
        logger.error(
            "stable-baselines3 is not installed. "
            "Please install stable-baselines3[extra]."
        )
        return
    Influxdb = InfluxDB(
        logger=logger,
        url=os.getenv("INFLUXDB_URL", "http://localhost:8086"),
        token=os.getenv("INFLUXDB_TOKEN", "my-token"),
        org=os.getenv("INFLUXDB_ORG", "my-org"),
        bucket=os.getenv("INFLUXDB_BUCKET", "my-bucket"),
    )
    try:
        metrics_endpoints_str = os.getenv(
            "METRICS_ENDPOINTS_METHOD", "[['/', 'GET'], ['/docs', 'GET']]"
        )
        metrics_endpoints_method = ast.literal_eval(metrics_endpoints_str)
    except (ValueError, SyntaxError):
        logger.warning("Invalid METRICS_ENDPOINTS_METHOD format, using default")
        metrics_endpoints_method = [["/", "GET"], ["/docs", "GET"]]

    env = KubernetesEnv(
        min_replicas=_env_int("MIN_REPLICAS", 1),
        max_replicas=_env_int("MAX_REPLICAS", 12),
        iteration=_env_int("ITERATION", 10),
        namespace=os.getenv("NAMESPACE", "default"),
        deployment_name=os.getenv("DEPLOYMENT_NAME", "ecom-api"),
        min_cpu=int(os.getenv("MIN_CPU", "10")),
        min_memory=int(os.getenv("MIN_MEMORY", "10")),
        max_cpu=int(os.getenv("MAX_CPU", "90")),
        max_memory=int(os.getenv("MAX_MEMORY", "90")),
        max_response_time=float(os.getenv("MAX_RESPONSE_TIME", "100.0")),
        timeout=int(os.getenv("TIMEOUT", "120")),
        wait_time=int(os.getenv("WAIT_TIME", "1")),
        verbose=True,
        logger=logger,
        influxdb=Influxdb,
        prometheus_url=os.getenv("PROMETHEUS_URL", "http://localhost:1234/prom"),
        metrics_endpoints_method=metrics_endpoints_method,
        metrics_interval=_env_int("METRICS_INTERVAL", 15),
        metrics_quantile=_env_float("METRICS_QUANTILE", 0.90),
        max_scaling_retries=_env_int("MAX_SCALING_RETRIES", 1000),
        response_time_weight=_env_float("RESPONSE_TIME_WEIGHT", 1.0),
        error_rate_weight=_env_float("ERROR_RATE_WEIGHT", 1.0),
        cpu_memory_weight=_env_float("CPU_MEMORY_WEIGHT", 0.5),
        cost_weight=_env_float("COST_WEIGHT", 0.3),
        # Pass safety/tuning parameters into the environment (use env vars when present)
        max_up_step=_env_int("MAX_UP_STEP", 4),
        max_down_step=_env_int("MAX_DOWN_STEP", 1),
        min_down_confirmations=_env_int("MIN_DOWN_CONFIRMATIONS", 2),
        cooldown_up_secs=_env_int("COOLDOWN_UP_SECS", 60),
        cooldown_down_secs=_env_int("COOLDOWN_DOWN_SECS", 240),
        error_block_threshold_pct=_env_float("ERROR_BLOCK_THRESHOLD_PCT", 1.0),
        ewma_alpha=_env_float("EWMA_ALPHA", 0.3),
        stability_penalty=_env_float("STABILITY_PENALTY", 0.05),
        blocked_penalty=_env_float("BLOCKED_PENALTY", 0.05),
    )

    sd = Path(args.save_dir)
    sd.mkdir(parents=True, exist_ok=True)

    if args.load_model:
        logger.info(f"Loading model from {args.load_model}")
        try:
            model = DQN.load(args.load_model, env=env)
        except ValueError as exc:
            logger.warning(
                "Model/environment observation-space mismatch: %s. "
                "Loading model without env and running manual rollout/eval.",
                exc,
            )
            model = DQN.load(args.load_model)

        if args.eval_episodes > 0:
            # Use a manual evaluation loop to avoid SB3 observation-space
            # checks that can fail when the saved model and runtime env differ.
            logger.info(
                "Starting manual evaluation for %d episodes", args.eval_episodes
            )
            rewards = []
            for ep in range(args.eval_episodes):
                obs = env.reset()
                ep_reward = 0.0
                while True:
                    obs_arr = obs_to_array(obs)
                    action, _states = model.predict(obs_arr, deterministic=True)
                    obs, reward, terminated, _ = env.step(int(action))
                    ep_reward += float(reward)
                    if terminated:
                        break
                rewards.append(ep_reward)
                logger.info("Episode %d reward=%.3f", ep + 1, ep_reward)

            mean_reward = statistics.mean(rewards) if rewards else 0.0
            std_reward = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
            logger.info(
                "Evaluate (manual): mean_reward=%.3f +/- %.3f", mean_reward, std_reward
            )
        else:
            logger.info("Model loaded. Starting interactive rollout (Ctrl-C to stop)")
            obs = env.reset()
            while True:
                obs_arr = obs_to_array(obs)
                action, _states = model.predict(obs_arr, deterministic=True)
                obs, reward, terminated, _ = env.step(int(action))
                logger.info(
                    f"Action={action} reward={reward:.3f} terminated={terminated}"
                )
                log_verbose_details(
                    observation=obs,
                    agent=SB3AgentWrapper(model),
                    verbose=True,
                    logger=logger,
                )
                if terminated:
                    obs = env.reset()
        return
    logger.error(
        "No --load-model provided. This script is for running/evaluation only."
        " Use the offline trainer or other training scripts to create a model."
    )
    return


if __name__ == "__main__":
    main()
