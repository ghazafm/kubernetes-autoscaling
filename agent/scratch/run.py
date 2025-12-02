import ast
import os
import time

from database import InfluxDB
from dotenv import load_dotenv
from environment import (
    KubernetesEnv,
)
from rl import DQN, Q
from utils import (
    log_verbose_details,
    setup_logger,
)


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

if __name__ == "__main__":
    start_time = int(time.time())
    logger = setup_logger(
        "kubernetes_agent", log_level=os.getenv("LOG_LEVEL", "INFO"), log_to_file=True
    )
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

    choose_algorithm = os.getenv("ALGORITHM", "Q").upper()
    if choose_algorithm == "Q":
        agent = Q(
            learning_rate=float(os.getenv("LEARNING_RATE", "0.1")),
            discount_factor=float(os.getenv("DISCOUNT_FACTOR", "0.95")),
            epsilon_start=0.0,
            epsilon_decay=0.0,
            epsilon_min=0.0,
            created_at=start_time,
            logger=logger,
        )
    elif choose_algorithm == "DQN":
        agent = DQN(
            learning_rate=float(os.getenv("LEARNING_RATE", "0.001")),
            discount_factor=float(os.getenv("DISCOUNT_FACTOR", "0.95")),
            epsilon_start=0.0,
            epsilon_decay=0.0,
            epsilon_min=0.0,
            device=os.getenv("DEVICE", "cpu"),
            buffer_size=int(os.getenv("BUFFER_SIZE", "50000")),
            batch_size=int(os.getenv("BATCH_SIZE", "64")),
            target_update_freq=int(os.getenv("TARGET_UPDATE_FREQ", "100")),
            grad_clip_norm=float(os.getenv("GRAD_CLIP_NORM", "10.0")),
            created_at=start_time,
            logger=logger,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {choose_algorithm}")

    model_path = os.getenv("MODEL_PATH", "model/dqn/1697041234_default/best_model.pth")
    logger.info(f"Loading trained model from: {model_path}")
    agent.load_model(model_path)
    agent.epsilon = 0.0
    agent.epsilon_decay = 0.0
    agent.epsilon_min = 0.0
    logger.info("Model loaded successfully. Running in inference mode (epsilon=0).")

    obs = env.reset()
    total_reward = 0.0
    step_count = 0

    while True:
        act = agent.get_action(obs)

        nxt, rew, term, info = env.step(act)

        total_reward += rew
        step_count += 1
        obs = nxt

        logger.info(
            f"Step: {step_count} | Action: {act} | Reward: {rew:.4f} | "
            f"Total Reward: {total_reward:.4f} | Iteration: {info['iteration']}"
        )

        logger.debug(f"Observation type: {type(obs)}, value: {obs}")

        log_verbose_details(
            observation=obs,
            agent=agent,
            verbose=True,
            logger=logger,
        )
