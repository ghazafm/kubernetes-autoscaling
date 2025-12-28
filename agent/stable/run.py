import ast
import os
import signal
import sys
import threading
import time

from dotenv import load_dotenv
from environment import KubernetesEnv
from stable_baselines3 import DQN
from utils import setup_logger

from database import InfluxDB

load_dotenv(".env.test")

# Graceful shutdown using threading Event
shutdown_event = threading.Event()


def shutdown_handler(signum, frame):
    shutdown_event.set()


signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

if __name__ == "__main__":
    start_time = int(time.time())
    logger, log_dir = setup_logger(
        "kubernetes_agent", log_level=os.getenv("LOG_LEVEL", "INFO"), log_to_file=True
    )

    influxdb = InfluxDB(
        logger=logger,
        url=os.getenv("INFLUXDB_URL", "http://localhost:8086"),
        token=os.getenv("INFLUXDB_TOKEN", "my-token"),
        org=os.getenv("INFLUXDB_ORG", "my-org"),
        bucket=os.getenv("INFLUXDB_BUCKET", "my-bucket"),
    )
    metrics_endpoints_method = ast.literal_eval(os.getenv("METRICS_ENDPOINTS_METHOD"))

    iteration = int(os.getenv("ITERATION", "100"))
    env = KubernetesEnv(
        min_replicas=int(os.getenv("MIN_REPLICAS", "1")),
        max_replicas=int(os.getenv("MAX_REPLICAS", "10")),
        iteration=iteration,
        namespace=os.getenv("NAMESPACE", "default"),
        deployment_name=os.getenv("DEPLOYMENT_NAME", "your-app"),
        max_response_time=float(os.getenv("MAX_RESPONSE_TIME", "1000")),
        timeout=int(os.getenv("TIMEOUT", "120")),
        wait_time=int(os.getenv("WAIT_TIME", "5")),
        logger=logger,
        influxdb=influxdb,
        prometheus_url=os.getenv("PROMETHEUS_URL", "http://localhost:9090"),
        metrics_endpoints_method=metrics_endpoints_method,
        metrics_interval=int(os.getenv("METRICS_INTERVAL", "60")),
        metrics_quantile=float(os.getenv("METRICS_QUANTILE", "0.9")),
        max_scaling_retries=int(os.getenv("MAX_SCALING_RETRIES", "3")),
        weight_response_time=float(os.getenv("WEIGHT_RESPONSE_TIME", "0.7")),
        weight_cost=float(os.getenv("WEIGHT_COST", "0.3")),
        render_mode="human",
        mode="prod",
    )

    logger.info("Loading model from %s", os.getenv("MODEL_PATH"))
    model = DQN.load(os.getenv("MODEL_PATH"), env=env, device="auto")
    vec_env = model.get_env()
    logger.info("Resetting environment...")
    logger.info(f"Deployment: {env.deployment_name} in Namespace: {env.namespace}")

    obs = vec_env.reset()

    episode = 0
    episode_reward = 0.0
    step_count = 0
    last_action = 0
    max_scale_down_steps = int(os.getenv("MAX_SCALE_DOWN", "100"))
    max_scale_up_steps = int(os.getenv("MAX_SCALE_UP", "100"))

    logger.info("Starting inference loop...")

    try:
        while not shutdown_event.is_set():
            try:
                action, _ = model.predict(obs, deterministic=True)
                # action_int = int(action[0])
                # if action > last_action:
                #     action[0] = min(action_int, last_action + max_scale_up_steps)
                # else:
                #     action[0] = max(action_int, last_action - max_scale_down_steps)

                obs, rewards, dones, info = vec_env.step(action)
                last_action = action

                episode_reward += rewards[0]
                step_count += 1

                if dones[0]:
                    episode += 1
                    logger.info(
                        f"Episode {episode} finished | Steps: {step_count} | "
                        f"Total Reward: {episode_reward:.3f}"
                    )
                    episode_reward = 0.0
                    step_count = 0

            except Exception as e:
                logger.error(f"Error during inference: {e}")
                time.sleep(5)

    finally:
        try:
            if "influxdb" in locals() and hasattr(influxdb, "close"):
                influxdb.close()
        except Exception:
            logger.exception("Error while closing InfluxDB client")

        logger.info("Shutting down gracefully...")
        sys.exit(0)
