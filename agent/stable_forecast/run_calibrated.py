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


def calibrate_action(action: int, state: dict) -> int:
    if state["min_action_seen"] is None or state["max_action_seen"] is None:
        if state["min_action_seen"] is None:
            state["min_action_seen"] = action
        if state["max_action_seen"] is None:
            state["max_action_seen"] = action
        logger.info(f"Calibration: Initialized with action {action}")

    old_min = state["min_action_seen"]
    old_max = state["max_action_seen"]
    state["min_action_seen"] = min(state["min_action_seen"], action)
    state["max_action_seen"] = max(state["max_action_seen"], action)

    if state["min_action_seen"] < old_min:
        logger.info(
            f"Calibration: Found lower action {state['min_action_seen']} "
            f"(was {old_min}) after {state['calibration_steps']} steps"
        )
    if state["max_action_seen"] > old_max:
        logger.info(
            f"Calibration: Found higher action {state['max_action_seen']} "
            f"(was {old_max}) after {state['calibration_steps']} steps"
        )
    state["calibration_steps"] += 1
    if state["calibration_steps"] == state["min_steps"]:
        logger.info(
            f"Calibration: Implemented after {state['calibration_steps']} steps. "
            f"Min action: {state['min_action_seen']}, "
            f"Max action: {state['max_action_seen']}"
        )

    action_range = state["max_action_seen"] - state["min_action_seen"]
    if action_range > 0 and state["calibration_steps"] >= state["min_steps"]:
        normalized = (action - state["min_action_seen"]) / action_range * 99
        return round(normalized)
    return action


def reverse_calibrate_action(calibrated_action: int, state: dict) -> int:
    """Convert calibrated action back to model's original action space."""
    action_range = state["max_action_seen"] - state["min_action_seen"]
    if (
        action_range > 0
        and state["calibration_steps"] >= state["min_steps"]
        and state["min_action_seen"] is not None
    ):
        original = (calibrated_action / 99.0) * action_range + state["min_action_seen"]
        return round(original)
    return calibrated_action


calibration_state = {
    "min_action_seen": int(os.getenv("MIN_ACTION_SEEN"))
    if os.getenv("MIN_ACTION_SEEN")
    else None,
    "max_action_seen": int(os.getenv("MAX_ACTION_SEEN"))
    if os.getenv("MAX_ACTION_SEEN")
    else None,
    "min_steps": int(os.getenv("MIN_CALIBRATION_STEPS", "100")),
    "calibration_steps": 0,
}


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
    scale_down_attempts = 0
    min_scale_down_attempts = int(os.getenv("MIN_SCALE_DOWN_ATTEMPTS", "3"))
    max_scale_down_steps = int(os.getenv("MAX_SCALE_DOWN_STEPS", "5"))
    logger.info(
        f"Minimum scale-down attempts before scaling-down: {min_scale_down_attempts}"
    )
    logger.info(
        f"Maximum scale-down steps when forcing scaling-down: {max_scale_down_steps}"
    )

    logger.info("Starting inference loop...")

    try:
        while not shutdown_event.is_set():
            try:
                action, _ = model.predict(obs, deterministic=True)
                raw_action = int(action[0])
                action_calibrated = calibrate_action(raw_action, calibration_state)

                idle = obs[0][3] == 0.0 and obs[0][6] == 0.0

                if action_calibrated > last_action:
                    scale_down_attempts = 0
                else:
                    scale_down_attempts += 1
                    if scale_down_attempts >= min_scale_down_attempts or idle:
                        if not idle:
                            action_calibrated = max(
                                action_calibrated, last_action - max_scale_down_steps
                            )
                            scale_down_attempts = 0
                            raw_action = reverse_calibrate_action(
                                action_calibrated, calibration_state
                            )
                    else:
                        action_calibrated = last_action

                obs, rewards, dones, info = vec_env.step([action_calibrated])
                last_action = int(action_calibrated)

                obs[0][0] = raw_action / 99.0

                episode_reward += rewards[0]
                step_count += 1

                if dones[0]:
                    episode += 1
                    baseline_info = (
                        f" | Min/Max action: {calibration_state['min_action_seen']}/"
                        f"{calibration_state['max_action_seen']} "
                        f"(observed after {calibration_state['calibration_steps']} steps)"  # noqa: E501
                        if calibration_state["min_action_seen"]
                        and calibration_state["max_action_seen"] is not None
                        else ""
                    )
                    logger.info(
                        f"Episode {episode} finished | Steps: {step_count} | "
                        f"Total Reward: {episode_reward:.3f}{baseline_info}"
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
