import ast
import os
import time

import numpy as np
from database import InfluxDB
from dotenv import load_dotenv
from environment import (
    KubernetesEnv,
)
from func import train_agent
from rl import DQN, Q
from utils import (
    setup_logger,
)

load_dotenv()

if __name__ == "__main__":
    logger = setup_logger("kubernetes_agent", log_level="INFO", log_to_file=True)
    Influxdb = InfluxDB(
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
        min_replicas=int(os.getenv("MIN_REPLICAS", "1")),
        max_replicas=int(os.getenv("MAX_REPLICAS", "12")),
        iteration=int(os.getenv("ITERATION", "10")),
        namespace=os.getenv("NAMESPACE", "default"),
        deployment_name=os.getenv("DEPLOYMENT_NAME", "ecom-api"),
        min_cpu=int(os.getenv("MIN_CPU", "10")),
        min_memory=int(os.getenv("MIN_MEMORY", "10")),
        max_cpu=int(os.getenv("MAX_CPU", "90")),
        max_memory=int(os.getenv("MAX_MEMORY", "90")),
        timeout=int(os.getenv("TIMEOUT", "120")),
        wait_time=int(os.getenv("WAIT_TIME", "1")),
        verbose=True,
        logger=logger,
        influxdb=Influxdb,
        prometheus_url=os.getenv("PROMETHEUS_URL", "http://localhost:1234/prom"),
        metrics_endpoints_method=metrics_endpoints_method,
        metrics_interval=int(os.getenv("METRICS_INTERVAL", "15")),
        metrics_quantile=float(os.getenv("METRICS_QUANTILE", "0.90")),
    )

    choose_algorithm = os.getenv("ALGORITHM", "Q").upper()
    if choose_algorithm == "Q":
        algorithm = Q(
            learning_rate=float(os.getenv("LEARNING_RATE", None)),
            discount_factor=float(os.getenv("DISCOUNT_FACTOR", None)),
            epsilon_start=float(os.getenv("EPSILON_START", None)),
            epsilon_decay=float(os.getenv("EPSILON_DECAY", None)),
            epsilon_min=float(os.getenv("EPSILON_MIN", None)),
        )
    elif choose_algorithm == "DQN":
        algorithm = DQN(
            learning_rate=float(os.getenv("LEARNING_RATE", None)),
            discount_factor=float(os.getenv("DISCOUNT_FACTOR", None)),
            epsilon_start=float(os.getenv("EPSILON_START", None)),
            epsilon_decay=float(os.getenv("EPSILON_DECAY", None)),
            epsilon_min=float(os.getenv("EPSILON_MIN", None)),
            device=os.getenv("DEVICE", None),
            buffer_size=int(os.getenv("BUFFER_SIZE", None)),
            batch_size=int(os.getenv("BATCH_SIZE", None)),
            target_update_freq=int(os.getenv("TARGET_UPDATE_FREQ", None)),
            grad_clip_norm=float(os.getenv("GRAD_CLIP_NORM", None)),
        )
    else:
        raise ValueError(f"Unsupported algorithm: {choose_algorithm}")

    trained_agent, environment = train_agent(
        agent=algorithm,
        environment=env,
        episodes=int(os.getenv("EPISODES", None)),
        verbose=True,
        metrics_endpoints_method=os.getenv(
            "METRICS_ENDPOINTS_METHOD", "[['/', 'GET'], ['/docs', 'GET']]"
        ),
        logger=logger,
    )

    if hasattr(trained_agent, "q_table"):
        logger.info(f"\nQ-table size: {len(trained_agent.q_table)} states")
        logger.info("Sample Q-values:")
        for _, (state, q_values) in enumerate(list(trained_agent.q_table.items())[:5]):
            max_q = np.max(q_values)
            best_action = np.argmax(q_values)
            logger.info(
                f"State {state}: Best action = {best_action}, Max Q-value = {max_q:.3f}"
            )
    else:
        logger.info("\nDQN model trained (no Q-table to display)")
    # Save the trained Q-table
    trained_agent.save_model(f"model/final/{time.time()}.pkl")
