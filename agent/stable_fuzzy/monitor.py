import argparse
import ast
import os

from dotenv import load_dotenv
from environment import calculate_reward
from prometheus_api_client import PrometheusConnect
from utils import (
    get_metrics,
    get_raw_metrics,
    get_replica,
    setup_logger,
)

from database import InfluxDB

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--test", action="store_true", help="Use .env.test file")
parser.add_argument("--deployment", type=str, help="Deployment name to monitor")
args, _ = parser.parse_known_args()

if args.test:
    load_dotenv(".env.test")
else:
    load_dotenv()


logger, log_dir = setup_logger(
    "monitoring_kubernetes", log_level=os.getenv("LOG_LEVEL", "INFO"), log_to_file=True
)

metrics_endpoints_method = ast.literal_eval(os.getenv("METRICS_ENDPOINTS_METHOD"))
prometheus_url = os.getenv("PROMETHEUS_URL")
namespace = os.getenv("NAMESPACE", "default")
deployment_name = args.deployment or os.getenv("DEPLOYMENT_NAME", "my-deployment")
metrics_interval = int(os.getenv("METRICS_INTERVAL"))
metrics_quantile = float(os.getenv("METRICS_QUANTILE"))
max_response_time = float(os.getenv("MAX_RESPONSE_TIME"))

# Scaling range for action calculation
min_replicas = int(os.getenv("MIN_REPLICAS", "1"))
max_replicas = int(os.getenv("MAX_REPLICAS", "10"))
range_replicas = max(1, max_replicas - min_replicas)

# Reward weights
weight_response_time = float(os.getenv("WEIGHT_RESPONSE_TIME", "1.0"))
weight_cost = float(os.getenv("WEIGHT_COST", "1.0"))

influxdb = InfluxDB(
    logger=logger,
    url=os.getenv("INFLUXDB_URL", "http://localhost:8086"),
    token=os.getenv("INFLUXDB_TOKEN", "my-token"),
    org=os.getenv("INFLUXDB_ORG", "my-org"),
    bucket=os.getenv("INFLUXDB_BUCKET", "my-bucket"),
)
prometheus = PrometheusConnect(
    url=prometheus_url,
    disable_ssl=True,
)

logger.info("Starting Kubernetes cluster monitoring...")
while True:
    cpu, memory, response_time = get_metrics(
        prometheus=prometheus,
        namespace=namespace,
        deployment_name=deployment_name,
        interval=metrics_interval,
        max_response_time=max_response_time,
        quantile=metrics_quantile,
        endpoints_method=metrics_endpoints_method,
    )
    cpu_raw, memory_raw, cpu_limit, memory_limit, response_time_raw = get_raw_metrics(
        prometheus=prometheus,
        namespace=namespace,
        deployment_name=deployment_name,
        interval=metrics_interval,
        quantile=metrics_quantile,
        endpoints_method=metrics_endpoints_method,
    )

    desired_replica, replica = get_replica(
        prometheus,
        namespace,
        deployment_name,
        wait_time=0.5,
    )

    # Calculate action from replica (reverse of environment logic)
    action = int((replica - min_replicas) * 99 / range_replicas)
    action = max(0, min(99, action))

    reward, reward_details = calculate_reward(
        action=action,
        response_time=response_time,
        weight_response_time=weight_response_time,
        weight_cost=weight_cost,
    )

    info = {
        "cpu": cpu,
        "memory": memory,
        "response_time": response_time,
        "cpu_raw": cpu_raw,
        "memory_raw": memory_raw,
        "response_time_raw": response_time_raw,
        "cpu_limit": cpu_limit,
        "memory_limit": memory_limit,
        "replicas": replica,
        "desired_replicas": desired_replica,
        "action": action,
        "reward": reward,
        "rt_penalty": reward_details["rt_penalty"],
        "cost_penalty": reward_details["cost_eff"],
        "total_penalty": reward_details["total_penalty"],
    }

    influxdb.write_point(
        measurement="monitoring_cluster",
        tags={
            "namespace": namespace,
            "deployment": deployment_name,
        },
        fields={**info},
    )

    logger.info(f"Monitoring data written to InfluxDB: {info}")
