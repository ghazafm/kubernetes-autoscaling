import argparse
import ast
import os

from database import InfluxDB
from dotenv import load_dotenv
from prometheus_api_client import PrometheusConnect
from utils import calculate_distance, get_metrics, get_replica, setup_logger

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--test", action="store_true", help="Use .env.test file")
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
deployment_name = os.getenv("DEPLOYMENT_NAME")
metrics_interval = int(os.getenv("METRICS_INTERVAL"))
metrics_quantile = float(os.getenv("METRICS_QUANTILE"))
max_response_time = float(os.getenv("MAX_RESPONSE_TIME"))
max_cpu = float(os.getenv("MAX_CPU"))
min_cpu = float(os.getenv("MIN_CPU"))
max_memory = float(os.getenv("MAX_MEMORY"))
min_memory = float(os.getenv("MIN_MEMORY"))

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

    cpu_relative, memory_relative, cpu_distance, memory_distance = calculate_distance(
        cpu, memory, max_cpu, min_cpu, max_memory, min_memory
    )

    desired_replica, replica = get_replica(
        prometheus,
        namespace,
        deployment_name,
        wait_time=0.5,
    )

    info = {
        "cpu": cpu,
        "memory": memory,
        "response_time": response_time,
        "replicas": replica,
        "desired_replicas": desired_replica,
        "cpu_relative": cpu_relative,
        "memory_relative": memory_relative,
        "cpu_distance": cpu_distance,
        "memory_distance": memory_distance,
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
