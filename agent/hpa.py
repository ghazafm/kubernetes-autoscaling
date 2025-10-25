import os
import time

from database import InfluxDB
from dotenv import load_dotenv
from prometheus_api_client import PrometheusConnect
from utils import get_metrics, setup_logger

load_dotenv("hpa.env")


class HPAMonitor:
    """
    Monitoring agent that only observes Kubernetes default HPA behavior.
    Scrapes metrics from Prometheus and writes to InfluxDB for comparison.
    Does NOT perform any scaling actions.
    """

    def __init__(
        self,
        namespace: str,
        deployment_name: str,
        prometheus_url: str,
        influxdb_url: str,
        influxdb_token: str,
        influxdb_org: str,
        influxdb_bucket: str,
        timeout: int = 30,
        metrics_interval: int = 15,
        metrics_quantile: float = 0.90,
        metrics_endpoints_method: list[list[str]] | None = None,
        wait_time: int = 30,
        check_interval: int = 10,
        logger=None,
    ) -> None:
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.timeout = timeout
        self.metrics_interval = metrics_interval
        self.metrics_quantile = metrics_quantile
        self.metrics_endpoints_method = metrics_endpoints_method or [
            ["/", "GET"],
            ["/docs", "GET"],
        ]
        self.wait_time = wait_time
        self.check_interval = check_interval
        self.logger = logger or setup_logger("HPAMonitor", "INFO")
        self.agent_type = "HPA_MONITOR"

        # Initialize Prometheus connection
        self.prometheus = PrometheusConnect(url=prometheus_url, disable_ssl=True)

        # Initialize InfluxDB connection
        self.influxdb = InfluxDB(
            url=influxdb_url,
            token=influxdb_token,
            org=influxdb_org,
            bucket=influxdb_bucket,
            logger=self.logger,
        )

        self.logger.info("Initialized HPA Monitor (Read-only)")
        self.logger.info(
            f"Monitoring deployment: {self.deployment_name} "
            f"in namespace: {self.namespace}"
        )
        self.logger.info(
            f"Wait time after scale-up: {self.wait_time}s, "
            f"Check interval: {self.check_interval}s"
        )

    def _get_current_replicas(self) -> int:
        """Get current number of ready replicas from Prometheus."""
        scope_ready = f"""
            (kube_pod_status_ready{{namespace="{self.namespace}",
             condition="true"}} == 1)
            and on(pod)
            (
              label_replace(
                kube_pod_owner{{namespace="{self.namespace}",
                 owner_kind="ReplicaSet"}},
                "replicaset", "$1", "owner_name", "(.*)"
              )
              * on(namespace, replicaset) group_left(owner_name)
                kube_replicaset_owner{{
                  namespace="{self.namespace}", owner_kind="Deployment",
                  owner_name="{self.deployment_name}"
                }}
            )
        """

        q_ready = f"scalar(sum({scope_ready}))"

        try:
            result = self.prometheus.custom_query(query=q_ready)
            ready_replicas = int(result[1]) if result and len(result) > 1 else 0
            self.logger.debug(f"Current ready replicas: {ready_replicas}")
            return ready_replicas
        except Exception as e:
            self.logger.error(f"Error querying ready replicas: {e}")
            return 0

    def _get_desired_replicas(self) -> int:
        """Get desired number of replicas from Prometheus."""
        q_desired = f"""
        scalar(
          sum(
            kube_deployment_spec_replicas{{namespace="{self.namespace}",
            deployment="{self.deployment_name}"}}
            )
        )
        """

        try:
            result = self.prometheus.custom_query(query=q_desired)
            desired_replicas = int(result[1]) if result and len(result) > 1 else 0
            self.logger.debug(f"Desired replicas: {desired_replicas}")
            return desired_replicas
        except Exception as e:
            self.logger.error(f"Error querying desired replicas: {e}")
            return 0

    def monitor(self):
        """
        Main monitoring loop that scrapes metrics and writes to InfluxDB.
        Only observes; does not perform any scaling actions.
        """
        iteration = 0
        previous_ready_replicas = 0
        self.logger.info("Starting monitoring loop...")

        while True:
            try:
                iteration += 1
                self.logger.info(f"=== Iteration {iteration} ===")

                # Get current replica counts
                ready_replicas = self._get_current_replicas()
                desired_replicas = self._get_desired_replicas()

                # Detect if new pods were provisioned (scale up)
                pods_increased = ready_replicas > previous_ready_replicas

                if pods_increased:
                    self.logger.info(
                        f"New pods detected! Scaled from {previous_ready_replicas} "
                        f"to {ready_replicas}. "
                        f"Waiting {self.wait_time}s for stabilization..."
                    )
                    time.sleep(self.wait_time)

                # Scrape metrics from Prometheus
                cpu_usage, memory_usage, response_time, _ = get_metrics(
                    replicas=ready_replicas,
                    timeout=self.timeout,
                    namespace=self.namespace,
                    deployment_name=self.deployment_name,
                    wait_time=0,  # Don't wait inside get_metrics, we handle it above
                    prometheus=self.prometheus,
                    interval=self.metrics_interval,
                    quantile=self.metrics_quantile,
                    endpoints_method=self.metrics_endpoints_method,
                    increase=False,
                    logger=self.logger,
                )

                # Log collected metrics
                self.logger.info(
                    f"Metrics - Ready: {ready_replicas}/{desired_replicas}, "
                    f"CPU: {cpu_usage:.2f}%, Memory: {memory_usage:.2f}%, "
                    f"Response Time: {response_time:.2f}ms"
                )

                # Write to InfluxDB
                self.influxdb.write_point(
                    measurement="autoscaling_metrics",
                    tags={
                        "namespace": self.namespace,
                        "deployment": self.deployment_name,
                        "algorithm": "HPA",
                    },
                    fields={
                        "iteration": iteration,
                        "replica_state": ready_replicas,
                        "desired_replicas": desired_replicas,
                        "cpu_usage": cpu_usage,
                        "memory_usage": memory_usage,
                        "response_time": response_time,
                    },
                )

                # Update previous replica count for next iteration
                previous_ready_replicas = ready_replicas

                # Wait before next scrape (shorter interval between checks)
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                self.logger.info("Monitoring stopped by user.")
                break
            except Exception as e:
                self.logger.exception(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)

    def close(self):
        """Cleanup resources."""
        self.logger.info("Closing connections...")
        self.influxdb.close()


def main():
    monitor = HPAMonitor(
        namespace=os.getenv("NAMESPACE", "default"),
        deployment_name=os.getenv("DEPLOYMENT_NAME"),
        prometheus_url=os.getenv("PROMETHEUS_URL", "http://localhost:9090"),
        influxdb_url=os.getenv("INFLUXDB_URL", "http://localhost:8086"),
        influxdb_token=os.getenv("INFLUXDB_TOKEN"),
        influxdb_org=os.getenv("INFLUXDB_ORG", "my-org"),
        influxdb_bucket=os.getenv("INFLUXDB_BUCKET", "my-bucket"),
        wait_time=int(os.getenv("WAIT_TIME", "30")),
        check_interval=int(os.getenv("CHECK_INTERVAL", "10")),
        metrics_interval=int(os.getenv("METRICS_INTERVAL", "15")),
        metrics_quantile=float(os.getenv("METRICS_QUANTILE", "0.90")),
    )

    try:
        monitor.monitor()
    finally:
        monitor.close()


if __name__ == "__main__":
    main()
