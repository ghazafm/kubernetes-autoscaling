import time
from logging import Logger
from typing import Optional

from database.influxdb import InfluxDB
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from prometheus_api_client import PrometheusConnect
from utils import get_metrics, wait_for_pods_ready


class KubernetesEnv:
    def __init__(  # noqa: PLR0913
        self,
        min_replicas: int = 1,
        max_replicas: int = 50,
        iteration: int = 100,
        namespace: str = "default",
        deployment_name: str = "default",
        min_cpu: float = 20,
        min_memory: float = 20,
        max_cpu: float = 90,
        max_memory: float = 90,
        max_response_time: float = 100.0,
        timeout: int = 120,
        wait_time: int = 30,
        verbose: bool = False,
        logger: Optional[Logger] = None,
        influxdb: Optional[InfluxDB] = None,
        prometheus_url: str = "http://localhost:1234/prom",
        metrics_endpoints_method: list[tuple[str, str]] = (
            ("/", "GET"),
            ("/docs", "GET"),
        ),
        metrics_interval: int = 15,
        metrics_quantile: float = 0.90,
        max_scaling_retries: int = 1000,
        response_time_weight: float = 1.0,
        cpu_memory_weight: float = 0.5,
        cost_weight: float = 0.3,
    ) -> None:
        self.logger = logger
        config.load_kube_config()
        self.cluster = client.AppsV1Api()
        self.api = client.CustomObjectsApi()
        self.core = client.CoreV1Api()
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.range_replicas = max(1, self.max_replicas - self.min_replicas)
        self.iteration = iteration
        self.initial_iteration = iteration
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.min_cpu = min_cpu
        self.min_memory = min_memory
        self.max_cpu = max_cpu
        self.max_memory = max_memory
        self.max_response_time = max_response_time
        self.verbose = verbose
        self.timeout = timeout
        self.wait_time = wait_time
        self.last_action = 0
        self.influxdb = influxdb
        self.prometheus = PrometheusConnect(
            url=prometheus_url,
            disable_ssl=True,
        )
        self.metrics_endpoints_method = metrics_endpoints_method
        self.metrics_interval = metrics_interval
        self.metrics_quantile = metrics_quantile
        self.max_scaling_retries = max_scaling_retries

        self.action_space = list(range(100))
        self.response_time_weight = response_time_weight
        self.cpu_memory_weight = cpu_memory_weight
        self.cost_weight = cost_weight

        self.observation_space = {
            "cpu_usage": (0, 100.0),
            "memory_usage": (0, 100.0),
            "response_time": (0, 100.0),
            "last_action": (0, 99),  # Fixed: should be 0-99, not 1-99
        }

    def _scale(self) -> None:
        """Scale deployment with persistent retry until success.

        This method will keep retrying indefinitely until the scaling operation
        succeeds, using exponential backoff with jitter to handle cluster issues.
        """
        HTTP_INTERNAL_SERVER_ERROR = 500
        HTTP_CONFLICT = 409

        base_timeout = 60
        max_timeout = 300
        base_delay = 1.0
        max_delay = 30.0
        attempt = 0

        self.logger.info(
            f"Scaling to {self.replica_state} replicas | action {self.last_action}%"
        )

        while attempt < self.max_scaling_retries:
            attempt += 1

            current_timeout = min(base_timeout * (1.5 ** (attempt - 1)), max_timeout)
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)

            try:
                self.cluster.patch_namespaced_deployment_scale(
                    name=self.deployment_name,
                    body=client.V1Scale(
                        spec=client.V1ScaleSpec(replicas=int(self.replica_state))
                    ),
                    namespace=self.namespace,
                    _request_timeout=current_timeout,
                )

                if attempt > 1:
                    self.logger.info(
                        f"✅ Scaling succeeded on attempt {attempt} "
                        f"(timeout: {current_timeout}s)"
                    )
                return

            except ApiException as e:
                if e.status == HTTP_INTERNAL_SERVER_ERROR:
                    if "etcdserver: request timed out" in str(e):
                        self.logger.warning(
                            f"⏰ Etcd timeout on attempt {attempt} "
                            f"(timeout: {current_timeout}s). "
                            f"Retrying in {delay:.1f}s..."
                        )
                    else:
                        self.logger.warning(
                            f"🔄 Server error on attempt {attempt}: {e.reason}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                elif e.status == HTTP_CONFLICT:
                    self.logger.warning(
                        f"⚠️  Conflict on attempt {attempt} "
                        f"(likely concurrent modification). "
                        f"Retrying in {delay:.1f}s..."
                    )
                else:
                    self.logger.warning(
                        f"🚨 API error on attempt {attempt} "
                        f"(status: {e.status}): {e.reason}. "
                        f"Retrying in {delay:.1f}s..."
                    )

            except Exception as e:
                self.logger.warning(
                    f"💥 Unexpected error on attempt {attempt}: {type(e).__name__}: "
                    f"{e}. "
                    f"Retrying in {delay:.1f}s..."
                )

            if attempt % 10 == 0:
                self.logger.info(
                    f"🔄 Still retrying scaling operation... "
                    f"Attempt {attempt}, next timeout: {current_timeout}s"
                )

            time.sleep(delay)

        self.logger.error(
            f"❌ CRITICAL: Failed to scale after {self.max_scaling_retries} attempts. "
            f"This indicates a serious cluster issue. "
            f"Proceeding with current replica state to avoid blocking training."
        )

    def _calculate_reward(self) -> float:
        # membuat jadi percentage, agar applicable di semua skala SLA
        response_time_percentage = (self.response_time / self.max_response_time) * 100.0

        # Penalti biner: 0 jika dalam batas, 1 jika di luar
        if self.cpu_usage < self.min_cpu:
            cpu_pen = (self.min_cpu - self.cpu_usage) / self.min_cpu
        elif self.cpu_usage > self.max_cpu:
            cpu_pen = (self.cpu_usage - self.max_cpu) / (100 - self.max_cpu)
        else:
            cpu_pen = 0.0

        if self.memory_usage < self.min_memory:
            mem_pen = (self.min_memory - self.memory_usage) / self.min_memory
        elif self.memory_usage > self.max_memory:
            mem_pen = (self.memory_usage - self.max_memory) / (100 - self.max_memory)
        else:
            mem_pen = 0.0

        # Response time penalty only if exceeding 100% of SLA, normalized to ~0..1
        # 0-100% = no penalty, >100% = increasing penalty
        resp_pen = min(
            self.response_time_weight,
            max(0.0, (response_time_percentage - 100.0) / 100.0),
        )  # Cap penalty at 1.0 for stability
        # max() ensures no negative penalty when response_time < 100% SLA (ReLU-like)

        cpu_mem_pen = self.cpu_memory_weight * (cpu_pen + mem_pen)

        cost_pen = (
            self.cost_weight
            * (self.replica_state - self.min_replicas)
            / self.range_replicas
        )

        # Reward sederhana: mulai dari 1, kurangi penalti
        reward = 1.0 - resp_pen - cpu_mem_pen - cost_pen

        # Clamp agar stabil
        return float(max(min(reward, 1.0), -1.0))

    def _scale_and_get_metrics(self) -> None:
        self._scale()
        increase: int = self.replica_state > self.replica_state_old
        ready, desired_replicas, ready_replicas = wait_for_pods_ready(
            prometheus=self.prometheus,
            deployment_name=self.deployment_name,
            desired_replicas=self.replica_state,
            namespace=self.namespace,
            timeout=self.timeout,
            logger=self.logger,
        )
        self.cpu_usage, self.memory_usage, self.response_time, self.replica = (
            get_metrics(
                replicas=ready_replicas,
                timeout=self.timeout,
                namespace=self.namespace,
                deployment_name=self.deployment_name,
                wait_time=self.wait_time,
                prometheus=self.prometheus,
                interval=self.metrics_interval,
                quantile=self.metrics_quantile,
                endpoints_method=self.metrics_endpoints_method,
                increase=increase,
                logger=self.logger,
            )
        )

        if not ready:
            self.logger.warning(
                f"Pods are not ready, {ready_replicas}/{desired_replicas} ready"
            )

    def _get_observation(self) -> dict[str, float]:
        response_time_percentage = min(
            (self.response_time / self.max_response_time) * 100.0, 100.0
        )
        return {
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "response_time": response_time_percentage,
            "last_action": self.last_action,
        }

    def step(self, action: int) -> tuple[dict[str, float], float, bool, dict]:
        self.last_action = action

        # percentage = action + 1  # Convert 0-99 to 1-100%
        # ratio = percentage / 100.0
        percentage = (
            (action / 99.0) if len(self.action_space) > 1 else 0.0
        )  # Map 0-99 to 0.0-1.0
        self.replica_state_old = self.replica_state
        self.replica_state = round(self.min_replicas + percentage * self.range_replicas)
        self.replica_state = max(
            self.min_replicas, min(self.replica_state, self.max_replicas)
        )

        self._scale_and_get_metrics()

        reward = self._calculate_reward()

        self.iteration -= 1
        terminated = bool(self.iteration <= 0)

        observation = self._get_observation()
        info = {
            "iteration": self.iteration,
            "action": action,
            "reward": reward,
            "terminated": terminated,
            "replica_state": self.replica_state,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "response_time": self.response_time,
            "last_action": self.last_action,
        }
        self.influxdb.write_point(
            measurement="autoscaling_metrics",
            tags={
                "namespace": self.namespace,
                "deployment": self.deployment_name,
            },
            fields={**info},
        ) if self.influxdb else None
        return observation, reward, terminated, info

    def reset(self) -> dict[str, float]:
        self.iteration = self.initial_iteration
        self.replica_state_old = (
            self.replica_state if hasattr(self, "replica_state") else self.min_replicas
        )
        self.replica_state = self.min_replicas
        self._scale_and_get_metrics()
        self.last_action = 0
        return self._get_observation()
