import math
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
        iteration: int | float = 100,  # int for training, float("inf") for production
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
        error_rate_weight: float = 0.8,
        cpu_memory_weight: float = 0.4,
        cost_weight: float = 0.2,
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
        self.error_rate_weight = error_rate_weight
        self.cpu_memory_weight = cpu_memory_weight
        self.cost_weight = cost_weight

        self.observation_space = {
            "cpu_usage": (0, 100.0),
            "memory_usage": (0, 100.0),
            "response_time": (0, 1000.0),  # FIXED: Allow RT up to 1000% (10x violation)
            "current_replica_pct": (0, 100.0),
            "last_action": (0, 99),
            "cpu_delta": (-100.0, 100.0),
            "memory_delta": (-100.0, 100.0),
            "rt_delta": (-1000.0, 1000.0),  # FIXED: Match RT range
            "time_in_state": (0, 1.0),
            "scaling_direction": (0, 1.0),  # 0=down, 0.5=same, 1=up
            # NEW: Load indicators (scale-independent)
            "rps_per_pod": (0, 100.0),  # Requests per second per pod
            "rps_delta": (-100.0, 100.0),  # Change in RPS per pod
            "error_rate": (0, 10.0),  # Error percentage (0-10%)
        }

        # Track last known good metrics for fallback during timeout
        self.last_known_metrics: (
            tuple[float, float, float, int, float, float] | None
        ) = None

        # Track previous metrics for delta calculation
        self.prev_cpu_usage = 0.0
        self.prev_memory_usage = 0.0
        self.prev_response_time = 0.0
        self.prev_rps_per_pod = 0.0  # NEW: Track previous RPS per pod

        # Track time in current replica state for stability
        self.steps_at_current_replica = 0
        self.max_steps_tracking = 20

        self.logger.info("Initialized KubernetesEnv environment")
        self.logger.info(f"Environment configuration: {self.__dict__}")

    def verify_deployment_resources(self) -> bool:
        """
        Verify that all pods in the deployment have resource limits configured.

        Returns:
            bool: True if all pods have limits, False otherwise

        Raises:
            ValueError: If deployment has pods without resource limits
        """
        try:
            pods = self.core.list_namespaced_pod(
                namespace=self.namespace, label_selector=f"app={self.deployment_name}"
            )

            if not pods.items:
                self.logger.warning(
                    f"No pods found for deployment {self.deployment_name}. "
                    "Skipping resource limit verification."
                )
                return True

            missing_limits = []
            for pod in pods.items:
                pod_name = pod.metadata.name
                for container in pod.spec.containers:
                    if not container.resources or not container.resources.limits:
                        missing_limits.append(
                            f"{pod_name}/{container.name}: No limits defined"
                        )
                    else:
                        if not container.resources.limits.get("cpu"):
                            missing_limits.append(
                                f"{pod_name}/{container.name}: CPU limit missing"
                            )
                        if not container.resources.limits.get("memory"):
                            missing_limits.append(
                                f"{pod_name}/{container.name}: Memory limit missing"
                            )

            if missing_limits:
                error_msg = (
                    f"Deployment {self.deployment_name} has pods without proper "
                    f"resource limits:\n"
                    + "\n".join(missing_limits)
                    + "\n\nThis will cause metrics collection to fail. "
                    "Please update your deployment with resource limits."
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            self.logger.info(
                f"✅ All pods in deployment {self.deployment_name} have "
                "proper resource limits configured"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to verify deployment resources: {e}")
            raise

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
        # Convert response time to percentage for scale-independent SLA
        response_time_percentage = (self.response_time / self.max_response_time) * 100.0

        # CPU penalty: normalized to [0, 1] range
        if self.cpu_usage < self.min_cpu:
            cpu_pen = (self.min_cpu - self.cpu_usage) / 100.0
        elif self.cpu_usage > self.max_cpu:
            cpu_pen = (self.cpu_usage - self.max_cpu) / 100.0
        else:
            cpu_pen = 0.0

        # Memory penalty: normalized to [0, 1] range
        if self.memory_usage < self.min_memory:
            mem_pen = (self.min_memory - self.memory_usage) / 100.0
        elif self.memory_usage > self.max_memory:
            mem_pen = (self.memory_usage - self.max_memory) / 100.0
        else:
            mem_pen = 0.0

        # Response Time Penalty - FIXED FORMULA
        # Uses sigmoid-like curve to properly handle extreme violations
        # while keeping penalties in [0, 1] range
        RESPONSE_TIME_HIGH_THRESHOLD = 80.0
        RESPONSE_TIME_VIOLATION_THRESHOLD = 100.0

        if response_time_percentage < RESPONSE_TIME_HIGH_THRESHOLD:
            # No penalty for acceptable response times
            resp_pen = 0.0
        elif response_time_percentage < RESPONSE_TIME_VIOLATION_THRESHOLD:
            # Linear ramp from 80% to 100%: penalty grows from 0.0 to 1.0
            normalized_rt = (response_time_percentage - 80.0) / 20.0  # 0.0 to 1.0
            resp_pen = normalized_rt
        else:
            # Exponential penalty for severe violations (RT > 100%)
            # This ensures severe violations are heavily penalized
            # but still capped at 1.0 to prevent overwhelming other penalties
            violation_severity = (response_time_percentage - 100.0) / 100.0
            # Use tanh to create smooth saturation at 1.0

            resp_pen = math.tanh(
                1.0 + violation_severity
            )  # Approaches 1.0 asymptotically

        # Error Rate Penalty - normalized to [0, 1]
        ERROR_RATE_THRESHOLD = 1.0
        if self.error_rate > ERROR_RATE_THRESHOLD:
            error_pen = min((self.error_rate - 1.0) / 9.0, 1.0)
        else:
            error_pen = 0.0

        # Combine penalties with weights
        # All individual penalties are now in [0, 1] range
        # Weights determine relative importance
        weighted_resp_pen = self.response_time_weight * resp_pen
        weighted_error_pen = self.error_rate_weight * error_pen
        weighted_cpu_mem_pen = self.cpu_memory_weight * (cpu_pen + mem_pen)
        weighted_cost_pen = (
            self.cost_weight
            * (self.replica_state - self.min_replicas)
            / self.range_replicas
        )

        # Calculate total penalty
        total_penalty = (
            weighted_resp_pen
            + weighted_error_pen
            + weighted_cpu_mem_pen
            + weighted_cost_pen
        )

        # Normalize total penalty to ensure reward stays in [-1, 1] range
        # Maximum possible penalty is sum of all weights
        max_possible_penalty = (
            self.response_time_weight
            + self.error_rate_weight
            + self.cpu_memory_weight * 2.0  # CPU + Memory can both max out
            + self.cost_weight
        )

        # Normalize penalty to [0, 1] range
        normalized_penalty = min(total_penalty / max_possible_penalty, 1.0)

        # Final reward: 1.0 (perfect) to -1.0 (worst)
        reward = 1.0 - 2.0 * normalized_penalty

        # Clamp for stability (should already be in range, but safety check)
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
        (
            self.cpu_usage,
            self.memory_usage,
            self.response_time,
            self.replica,
            self.request_rate,
            self.error_rate,
        ) = get_metrics(
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
            last_known_metrics=self.last_known_metrics,
            desired_replicas=desired_replicas,
        )

        if self.replica > 0:
            self.last_known_metrics = (
                self.cpu_usage,
                self.memory_usage,
                self.response_time,
                self.replica,
                self.request_rate,
                self.error_rate,
            )

        if not ready:
            self.logger.warning(
                f"Pods are not ready, {ready_replicas}/{desired_replicas} ready"
            )

    def _get_observation(self) -> dict[str, float]:
        # FIXED: Don't clamp RT percentage - let DQN see full severity of violations
        # This allows the neural network to learn the true state of the system
        response_time_percentage = (self.response_time / self.max_response_time) * 100.0

        # However, cap at a reasonable maximum for numerical stability (10x violation)
        # This prevents extreme outliers from destabilizing training
        response_time_percentage = min(response_time_percentage, 1000.0)

        # Calculate current replica percentage based on ACTUAL ready replicas
        current_replica_percentage = (
            (self.replica - self.min_replicas) / self.range_replicas * 100.0
            if self.range_replicas > 0
            else 0.0
        )

        # Calculate deltas (normalized to -100 to +100 range)
        cpu_delta = self.cpu_usage - self.prev_cpu_usage
        memory_delta = self.memory_usage - self.prev_memory_usage
        rt_delta = response_time_percentage - self.prev_response_time

        # Calculate scaling direction: -1 (down), 0 (same), +1 (up)
        # Based on actual replica change, not action intent
        if hasattr(self, "prev_replica"):
            scaling_direction_raw = self.replica - self.prev_replica
            if scaling_direction_raw > 0:
                scaling_direction = 1.0  # Scaled up
            elif scaling_direction_raw < 0:
                scaling_direction = 0.0  # Scaled down
            else:
                scaling_direction = 0.5  # No change
        else:
            scaling_direction = 0.5  # First observation, no previous state

        # Calculate time in current state (normalized 0-1)
        time_in_state = min(
            self.steps_at_current_replica / self.max_steps_tracking, 1.0
        )

        # Calculate per-pod RPS (scale-independent normalization)
        # This makes the metric flexible across any replica range
        rps_per_pod = (self.request_rate / self.replica) if self.replica > 0 else 0.0

        # Calculate RPS delta (change in per-pod load)
        rps_delta = rps_per_pod - self.prev_rps_per_pod

        return {
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "response_time": response_time_percentage,
            "current_replica_pct": current_replica_percentage,
            "last_action": self.last_action,
            "cpu_delta": cpu_delta,
            "memory_delta": memory_delta,
            "rt_delta": rt_delta,
            "time_in_state": time_in_state,
            "scaling_direction": scaling_direction,
            # NEW: Load indicators
            "rps_per_pod": rps_per_pod,
            "rps_delta": rps_delta,
            "error_rate": self.error_rate,
        }

    def step(self, action: int) -> tuple[dict[str, float], float, bool, dict]:
        self.last_action = action

        # Store previous replica count for scaling direction calculation
        self.prev_replica = (
            self.replica if hasattr(self, "replica") else self.min_replicas
        )

        # Map discrete action (0-99) to continuous percentage (0.0-1.0)
        # Action 0 → 0.0 (min_replicas)
        # Action 99 → 1.0 (max_replicas)
        # Example: min=1, max=12, action=50 → 50/99≈0.505 → 1+0.505*11≈6.5→7 replicas
        percentage = (
            (action / 99.0) if len(self.action_space) > 1 else 0.0
        )  # Map 0-99 to 0.0-1.0
        self.replica_state_old = self.replica_state
        self.replica_state = round(self.min_replicas + percentage * self.range_replicas)
        self.replica_state = max(
            self.min_replicas, min(self.replica_state, self.max_replicas)
        )

        self._scale_and_get_metrics()

        # Update time-in-state counter
        if self.replica == self.prev_replica:
            self.steps_at_current_replica += 1
        else:
            self.steps_at_current_replica = 0

        reward = self._calculate_reward()

        self.iteration -= 1
        terminated = bool(self.iteration <= 0)

        observation = self._get_observation()

        # Update previous metrics for next delta calculation
        response_time_percentage = min(
            (self.response_time / self.max_response_time) * 100.0, 100.0
        )
        rps_per_pod = (self.request_rate / self.replica) if self.replica > 0 else 0.0
        self.prev_cpu_usage = self.cpu_usage
        self.prev_memory_usage = self.memory_usage
        self.prev_response_time = response_time_percentage
        self.prev_rps_per_pod = rps_per_pod
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
        self.last_known_metrics = None

        # Reset previous metrics tracking
        self.prev_cpu_usage = 0.0
        self.prev_memory_usage = 0.0
        self.prev_response_time = 0.0
        self.prev_rps_per_pod = 0.0  # NEW: Reset RPS tracking

        # Reset time-in-state tracking
        self.steps_at_current_replica = 0

        self._scale_and_get_metrics()

        # Verify resource limits are configured (will raise if missing)
        try:
            self.verify_deployment_resources()
        except ValueError:
            self.logger.error(
                "Resource limit verification failed. "
                "Training cannot proceed without proper limits."
            )
            raise

        # Initialize previous metrics after first measurement
        response_time_percentage = min(
            (self.response_time / self.max_response_time) * 100.0, 100.0
        )
        rps_per_pod = (self.request_rate / self.replica) if self.replica > 0 else 0.0
        self.prev_cpu_usage = self.cpu_usage
        self.prev_memory_usage = self.memory_usage
        self.prev_response_time = response_time_percentage
        self.prev_rps_per_pod = rps_per_pod
        self.prev_replica = self.replica

        self.last_action = 0
        return self._get_observation()
