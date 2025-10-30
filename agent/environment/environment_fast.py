from logging import Logger
from typing import Optional

import pandas as pd
from database.influxdb import InfluxDB
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect


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
        cpu_memory_weight: float = 0.5,
        cost_weight: float = 0.3,
        data_path: Optional[str] = None,
        data_frame: Optional[pd.DataFrame] = None,
    ) -> None:
        self.logger = logger
        # For the "fast" environment we don't perform real cluster scaling.
        # Still initialize clients for compatibility, but _scale() is a no-op.
        try:
            config.load_kube_config()
            self.cluster = client.AppsV1Api()
            self.api = client.CustomObjectsApi()
            self.core = client.CoreV1Api()
        except Exception:
            # In CI / headless training there may be no kubeconfig; ignore.
            self.cluster = None
            self.api = None
            self.core = None
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
        # Data source for offline/simulated training. The caller should pass
        # either `data_path` (CSV file) or `data_frame` (pandas.DataFrame)
        # containing the captured traffic rows. Expected columns: at least
        # ['cpu_usage', 'memory_usage', 'response_time', 'replica']
        self.data: pd.DataFrame | None = None
        self.data_idx = 0
        # If caller provided a DataFrame or path, load it. Keep DataFrame
        # indexed and ready to be read sequentially by `step()`.
        if data_frame is not None:
            self.data = data_frame.reset_index(drop=True)
        elif data_path:
            self.data = pd.read_csv(data_path).reset_index(drop=True)
        else:
            # No data provided: leave None and rely on user to set `env.data`
            self.data = None

        self.logger.info("Initialized KubernetesEnv environment")
        self.logger.info(f"Environment configuration: {self.__dict__}")

    def _scale(self) -> None:
        """Fast/no-op scale used for offline training.

        We intentionally don't call the Kubernetes API here. The environment
        simulates that scaling completed instantly. This avoids hitting a
        cluster during offline model training with captured traffic data.
        """
        if self.logger:
            self.logger.debug("[fast env] _scale called - no-op (simulated scaling)")

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
        # Simulate scaling (no-op) and then read the next row from the
        # captured traffic dataset (if provided).
        self._scale()

        # If dataset not provided, fallback to zero metrics and replica
        if self.data is None or len(self.data) == 0:
            self.cpu_usage = 0.0
            self.memory_usage = 0.0
            self.response_time = 0.0
            self.replica = int(self.replica_state)
            if self.logger:
                self.logger.debug("[fast env] No data provided - using zeros")
            return

        # Clamp index and read row
        if self.data_idx >= len(self.data):
            # If we've exhausted the dataset, keep returning the last row
            self.data_idx = len(self.data) - 1

        row = self.data.iloc[self.data_idx]

        # Accept common column names; fall back to positional access if needed
        def get_val(r, keys, default=0.0):
            for k in keys:
                if k in r:
                    return float(r[k])
            return float(default)

        self.cpu_usage = (
            get_val(row, ["cpu_usage", "cpu", "cpu%", "cpu_percent"]) or 0.0
        )
        self.memory_usage = (
            get_val(
                row,
                ["memory_usage", "mem_usage", "memory%", "memory_percent"],
            )
            or 0.0
        )
        self.response_time = (
            get_val(row, ["response_time", "resp_time", "latency_ms", "latency"]) or 0.0
        )
        # replica column may be named 'replica' or 'replicas'
        self.replica = int(
            get_val(
                row,
                ["replica", "replicas", "replica_count"],
                default=self.replica_state,
            )
        )

        if self.logger:
            msg = (
                f"[fast env] Read data row {self.data_idx}: cpu={self.cpu_usage}, "
                f"mem={self.memory_usage}, resp={self.response_time}, "
                f"replica={self.replica}"
            )
            self.logger.debug(msg)

        # advance to next record for next step
        self.data_idx += 1

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
