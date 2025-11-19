import logging
import time
from math import copysign
from typing import Optional

from database.influxdb import InfluxDB
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from prometheus_api_client import PrometheusConnect
from utils import get_metrics, wait_for_pods_ready

# NOTE:
# This module no longer reads tuning values from environment variables.
# Tuning/safety parameters (max_up_step, max_down_step, etc.) are accepted
# as constructor arguments to `KubernetesEnv` so callers (for example
# `agent/train.py`) can centralize env parsing and pass the values in.


class KubernetesEnv:
    def __init__(  # noqa: PLR0913, PLR0915
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
        logger: Optional[logging.Logger] = None,
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
        # Safety / stability tuning (prefer passing these from caller e.g. train.py)
        max_up_step: int = 4,
        max_down_step: int = 1,
        min_down_confirmations: int = 2,
        cooldown_up_secs: int = 60,
        cooldown_down_secs: int = 240,
        error_block_threshold_pct: float = 1.0,
        ewma_alpha: float = 0.3,
        stability_penalty: float = 0.05,
    ) -> None:
        # Ensure a usable logger is always present
        self.logger: logging.Logger = (
            logger if logger is not None else logging.getLogger(__name__)
        )
        config.load_kube_config()
        self.cluster = client.AppsV1Api()
        self.api = client.CustomObjectsApi()
        self.core = client.CoreV1Api()
        self.min_replicas: int = min_replicas
        self.max_replicas: int = max_replicas
        self.range_replicas: int = max(1, self.max_replicas - self.min_replicas)
        self.iteration: int | float = iteration
        self.initial_iteration: int | float = iteration
        self.namespace: str = namespace
        self.deployment_name: str = deployment_name
        self.min_cpu: float = min_cpu
        self.min_memory: float = min_memory
        self.max_cpu: float = max_cpu
        self.max_memory: float = max_memory
        self.max_response_time: float = max_response_time
        self.verbose: bool = verbose
        self.timeout: int = timeout
        self.wait_time: int = wait_time
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

        # Safety / stability parameters (passed from caller)
        self.max_up_step: int = int(max_up_step)
        self.max_down_step: int = int(max_down_step)
        self.min_down_confirmations: int = int(min_down_confirmations)
        self.cooldown_up_secs: int = int(cooldown_up_secs)
        self.cooldown_down_secs: int = int(cooldown_down_secs)
        # Threshold (in percent) above which error rate will block downscales.
        self.error_block_threshold_pct: float = float(error_block_threshold_pct)
        self.ewma_alpha: float = float(ewma_alpha)
        self.stability_penalty: float = float(stability_penalty)

        # Validation / safety clamps
        # EWMA_ALPHA should be in [0.0, 1.0]
        if self.ewma_alpha < 0.0:
            self.logger.warning(f"ewma_alpha {self.ewma_alpha} < 0.0, clamping to 0.0")
            self.ewma_alpha = 0.0
        if self.ewma_alpha > 1.0:
            self.logger.warning(f"ewma_alpha {self.ewma_alpha} > 1.0, clamping to 1.0")
            self.ewma_alpha = 1.0

        # Ensure sensible integer bounds
        if self.max_up_step < 1:
            self.logger.warning(f"max_up_step {self.max_up_step} < 1, setting to 1")
            self.max_up_step = 1
        if self.max_down_step < 1:
            self.logger.warning(f"max_down_step {self.max_down_step} < 1, setting to 1")
            self.max_down_step = 1
        self.cooldown_up_secs = max(self.cooldown_up_secs, 0)
        self.cooldown_down_secs = max(self.cooldown_down_secs, 0)

        # ERROR_BLOCK_THRESHOLD_PCT should be reasonable (0..100)
        self.error_block_threshold_pct = max(self.error_block_threshold_pct, 0.0)
        self.error_block_threshold_pct = min(self.error_block_threshold_pct, 100.0)

        # Runtime state for safety wrapper and initial replica bookkeeping
        self._pending_down_count: int = 0
        self._last_scale_up_time: float = 0.0
        self._last_scale_down_time: float = 0.0
        self.smoothed_response_time: float = 0.0
        self.smoothed_cpu: float = 0.0
        self.smoothed_memory: float = 0.0
        self.smoothed_error_rate: float = 0.0

        # Replica bookkeeping: ensure attributes exist and have sensible defaults
        # min_replicas is already an int from the constructor signature,
        # no need to cast again.
        self.replica: int = self.min_replicas
        self.replica_state: int = self.min_replicas
        self.replica_state_old: int = self.min_replicas
        self._last_applied_delta: int = 0

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
            "error_rate": (0, 100.0),  # Error percentage (0-100%)
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
        # Log a subset of configuration to avoid dumping large objects
        conf = {
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "namespace": self.namespace,
            "deployment_name": self.deployment_name,
        }
        self.logger.info(f"Environment configuration: {conf}")

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
                        spec=client.V1ScaleSpec(replicas=self.replica_state)
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
        response_time_percentage = (self.response_time / self.max_response_time) * 100.0
        response_time_percentage = min(response_time_percentage, 1000.0)

        # Logika dasar cpu dan memory penalty
        # Penalti dihitung berdasarkan seberapa jauh metrik berada di luar
        # batas yang ditentukan (min/max). Penalti dinormalisasi ke rentang 0..1
        if self.cpu_usage < self.min_cpu:
            cpu_pen = (self.min_cpu - self.cpu_usage) / 100.0
        elif self.cpu_usage > self.max_cpu:
            cpu_pen = (self.cpu_usage - self.max_cpu) / 100.0
        else:
            cpu_pen = 0.0

        if self.memory_usage < self.min_memory:
            mem_pen = (self.min_memory - self.memory_usage) / 100.0
        elif self.memory_usage > self.max_memory:
            mem_pen = (self.memory_usage - self.max_memory) / 100.0
        else:
            mem_pen = 0.0

        # Response time thresholds (persentase dari max_response_time)
        RESPONSE_TIME_HIGH_THRESHOLD = 80.0
        RESPONSE_TIME_VIOLATION_THRESHOLD = 100.0

        MAX_RESPONSE_PENALTY = 2.0
        if response_time_percentage <= RESPONSE_TIME_HIGH_THRESHOLD:
            resp_pen = 0.0
        elif response_time_percentage <= RESPONSE_TIME_VIOLATION_THRESHOLD:
            # Linear mapping from [HIGH, VIOLATION] -> [0.0, 1.0]
            resp_pen = (response_time_percentage - RESPONSE_TIME_HIGH_THRESHOLD) / (
                RESPONSE_TIME_VIOLATION_THRESHOLD - RESPONSE_TIME_HIGH_THRESHOLD
            )
        else:
            # Berguna untuk menambahkan penalti lebih dari batas pelanggaran
            # Menghitung penalti yang melebihi batas atas dengan maksimal
            # MAX_RESPONSE_PENALTY
            over = (
                response_time_percentage - RESPONSE_TIME_VIOLATION_THRESHOLD
            ) / RESPONSE_TIME_VIOLATION_THRESHOLD
            resp_pen = 1.0 + over

        # Memberi batasan pada resp_pen [0.0, MAX_RESPONSE_PENALTY]
        resp_pen = max(0.0, min(resp_pen, MAX_RESPONSE_PENALTY))

        # Normalisasi penalti error terhadap batas atas ruang observasi.
        # Ekspektasi error_rate adalah persentase 0..100.0
        ERROR_RATE_MAX = self.observation_space["error_rate"][1]
        error_pen = min(max(self.error_rate, 0.0) / ERROR_RATE_MAX, 1.0)

        # Menghitung biaya berdasarkan jumlah replica yang digunakan
        # Membuat model cenderung meminimalkan jumlah replica untuk efisiensi biaya
        cost_pen = (
            (self.replica_state - self.min_replicas) / self.range_replicas
            if self.range_replicas > 0
            else 0.0
        )

        # Menghitung total penalty dengan bobot masing-masing komponen
        weighted_resp_pen = self.response_time_weight * resp_pen
        weighted_error_pen = self.error_rate_weight * error_pen
        weighted_cpu_mem_pen = self.cpu_memory_weight * (cpu_pen + mem_pen)
        weighted_cost_pen = self.cost_weight * cost_pen

        total_penalty = (
            weighted_resp_pen
            + weighted_error_pen
            + weighted_cpu_mem_pen
            + weighted_cost_pen
        )

        max_possible_penalty = (
            self.response_time_weight
            + self.error_rate_weight
            + self.cpu_memory_weight * 2.0
            + self.cost_weight
        )

        # dinormalisasi agar hasil penalti tidak lebih dari 1
        normalized_penalty = min(total_penalty / max_possible_penalty, 1.0)

        # [-1, 1]
        # Dikali dua agar minus bisa mencapai -1.0 pada kasus terburuk
        reward = 1.0 - 2.0 * normalized_penalty

        # Menerapkan stability penalti jika ada scaling yang terjadi pada langkah ini
        stability_penalty = float(self.stability_penalty)
        if stability_penalty and self._last_applied_delta != 0:
            reward -= stability_penalty

        # memastikikan reward dalam batasan
        return float(max(min(reward, 1.0), -1.0))

    def _percent_action_to_safe_replicas(
        self, action_percent: int, current_replicas: int
    ) -> tuple[int, int]:
        # returns: (final_replicas, applied_delta)
        # Hitung target absolut dari 0..99 (aksi)
        target: int = (
            self.min_replicas + round((action_percent / 99.0) * self.range_replicas)
            if self.range_replicas > 0
            else self.min_replicas
        )

        # besar perubahan yang diinginkan (positif/negatif)
        # delta = besar perubahan
        delta: int = target - current_replicas
        if delta == 0:
            # no-op: reset pending down counter for safety and return
            self._pending_down_count = 0
            self.logger.debug(
                f"No scaling needed: action {action_percent} -> target {target}, "
                f"current {current_replicas}"
            )
            return current_replicas, 0

        #  menentukan maksium step kebawah atau keatas
        max_step = int(self.max_up_step if delta > 0 else self.max_down_step)

        # membatasi step agar sesuai dengan max_step
        # copysign untuk menjaga tanda (positif/negatif) dari delta
        step = int(copysign(min(abs(delta), max_step), delta))
        proposed = current_replicas + step

        # Melakukan pengecekan sebelum melakukan downscale (menjaga stabilitas)
        is_downscale: bool = proposed < current_replicas
        if is_downscale:
            # Menggunakan nilai EWMA (Exponential Weighted Moving Average),
            # agar tidak terlalu sensitif terhadap fluktuasi metrik.
            rt = self.smoothed_response_time
            cpu = self.smoothed_cpu
            memory = self.smoothed_memory
            error_rate = self.smoothed_error_rate

            if (
                rt > self.max_response_time
                or cpu > self.max_cpu
                or memory > self.max_memory
                or error_rate > float(self.error_block_threshold_pct)
            ):
                # Jika salah satu metrik melebihi batas maksimum, batalkan downscale.
                # Hal ini untuk mencegah penurunan kapasitas saat aplikasi sedang
                # mengalami beban tinggi atau masalah performa.
                # Menghindari fluktuasi metrik yang dapat menyebabkan downscale yang
                # tidak diinginkan.
                self.logger.warning(
                    f"Downscale blocked: RT={rt:.1f} (max {self.max_response_time}), "
                    f"CPU={cpu:.1f} (max {self.max_cpu}), MEM={memory:.1f} "
                    f"(max {self.max_memory}), ERR={error_rate:.2f}"
                )
                return current_replicas, 0

            # Menahan downscale hingga agen RL meminta beberapa kali berturut-turut
            min_confirm = int(self.min_down_confirmations)
            if min_confirm > 0:
                self._pending_down_count = self._pending_down_count + 1
                if self._pending_down_count < min_confirm:
                    self.logger.info(
                        f"Downscale candidate (count "
                        f"{self._pending_down_count}/{min_confirm}) - not applied yet"
                    )
                    return current_replicas, 0
        else:
            # reset pending jika tidak sedang downscale
            self._pending_down_count = 0

        now = time.time()
        # Default tidak ada cooldown untuk upscale, hanya jika ada kasus khusus
        last_up_time = self._last_scale_up_time
        last_down_time = self._last_scale_down_time
        cooldown_up = self.cooldown_up_secs
        cooldown_down = self.cooldown_down_secs

        # Menerapkan wait cooldown sebelum melakukan scaling lagi
        if is_downscale and (now - last_down_time) < cooldown_down:
            self.logger.info(
                f"Downscale suppressed due to cooldown "
                f"({now - last_down_time:.0f}s<{cooldown_down}s)"
            )
            return current_replicas, 0
        if (not is_downscale) and (now - last_up_time) < cooldown_up:
            self.logger.info(
                f"Upscale suppressed due to cooldown "
                f"({now - last_up_time:.0f}s<{cooldown_up}s)"
            )
            return current_replicas, 0

        # Finalisasi target replica setelah batasan diterapkan
        applied_delta = step
        final_replicas = int(
            max(
                self.min_replicas,
                min(self.max_replicas, current_replicas + applied_delta),
            )
        )

        # update last scale timestamps
        if is_downscale:
            self._last_scale_down_time = now
        else:
            self._last_scale_up_time = now

        # debug log
        self.logger.info(
            f"Action {action_percent} -> target {target} | current {current_replicas} "
            f"| apply_delta {applied_delta} -> final {final_replicas}"
        )
        # Mengembalikan nilai akhir dan delta(perubahan) yang diterapkan
        return final_replicas, applied_delta

    def _scale_and_get_metrics(self) -> None:
        self._scale()
        # Increase digunakan untuk menandai apakah ada penambahan replica
        # (berguna untuk delay pada pengambilan metriks)
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

        # Menyimpan metrik terakhir yang diketahui
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
        # Menambahkan variabel untuk menghitung EWMA setiap metrik (berfungsi sebagai
        # parameter check pada saat downscale)
        # contoh 0.3 * 4000 + 0.7 * 3500 = 3650 (perubahan lebih halus)
        alpha = self.ewma_alpha
        self.smoothed_response_time = (
            alpha * self.response_time + (1 - alpha) * self.smoothed_response_time
        )
        self.smoothed_cpu = alpha * self.cpu_usage + (1 - alpha) * self.smoothed_cpu
        self.smoothed_memory = (
            alpha * self.memory_usage + (1 - alpha) * self.smoothed_memory
        )
        self.smoothed_error_rate = (
            alpha * self.error_rate + (1 - alpha) * self.smoothed_error_rate
        )

        # Menghitung persentase response time terhadap max_response_time
        response_time_percentage = (self.response_time / self.max_response_time) * 100.0

        # Beri batasan agar dapat mempercepat konvergensi pada reward dan observasi
        # (RELU-like)
        response_time_percentage = min(response_time_percentage, 1000.0)

        # Menghitung persentase replica saat ini terhadap min-max replica
        current_replica_percentage = (
            (self.replica - self.min_replicas) / self.range_replicas * 100.0
            if self.range_replicas > 0
            else 0.0
        )

        # Calculate deltas(perubahan) (normalized to -100 to +100 range)
        cpu_delta = self.cpu_usage - self.prev_cpu_usage
        memory_delta = self.memory_usage - self.prev_memory_usage
        rt_delta = response_time_percentage - self.prev_response_time

        # Menghitung arah scaling [0=down, 0.5=same, 1=up]
        if hasattr(self, "prev_replica"):
            scaling_direction_raw = self.replica - self.prev_replica
            if scaling_direction_raw > 0:
                scaling_direction = 1.0  # Scaled up
            elif scaling_direction_raw < 0:
                scaling_direction = 0.0  # Scaled down
            else:
                scaling_direction = 0.5  # No change
        else:
            scaling_direction = 0.5  # First observation, tidak ada kondisi sebelumnya

        # Calculate time in current state (normalized 0-1)
        time_in_state = min(
            self.steps_at_current_replica / self.max_steps_tracking, 1.0
        )

        # Menghitung RPS per pod
        # Ini membuat metrik fleksibel di berbagai rentang replika
        rps_per_pod = (self.request_rate / self.replica) if self.replica > 0 else 0.0

        # Menghitung delta (perubahan) RPS (perubahan beban per pod)
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
            "rps_per_pod": rps_per_pod,
            "rps_delta": rps_delta,
            "error_rate": self.error_rate,
        }

    def step(self, action: int) -> tuple[dict[str, float], float, bool, dict]:
        self.last_action = action

        # prev_replica always exists (initialized in __init__), set directly
        self.prev_replica = self.replica

        # Keep action encoding 0..99 but apply percent->safe-delta wrapper
        # to prevent harmful large jumps and flapping.
        applied_delta: int = 0
        final_replicas: int = self.replica
        current_replicas = self.replica
        self.replica_state_old = self.replica_state

        final_replicas, applied_delta = self._percent_action_to_safe_replicas(
            action, current_replicas
        )
        self.replica_state = final_replicas

        self._scale_and_get_metrics()

        # Update berapa step dengan replica saat ini
        if self.replica == self.prev_replica:
            self.steps_at_current_replica += 1
        else:
            self.steps_at_current_replica = 0

        # menyimpan delta yang diterapkan untuk perhitungan reward
        self._last_applied_delta = applied_delta
        reward = self._calculate_reward()

        self.iteration -= 1
        terminated = bool(self.iteration <= 0)

        observation = self._get_observation()

        # Memperbarui metrik sebelumnya untuk perhitungan delta di langkah berikutnya
        rps_per_pod = (self.request_rate / self.replica) if self.replica > 0 else 0.0
        self.prev_cpu_usage = self.cpu_usage
        self.prev_memory_usage = self.memory_usage
        self.prev_response_time = min(
            (self.response_time / self.max_response_time) * 100.0, 100.0
        )
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
            "applied_delta": applied_delta,
            "request_rate": self.request_rate,
            "error_rate": self.error_rate,
            "rps_per_pod": rps_per_pod,
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
        # replica_state is always present; no need for hasattr checks
        self.replica_state_old = self.replica_state
        self.replica_state = self.min_replicas
        self.last_known_metrics = None

        # Reset previous metrics tracking
        self.prev_cpu_usage = 0.0
        self.prev_memory_usage = 0.0
        self.prev_response_time = 0.0
        self.prev_rps_per_pod = 0.0  # NEW: Reset RPS tracking

        # Reset time-in-state tracking
        self.steps_at_current_replica = 0

        # Ensure the logged action reflects the reset (no-op) rather than
        # the previous agent action. _scale_and_get_metrics() calls _scale()
        # which logs the current `replica_state` and `last_action` — at
        # reset we want that `last_action` to be 0.
        self.last_action = 0
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

        # last_action is already set to 0 above
        return self._get_observation()
