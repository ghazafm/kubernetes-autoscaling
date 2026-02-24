import logging
import time
from typing import Any, Optional

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect
from utils import Fuzzy, TransitionLogger, get_metrics, get_replica, wait_for_pods_ready

from database import InfluxDB


def calculate_reward(
    action: int,
    response_time: float,
) -> tuple[float, dict]:
    rt_clipped = max(0.0, min(response_time, 100.0))
    rt_penalty = rt_clipped / 100.0
    cost = action / 99.0

    reward = (1.0 - rt_penalty) * (1.0 - cost)

    details = {
        "action": action,
        "response_time": response_time,
        "response_time_clipped": rt_clipped,
        "rt_penalty": rt_penalty,
        "cost": cost,
        "reward": reward,
    }
    return reward, details


class KubernetesEnv(Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(  # noqa: PLR0913
        self,
        namespace: str,
        deployment_name: str,
        min_replicas: int,
        max_replicas: int,
        max_response_time: float,
        iteration: int,
        timeout: int,
        wait_time: int,
        prometheus_url: str,
        metrics_interval: int,
        metrics_quantile: float,
        max_scaling_retries: int,
        logger: Optional[logging.Logger],
        influxdb: Optional[InfluxDB] = None,
        metrics_endpoints_method: list[tuple[str, str]] = (
            ("/cpu", "GET"),
            ("/memory", "GET"),
        ),
        render_mode: Optional[str] = None,
        csv_log_dir: Optional[str] = None,
        csv_log_prefix: str = "data",
        mode: str = "dev",
    ):
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Invalid render_mode '{render_mode}'. "
                f"Supported modes: {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode
        self.mode = mode

        config.load_kube_config()
        self.api = client.AppsV1Api()
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.prometheus = PrometheusConnect(
            url=prometheus_url,
            disable_ssl=True,
        )
        self.timeout = timeout
        self.wait_time = wait_time
        self.metrics_interval = metrics_interval
        self.metrics_quantile = metrics_quantile
        self.metrics_endpoints_method = metrics_endpoints_method
        self.logger = logger
        self.action_space = Discrete(100)
        # Fuzzy state: 4 metrics x 3 membership labels = 12 floats in [0, 1]
        self.fuzzy = Fuzzy(logger=logger, max_replicas=max_replicas)
        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32,
        )

        self.iteration = iteration
        self.iteration_init = iteration
        self.min_replicas: int = min_replicas
        self.max_replicas: int = max_replicas
        self.range_replicas: int = max(1, self.max_replicas - self.min_replicas)
        self.max_response_time: float = max_response_time
        self.influxdb = influxdb
        self.max_scaling_retries = max_scaling_retries

        self.observations = np.zeros(12, dtype=np.float32)
        self.last_reward = 0.0
        self.last_reward_details = {}  # Store reward calculation details
        # Cache raw metric values (0-100 scale) for estimate_metrics fallback
        self._last_cpu: float = 0.0
        self._last_memory: float = 0.0
        self._last_response_time: float = 0.0
        self._last_action: int = 0

        self.csv_logger = TransitionLogger(
            log_dir=csv_log_dir if csv_log_dir else "data",
            prefix=csv_log_prefix,
            enabled=csv_log_dir is not None,
        )
        if self.csv_logger.enabled and self.logger:
            self.logger.info(f"CSV logging enabled: {self.csv_logger.get_filepath()}")

        self.logger.info(f"max_replicas: {self.max_replicas}")
        self.logger.info(f"min_replicas: {self.min_replicas}")

    def step(self, action: int):
        self.iteration -= 1
        replica = int(action * self.range_replicas // 99 + self.min_replicas)
        replica = min(replica, self.max_replicas)

        prev_obs = self.observations.copy()

        cpu, memory, response_time = self.scale(replica)

        # kenapa digunakan response time sebelumnya?
        # karena ada kemungkinan response time sekarang
        # juga bernilai 0.0 karena pod belum siap atau tidak ada request masuk
        missing_cpu = cpu <= 0.0
        missing_mem = memory <= 0.0
        is_broken = (missing_cpu or missing_mem) and self._last_response_time > 0.0

        if is_broken:
            cpu, memory, response_time = self.estimate_metrics(
                cpu=cpu,
                memory=memory,
                response_time=response_time,
            )

        reward = self.calculate_reward(
            action=action,
            response_time=response_time,
        )
        self.last_reward = reward

        self.observations = self.observation(
            action=action,
            response_time=response_time,
            cpu=cpu,
            memory=memory,
        )

        if self.iteration <= 0:
            terminated = False
            truncated = True
        else:
            terminated = False
            truncated = False

        info = {
            "cpu": cpu,
            "memory": memory,
            "response_time": response_time,
            "replicas": replica,
            "action": action,
            "reward": reward,
            "rt_penalty": self.last_reward_details.get("rt_penalty", 0.0),
            # Keep legacy log keys, but map them to new reward detail fields.
            "cost_penalty": self.last_reward_details.get("cost", 0.0),
            "total_penalty": 1.0 - self.last_reward_details.get("reward", reward),
            "iteration": self.iteration,
        }

        self.csv_logger.log_transition(
            obs=prev_obs,
            action=action,
            reward=reward,
            next_obs=self.observations,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

        if self.influxdb:
            self.influxdb.write_point(
                measurement="autoscaling_metrics",
                tags={
                    "namespace": self.namespace,
                    "deployment": self.deployment_name,
                },
                fields={**info},
            )

        self.render()

        return self.observations, reward, terminated, truncated, info

    def scale(self, replica: int) -> None:
        attempt = 0
        while attempt < self.max_scaling_retries:
            attempt += 1
            delay = min(0.5 * (2 ** (attempt - 1)), 10)
            try:
                self.api.patch_namespaced_deployment_scale(
                    name=self.deployment_name,
                    namespace=self.namespace,
                    body={"spec": {"replicas": replica}},
                )
            except Exception as e:
                self.logger.warning(f"Scale attempt {attempt} failed: {e}")
                if attempt >= self.max_scaling_retries:
                    self.logger.error("Max retries reached, continuing with metrics")
                else:
                    time.sleep(delay)

            ready, _, _, _ = wait_for_pods_ready(
                prometheus=self.prometheus,
                deployment_name=self.deployment_name,
                namespace=self.namespace,
                replica=replica,
                timeout=self.timeout,
                wait_time=self.wait_time,
                logger=self.logger,
            )
            if ready:
                time.sleep(delay)
                break

        cpu, memory, response_time = get_metrics(
            prometheus=self.prometheus,
            namespace=self.namespace,
            deployment_name=self.deployment_name,
            interval=self.metrics_interval,
            max_response_time=self.max_response_time,
            quantile=self.metrics_quantile,
            endpoints_method=self.metrics_endpoints_method,
        )
        return cpu, memory, response_time

    def estimate_metrics(
        self,
        cpu: float,
        memory: float,
        response_time: float,
    ) -> tuple[float, float, float]:
        # Reuse cached raw values (0-100 scale) when current readings are missing
        if cpu <= 0.0:
            cpu = self._last_cpu
        if memory <= 0.0:
            memory = self._last_memory
        if response_time <= 0.0:
            response_time = self._last_response_time
        return cpu, memory, response_time

    def calculate_reward(
        self,
        action: int,
        response_time: float,
    ) -> float:
        reward, details = calculate_reward(
            action=action,
            response_time=response_time,
        )

        self.last_reward_details = details
        return reward

    def observation(
        self,
        action: int,
        response_time: float,
        cpu: float,
        memory: float,
    ) -> np.ndarray:
        # Cache raw values for estimate_metrics fallback
        self._last_cpu = cpu
        self._last_memory = memory
        self._last_response_time = response_time
        self._last_action = action

        fuzzy_state = self.fuzzy.fuzzify(
            {
                "cpu_usage": float(np.clip(cpu, 0.0, 100.0)),
                "memory_usage": float(np.clip(memory, 0.0, 100.0)),
                "response_time": float(np.clip(response_time, 0.0, 100.0)),
                "last_action": float(action),
            }
        )

        # Flatten in a fixed order: cpu_usage, memory_usage, response_time, last_action
        # Each metric has 3 labels (low, medium, high) -> 12 values total
        labels = ["low", "medium", "high"]
        metrics = ["cpu_usage", "memory_usage", "response_time", "last_action"]
        flat = [fuzzy_state[m][l] for m in metrics for l in labels]

        return np.array(flat, dtype=np.float32)

    def render(self) -> None:
        def _color(v: float, warn: float, crit: float, reverse: bool = False) -> str:
            GREEN, YELLOW, RED = "\033[32m", "\033[33m", "\033[31m"

            if reverse:
                if v >= crit:
                    return GREEN
                if v >= warn:
                    return YELLOW
                return RED
            if v >= crit:
                return RED
            if v >= warn:
                return YELLOW
            return GREEN

        def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
            return max(lo, min(hi, v))

        def _bar(pct: float, width: int = 12) -> str:
            pct = _clamp(pct)
            filled = round(pct / 100 * width)
            return "█" * filled + "░" * (width - filled)

        def _fmt_pct(v: float) -> str:
            return f"{v:6.2f}%"

        def _fmt_delta(v: float) -> str:
            sign = "+" if v >= 0 else ""
            val_str = f"{sign}{v:5.1f}%"

            RED, GREEN, RESET = "\033[31m", "\033[32m", "\033[0m"
            GREY = "\033[90m"

            if v > 5.0:  # noqa: PLR2004
                return f"{RED}{val_str}{RESET}"
            if v < -5.0:  # noqa: PLR2004
                return f"{GREEN}{val_str}{RESET}"
            return f"{GREY}{val_str}{RESET}"

        if self.render_mode == "human":
            # Use cached raw values (0-100 scale) for display
            cpu = self._last_cpu
            mem = self._last_memory
            rt = self._last_response_time
            action = self._last_action

            self.logger.debug(
                f"Render debug: action={action}, last_reward={self.last_reward:.3f}"
            )

            rt_col = _color(rt, warn=80, crit=100)
            RESET = "\033[0m"

            cpu_bar = _bar(cpu)
            mem_bar = _bar(mem)
            rt_bar = _bar(min(rt, 200.0), width=12)

            # line 1: Metrics summary
            hdr = "▶ "
            cpu_str = f"CPU {_fmt_pct(cpu)} {cpu_bar}"
            mem_str = f"MEM {_fmt_pct(mem)} {mem_bar}"
            rt_str = f"{rt_col}RT  {_fmt_pct(rt)} {rt_bar}{RESET}"
            act_str = f"ACT {action:3d}"
            reward_str = f"RWD {self.last_reward:+6.3f}"

            self.logger.info(
                f"{' ' * len(hdr)}| {cpu_str} | {mem_str} | {rt_str} | "
                f"{act_str} | {reward_str} |"
            )

            # line 2: Reward breakdown (only if we have details)
            if self.last_reward_details:
                d = self.last_reward_details
                details_str = (
                    f"{hdr}| rt_penalty={d['rt_penalty']:6.3f} | "
                    f"cost={d['cost']:6.3f} | "
                    f"penalty={1.0 - d['reward']:6.3f}"
                )
                self.logger.info(details_str)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)
        self.iteration = self.iteration_init
        if self.mode in {"prod", "test", "production"}:
            _, replica = get_replica(
                prometheus=self.prometheus,
                namespace=self.namespace,
                deployment_name=self.deployment_name,
                wait_time=1,
            )
            action = round((replica - self.min_replicas) * 99 / self.range_replicas)
            action = int(np.clip(action, 0, 99))
        elif self.mode in {"dev", "development"}:
            action = self.action_space.sample()
        else:
            raise ValueError(f"Invalid mode '{self.mode}'")
        replica = int(action * self.range_replicas // 99 + self.min_replicas)
        replica = min(replica, self.max_replicas)

        cpu, memory, response_time = self.scale(replica)

        self.observations = self.observation(
            action=action,
            cpu=cpu,
            memory=memory,
            response_time=response_time,
        )

        info = {
            "cpu": cpu,
            "memory": memory,
            "response_time": response_time,
            "replicas": replica,
            "action": action,
        }

        self.csv_logger.on_reset(self.observations, info)
        if self.influxdb:
            self.influxdb.write_point(
                measurement="autoscaling_metrics",
                tags={
                    "namespace": self.namespace,
                    "deployment": self.deployment_name,
                },
                fields={**info},
            )

        self.render()

        return self.observations, info
