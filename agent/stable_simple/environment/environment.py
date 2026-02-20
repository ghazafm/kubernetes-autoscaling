import logging
import time
from typing import Any, Optional

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect
from utils import TransitionLogger, get_metrics, get_replica, wait_for_pods_ready

from database import InfluxDB


def calculate_reward(
    action: int,
    response_time: float,
) -> tuple[float, dict]:
    response_time_penalty = response_time / 100.0
    cost = action / 99.0

    cost_weight = max(0.0, 1.0 - response_time_penalty)

    cost_penalty = cost * cost_weight

    reward = 1.0 - (response_time_penalty + cost_penalty)

    details = {
        "action": action,
        "response_time": response_time,
        "rt_penalty": response_time_penalty,
        "cost_raw": cost,
        "cost_mult": cost_weight,
        "cost_eff": cost_penalty,
        "total_penalty": response_time_penalty + cost_penalty,
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
        self.observation_space = Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -3.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0], dtype=np.float32),
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

        self.observations = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        self.last_reward = 0.0
        self.last_reward_details = {}  # Store reward calculation details

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
        RT_STALE_THRESHOLD = 0.3
        missing_cpu = cpu <= 0.0
        missing_mem = memory <= 0.0
        is_broken = (missing_cpu or missing_mem) and prev_obs[3] >= RT_STALE_THRESHOLD

        if is_broken:
            cpu, memory, response_time = self.estimate_metrics(
                prev_obs=prev_obs,
                action=action,
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
            last_observations=prev_obs,
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
            "cost_penalty": self.last_reward_details.get("cost_eff", 0.0),
            "total_penalty": self.last_reward_details.get("total_penalty", 0.0),
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
        prev_obs,
        action: int,
        cpu: float,
        memory: float,
        response_time: float,
    ) -> tuple[float, float, float]:
        prev_action = prev_obs[0]
        current_action = action / 99.0
        action_change = current_action - prev_action

        prev_cpu = float(prev_obs[1]) * 100.0
        prev_mem = float(prev_obs[2]) * 100.0
        prev_rt = float(prev_obs[3]) * 100.0

        scale_factor = 0.5

        missing_cpu = cpu <= 0.0
        missing_mem = memory <= 0.0
        missing_rt = response_time <= 0.0

        if missing_cpu:
            cpu = prev_cpu * (1 - action_change * scale_factor)
            cpu = max(0.01, cpu)

        if missing_mem:
            memory = prev_mem * (1 - action_change * scale_factor)
            memory = max(0.01, memory)

        if missing_rt:
            response_time = prev_rt * (1 - action_change * scale_factor)
            response_time = float(np.clip(response_time, 0.01, 300.0))

        if self.logger:
            est_parts: list[str] = []
            if missing_cpu:
                est_parts.append(f"cpu={cpu:.1f}%")
            if missing_mem:
                est_parts.append(f"mem={memory:.1f}%")
            if missing_rt:
                est_parts.append(f"rt={response_time:.1f}%")

            self.logger.info(
                "Metrics missing — estimated: %s",
                ", ".join(est_parts) if est_parts else "(none)",
            )

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

        if action > 50 and reward > 0.9:  # noqa: PLR2004
            self.logger.warning(
                f"⚠️ SUSPICIOUS REWARD: action={action}, rt={response_time:.1f}%, "
                f"rt_penalty={details['rt_penalty']:.3f}, "
                f"cost_penalty={details['cost_eff']:.3f}, "
                f"total_penalty={details['total_penalty']:.3f}, "
                f"reward={reward:.3f}, "
            )
        return reward

    def observation(
        self,
        last_observations: np.ndarray,
        action: int,
        response_time: float,
        cpu: float,
        memory: float,
    ) -> np.ndarray:
        action = action / 99.0
        cpu = float(np.clip(cpu / 100.0, 0.0, 1.0))
        memory = float(np.clip(memory / 100.0, 0.0, 1.0))
        response_time = float(np.clip(response_time / 100.0, 0.0, 3.0))
        delta_cpu = cpu - last_observations[1]
        delta_memory = memory - last_observations[2]
        delta_response_time = response_time - last_observations[3]

        return np.array(
            [
                action,
                cpu,
                memory,
                response_time,
                delta_cpu,
                delta_memory,
                delta_response_time,
            ],
            dtype=np.float32,
        )

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
            action = round(self.observations[0] * 99)
            cpu = self.observations[1] * 100.0
            mem = self.observations[2] * 100.0
            rt = self.observations[3] * 100.0
            d_cpu = self.observations[4] * 100.0
            d_mem = self.observations[5] * 100.0
            d_rt = self.observations[6] * 100.0

            self.logger.debug(
                f"Render debug: obs[0]={self.observations[0]:.4f}, "
                f"action={action}, last_reward={self.last_reward:.3f}"
            )

            rt_col = _color(rt, warn=80, crit=100)
            RESET = "\033[0m"

            cpu_bar = _bar(cpu)
            mem_bar = _bar(mem)
            rt_bar = _bar(min(rt, 200.0), width=12)

            RESET = "\033[0m"

            # line 1: Metrics summary
            hdr = "▶ "
            cpu_str = f"CPU {_fmt_pct(cpu)} {_fmt_delta(d_cpu)} {cpu_bar}"
            mem_str = f"MEM {_fmt_pct(mem)} {_fmt_delta(d_mem)} {mem_bar}"
            rt_str = f"{rt_col}RT  {_fmt_pct(rt)} {_fmt_delta(d_rt)} {rt_bar}{RESET}"
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
                    f"cost_raw={d['cost_raw']:6.3f} cost_mult={d['cost_mult']:5.2f} "
                    f"cost_eff={d['cost_eff']:6.3f} | penalty={d['total_penalty']:6.3f}"
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
        prev_obs = self.observation(
            last_observations=self.observations,
            action=action,
            cpu=cpu,
            memory=memory,
            response_time=response_time,
        )

        self.observations = self.observation(
            last_observations=prev_obs,
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
