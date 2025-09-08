import collections
import logging
import random
import time
from typing import Any

import numpy as np
import urllib3
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from kubernetes import client, config

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=f"simulation_environment_{time.strftime('%Y%m%d_%H%M%S')}.log",
    filemode="a",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class K8sAutoscalerEnv(Env):
    MIN_HISTORY_SIZE = 2
    CPU_LIMIT_THRESHOLD = 100.0
    MEMORY_LIMIT_THRESHOLD = 100.0
    CPU_WARNING_THRESHOLD = 95.0
    MEMORY_WARNING_THRESHOLD = 95.0
    MIN_EFFICIENT_CPU = 30.0
    MIN_EFFICIENT_MEMORY = 20.0
    LOW_USAGE_CPU = 10.0
    LOW_USAGE_MEMORY = 10.0
    LARGE_ACTION_THRESHOLD = 20
    MEDIUM_ACTION_THRESHOLD = 10
    SMALL_ACTION_THRESHOLD = 5
    MIN_METRICS_RELIABILITY = 0.9
    HTTP_INTERNAL_SERVER_ERROR = 500

    CRITICAL_REPLICA_THRESHOLD = 2
    HIGH_RESOURCE_THRESHOLD = 70.0

    MIN_HISTORY_FOR_ANALYSIS = 3
    MAX_SCALING_VELOCITY = 10
    MIN_LOAD_TREND_HISTORY = 2
    NEUTRAL_USAGE_THRESHOLD = 50
    OSCILLATION_WINDOW = 4
    MIN_SIGN_CHANGES = 2
    LARGE_REPLICA_SWING = 15
    OSCILLATION_PENALTY_MULTIPLIER = 50
    REPLICA_OSCILLATION_MULTIPLIER = 2

    LARGE_DEPLOYMENT_THRESHOLD = 100
    MEDIUM_DEPLOYMENT_THRESHOLD = 50
    VERY_LARGE_DEPLOYMENT_THRESHOLD = 200
    HIGH_RESOURCE_CPU_THRESHOLD = 1.0
    HIGH_RESOURCE_MEMORY_THRESHOLD = 512
    HIGH_AVAILABILITY_MIN_REPLICAS = 3

    CONTEXT_CRITICAL_CPU_THRESHOLD = 2.0
    CONTEXT_CRITICAL_MEMORY_THRESHOLD = 1024
    CONTEXT_TIGHT_PRECISION_THRESHOLD = 40
    CONTEXT_MODERATE_PRECISION_THRESHOLD = 60

    def __init__(
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
        verbose: bool = False,
        action_step: int = 1,
        timeout: int = 60,
        waste_check_mode: str = "adaptive",
    ):
        config.load_kube_config()
        self.verbose = verbose
        self.cluster = client.AppsV1Api()
        self.api = client.CustomObjectsApi()
        self.core = client.CoreV1Api()
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.timeout = timeout

        action_range = action_step * 2 + 1
        self.action_step = action_step
        if action_range <= 0:
            raise ValueError("action_step must be a positive integer")

        self.action_space = Discrete(action_range)

        self.observation_space = Box(
            low=np.array(
                [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32
            ),
            high=np.array(
                [100, 100, 100, 100, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32
            ),
        )
        """
        Optimized observation space with 15 features:
        0.  cpu_usage - Current CPU utilization percentage
        1.  memory_usage - Current memory utilization percentage
        2.  cpu_target_distance - Distance from CPU target range
        3.  memory_target_distance - Distance from memory target range
        4.  replica_trend - Recent scaling trend
        5.  time_since_last_scale - Time since last scaling action
        6.  resource_pressure_score - Overall resource pressure
        7.  replica_utilization - How close to capacity limits
        8.  oscillation_risk - Risk of oscillating behavior
        9.  scaling_velocity - Rate of recent scaling changes
        10. resource_balance - Balance between CPU and memory usage
        11. efficiency_score - Resource efficiency indicator
        12. stability_score - System stability indicator
        13. resource_saturation - How close to resource limits
        14. load_trend - Recent load change trend
        """

        self.replica_state = (
            min_replicas + 1 if min_replicas < max_replicas - 1 else min_replicas
        )
        self.target_cpu = [min_cpu, max_cpu]
        self.target_memory = [min_memory, max_memory]
        self.max_replicas = max_replicas
        self.min_replicas = min_replicas

        self.cpu_limit_cores, self.memory_limit_mb = (
            self._get_deployment_resource_limits()
        )
        logging.info(
            f"Deployment resource limits - CPU: {self.cpu_limit_cores} cores, "
            f"Memory: {self.memory_limit_mb} MB"
        )

        self.waste_check_threshold = self._calculate_waste_check_threshold(
            waste_check_mode
        )
        logging.info(
            f"Dynamic waste check threshold: {self.waste_check_threshold} "
            f"(mode: {waste_check_mode})"
        )

        self.replica_history = collections.deque(maxlen=5)
        self.action_history = collections.deque(maxlen=10)
        self.last_scale_time = time.time()

        self.initial_iteration = iteration
        self.iteration = iteration

    def _calculate_waste_check_threshold(self, mode: str) -> int:
        """Calculate dynamic waste check threshold based on deployment"""

        threshold_map = {
            "fixed": 2,
            "min_plus_one": max(2, self.min_replicas + 1),
            "adaptive": self._get_adaptive_threshold(),
            "percentage": self._get_percentage_threshold(),
            "context_aware": self._get_context_aware_threshold(),
        }

        return threshold_map.get(mode, max(2, self.min_replicas + 1))

    def _get_adaptive_threshold(self) -> int:
        """Get adaptive threshold based on deployment characteristics"""
        base_threshold = max(2, self.min_replicas + 1)

        if self.max_replicas > self.LARGE_DEPLOYMENT_THRESHOLD:
            base_threshold = max(base_threshold, 5)
        elif self.max_replicas > self.MEDIUM_DEPLOYMENT_THRESHOLD:
            base_threshold = max(base_threshold, 3)

        if (
            hasattr(self, "cpu_limit_cores")
            and hasattr(self, "memory_limit_mb")
            and (
                self.cpu_limit_cores > self.HIGH_RESOURCE_CPU_THRESHOLD
                or self.memory_limit_mb > self.HIGH_RESOURCE_MEMORY_THRESHOLD
            )
        ):
            base_threshold = max(base_threshold, 3)

        return base_threshold

    def _get_percentage_threshold(self) -> int:
        """Get percentage-based threshold"""
        percentage_threshold = max(1, int(self.max_replicas * 0.1))
        return max(percentage_threshold, self.min_replicas + 1, 2)

    def _get_context_aware_threshold(self) -> int:
        """Enhanced context-aware threshold with comprehensive operational analysis"""
        base_threshold = max(2, self.min_replicas + 1)

        ha_factor = self._calculate_ha_factor()

        scale_factor = self._calculate_scale_complexity()

        resource_factor = self._calculate_resource_criticality()

        precision_factor = self._calculate_target_precision()

        total_factor = ha_factor + scale_factor + resource_factor + precision_factor
        enhanced_threshold = int(base_threshold * (1 + total_factor))

        min_threshold = max(2, self.min_replicas + 1)
        max_threshold = min(int(self.max_replicas * 0.25), 15)

        return max(min_threshold, min(enhanced_threshold, max_threshold))

    def _calculate_ha_factor(self) -> float:
        """Calculate high availability complexity factor"""
        if self.min_replicas >= self.HIGH_AVAILABILITY_MIN_REPLICAS:
            return min(0.4, (self.min_replicas - 1) * 0.1)
        return 0.0

    def _calculate_scale_complexity(self) -> float:
        """Calculate deployment scale complexity factor"""
        if self.max_replicas > self.VERY_LARGE_DEPLOYMENT_THRESHOLD:
            return 0.6
        if self.max_replicas > self.LARGE_DEPLOYMENT_THRESHOLD:
            return 0.3
        if self.max_replicas > self.MEDIUM_DEPLOYMENT_THRESHOLD:
            return 0.1
        return 0.0

    def _calculate_resource_criticality(self) -> float:
        """Calculate resource criticality factor"""
        if not (hasattr(self, "cpu_limit_cores") and hasattr(self, "memory_limit_mb")):
            return 0.0

        cpu_critical = self.cpu_limit_cores > self.CONTEXT_CRITICAL_CPU_THRESHOLD
        memory_critical = self.memory_limit_mb > self.CONTEXT_CRITICAL_MEMORY_THRESHOLD

        if cpu_critical and memory_critical:
            return 0.3
        if cpu_critical or memory_critical:
            return 0.15
        return 0.0

    def _calculate_target_precision(self) -> float:
        """Calculate target range precision factor"""
        cpu_range = self.target_cpu[1] - self.target_cpu[0]
        memory_range = self.target_memory[1] - self.target_memory[0]
        avg_precision = (cpu_range + memory_range) / 2

        if avg_precision < self.CONTEXT_TIGHT_PRECISION_THRESHOLD:
            return 0.25
        if avg_precision < self.CONTEXT_MODERATE_PRECISION_THRESHOLD:
            return 0.15
        return 0.0

    def _calculate_replica_trend(self):
        """Calculate recent scaling trend"""
        if len(self.replica_history) >= self.MIN_HISTORY_SIZE:
            recent_change = self.replica_history[-1] - self.replica_history[-2]
            return np.tanh(recent_change / 5)
        return 0.0

    """
    Perlu menghitung resource pressure karena jika hanya melihat CPU dan Memory,
    ada beberapa kasus dengan jumlah memory dan cpu sama namun berat sebelah
    """

    def _calculate_resource_pressure(self, cpu_usage, memory_usage):
        """Calculate overall resource pressure score"""
        cpu_denominator = max(1, 100 - self.target_cpu[1])
        memory_denominator = max(1, 100 - self.target_memory[1])

        cpu_pressure = max(0, (cpu_usage - self.target_cpu[1]) / cpu_denominator)
        memory_pressure = max(
            0, (memory_usage - self.target_memory[1]) / memory_denominator
        )
        return min(1.0, (cpu_pressure + memory_pressure) / 2)

    def _calculate_target_distance(self, usage, target_range):
        """Calculate distance from target range (normalized)"""
        min_target, max_target = target_range
        if min_target <= usage <= max_target:
            return 0.0
        if usage < min_target:
            return (min_target - usage) / min_target
        return (usage - max_target) / (100 - max_target)

    def _calculate_replica_utilization(self):
        """Calculate how close to replica capacity limits"""
        if self.replica_state <= self.min_replicas:
            return 0.0

        return (self.replica_state - self.min_replicas) / (
            self.max_replicas - self.min_replicas
        )

    def _calculate_oscillation_risk(self):
        """Calculate risk of oscillating behavior"""
        if len(self.action_history) < self.MIN_HISTORY_FOR_ANALYSIS:
            return 0.0

        recent_actions = list(self.action_history)[-self.MIN_HISTORY_FOR_ANALYSIS :]
        sign_changes = sum(
            1
            for i in range(1, len(recent_actions))
            if recent_actions[i] * recent_actions[i - 1] < 0
        )
        return min(1.0, sign_changes / 2)

    def _calculate_scaling_velocity(self):
        """Calculate rate of recent scaling changes"""
        if len(self.replica_history) < self.MIN_HISTORY_FOR_ANALYSIS:
            return 0.0

        recent_replicas = list(self.replica_history)[-self.MIN_HISTORY_FOR_ANALYSIS :]
        changes = [
            abs(recent_replicas[i] - recent_replicas[i - 1])
            for i in range(1, len(recent_replicas))
        ]
        avg_change = sum(changes) / len(changes)
        return min(1.0, avg_change / self.MAX_SCALING_VELOCITY)

    def _calculate_resource_balance(self, cpu_usage, memory_usage):
        """Calculate balance between CPU and memory usage"""
        if cpu_usage == 0 and memory_usage == 0:
            return 1.0
        balance = 1 - abs(cpu_usage - memory_usage) / 100
        return max(0.0, balance)

    def _calculate_enhanced_efficiency_score(self, cpu_usage, memory_usage):
        """Calculate enhanced resource efficiency score"""

        cpu_efficiency = 1 - self._calculate_target_distance(cpu_usage, self.target_cpu)
        memory_efficiency = 1 - self._calculate_target_distance(
            memory_usage, self.target_memory
        )

        if cpu_usage < self.LOW_USAGE_CPU and memory_usage < self.LOW_USAGE_MEMORY:
            underutilization_penalty = 0.5
        else:
            underutilization_penalty = 0.0

        return max(
            0.0, (cpu_efficiency + memory_efficiency) / 2 - underutilization_penalty
        )

    def _calculate_stability_score(self):
        """Calculate system stability score"""

        oscillation_penalty = self._calculate_oscillation_risk()

        time_factor = min(1.0, (time.time() - self.last_scale_time) / 60)

        return max(0.0, time_factor - oscillation_penalty)

    def _calculate_resource_saturation(self, cpu_usage, memory_usage):
        """Calculate how close resources are to their limits"""
        cpu_saturation = cpu_usage / 100
        memory_saturation = memory_usage / 100
        return max(cpu_saturation, memory_saturation)

    def _calculate_load_trend(self, cpu_usage, memory_usage):
        """Calculate recent load change trend"""
        if len(self.replica_history) < self.MIN_LOAD_TREND_HISTORY:
            return 0.0

        recent_replica_change = self.replica_history[-1] - self.replica_history[-2]
        if recent_replica_change == 0:
            return 0.0

        avg_usage = (cpu_usage + memory_usage) / 2
        if avg_usage > self.NEUTRAL_USAGE_THRESHOLD:
            return min(1.0, avg_usage / 100)
        return max(
            -1.0,
            (avg_usage - self.NEUTRAL_USAGE_THRESHOLD) / self.NEUTRAL_USAGE_THRESHOLD,
        )

    def _get_observation(self, cpu_usage, memory_usage):
        """Get optimized observation with only relevant dynamic features"""

        self.replica_history.append(self.replica_state)

        cpu_target_distance = self._calculate_target_distance(
            cpu_usage, self.target_cpu
        )
        memory_target_distance = self._calculate_target_distance(
            memory_usage, self.target_memory
        )

        replica_trend = self._calculate_replica_trend()
        time_since_scale = min(1.0, (time.time() - self.last_scale_time) / 300)

        resource_pressure = self._calculate_resource_pressure(cpu_usage, memory_usage)
        replica_utilization = self._calculate_replica_utilization()
        oscillation_risk = self._calculate_oscillation_risk()
        scaling_velocity = self._calculate_scaling_velocity()

        resource_balance = self._calculate_resource_balance(cpu_usage, memory_usage)
        efficiency_score = self._calculate_enhanced_efficiency_score(
            cpu_usage, memory_usage
        )
        stability_score = self._calculate_stability_score()
        resource_saturation = self._calculate_resource_saturation(
            cpu_usage, memory_usage
        )
        load_trend = self._calculate_load_trend(cpu_usage, memory_usage)

        return np.array(
            [
                cpu_usage,
                memory_usage,
                cpu_target_distance,
                memory_target_distance,
                replica_trend,
                time_since_scale,
                resource_pressure,
                replica_utilization,
                oscillation_risk,
                scaling_velocity,
                resource_balance,
                efficiency_score,
                stability_score,
                resource_saturation,
                load_trend,
            ],
            dtype=np.float32,
        )

    def _scale_deployment(self):
        """Scale deployment with retry logic and exponential backoff"""
        max_retries = 3
        base_delay = 1.0
        http_timeout = 30

        for attempt in range(max_retries):
            try:
                logging.info(
                    f"Scaling deployment {self.deployment_name} to "
                    f"{self.replica_state} replicas (attempt {attempt + 1}/"
                    f"{max_retries})"
                )

                self.cluster.patch_namespaced_deployment_scale(
                    name=self.deployment_name,
                    body=client.V1Scale(
                        spec=client.V1ScaleSpec(replicas=int(self.replica_state))
                    ),
                    namespace=self.namespace,
                    _request_timeout=http_timeout,
                )

                logging.info(
                    f"Successfully scaled deployment to {self.replica_state} replicas"
                )
                return

            except client.ApiException as e:
                if (
                    e.status == self.HTTP_INTERNAL_SERVER_ERROR
                    and "etcdserver: request timed out" in str(e)
                ):
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)
                        logging.warning(
                            f"Kubernetes API timeout (attempt {attempt + 1}/"
                            f"{max_retries}). Retrying in {delay} seconds..."
                        )
                        time.sleep(delay)
                        continue

                    logging.error(
                        f"Failed to scale deployment after {max_retries} attempts: {e}"
                    )
                    raise

                logging.error(f"Unexpected API error during scaling: {e}")
                raise
            except Exception as e:
                logging.error(f"Unexpected error during scaling: {e}")
                raise

    def get_metrics(self, replicas):
        counter = 0
        replica = 0
        cpu_usage = []
        memory_usage = []

        while True:
            if counter >= self.timeout:
                logging.warning(
                    f"Timeout reached while fetching metrics after {self.timeout}s"
                )
                break
            counter += 1
            try:
                metric_data = self.api.list_namespaced_custom_object(
                    group="metrics.k8s.io",
                    version="v1beta1",
                    namespace=self.namespace,
                    plural="pods",
                )
            except Exception as e:
                logging.warning(f"Error fetching metrics: {e}")
                time.sleep(1)
                continue

            replica = 0
            cpu_usage = []
            memory_usage = []

            if not metric_data or "items" not in metric_data:
                logging.warning("No metric data found.")
                return 0, 0, 0

            for item in metric_data["items"]:
                if self.deployment_name in item["metadata"]["name"]:
                    cpu_str = item["containers"][0]["usage"]["cpu"]
                    cpu_value = self._parse_cpu_value(cpu_str)

                    cpu_percentage = (cpu_value / self.cpu_limit_cores) * 100

                    memory_str = item["containers"][0]["usage"]["memory"]
                    memory_mb = self._parse_memory_value(memory_str)

                    memory_percentage = (memory_mb / self.memory_limit_mb) * 100
                    cpu_usage.append(cpu_percentage)
                    memory_usage.append(memory_percentage)
                    replica += 1

            if replica == replicas:
                break
            time.sleep(1)

        cpu_usage_mean = np.mean(cpu_usage) if cpu_usage else 0
        memory_usage_mean = np.mean(memory_usage) if memory_usage else 0
        return cpu_usage_mean, memory_usage_mean, replica

    def _handle_scaling_with_fallback(self):
        """Handle scaling operation with fallback for API errors"""
        try:
            self._scale_deployment()
            ready, desired_replicas, ready_replicas = self._wait_for_pods_ready()
            cpu_usage, memory_usage, replica = self.get_metrics(replicas=ready_replicas)

            unschedulable_replicas = max(0, desired_replicas - ready_replicas)
            not_fetchable_replicas = max(0, ready_replicas - replica)
            metrics_reliability = replica / max(1, ready_replicas)

            return (
                cpu_usage,
                memory_usage,
                replica,
                ready_replicas,
                unschedulable_replicas,
                not_fetchable_replicas,
                metrics_reliability,
                ready,
            )

        except client.ApiException as e:
            if (
                e.status == self.HTTP_INTERNAL_SERVER_ERROR
                and "etcdserver: request timed out" in str(e)
            ):
                logging.error(f"Kubernetes API timeout - using fallback state: {e}")
                cpu_usage, memory_usage = 50.0, 50.0
                replica = self.replica_state
                ready_replicas = self.replica_state
                unschedulable_replicas = 0
                not_fetchable_replicas = 0
                metrics_reliability = 0.5
                ready = True

                return (
                    cpu_usage,
                    memory_usage,
                    replica,
                    ready_replicas,
                    unschedulable_replicas,
                    not_fetchable_replicas,
                    metrics_reliability,
                    ready,
                )

            logging.error(f"API error in scaling: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error in scaling: {e}")

            cpu_usage, memory_usage = 50.0, 50.0
            replica = self.replica_state
            ready_replicas = self.replica_state
            unschedulable_replicas = 0
            not_fetchable_replicas = 0
            metrics_reliability = 0.5
            ready = True

            return (
                cpu_usage,
                memory_usage,
                replica,
                ready_replicas,
                unschedulable_replicas,
                not_fetchable_replicas,
                metrics_reliability,
                ready,
            )

    def step(self, action):
        mapped_action = action - self.action_step

        self.replica_state = max(
            self.min_replicas,
            min(self.max_replicas, self.replica_state + mapped_action),
        )

        if mapped_action != 0:
            self.last_scale_time = time.time()
            logging.info(f"Action: {mapped_action}, New replicas: {self.replica_state}")
        self.iteration -= 1

        (
            cpu_usage,
            memory_usage,
            replica,
            ready_replicas,
            unschedulable_replicas,
            not_fetchable_replicas,
            metrics_reliability,
            ready,
        ) = self._handle_scaling_with_fallback()

        reward = self._calculate_total_reward(
            cpu_usage,
            memory_usage,
            ready_replicas,
            mapped_action,
            unschedulable_replicas,
            not_fetchable_replicas,
            metrics_reliability,
            ready,
        )

        if self.iteration <= 0:
            terminated = True
            truncated = False
        else:
            terminated = False
            truncated = False

        observation = self._get_observation(cpu_usage, memory_usage)

        info = {
            "current_replicas": self.replica_state,
            "actual_replicas": replica,
            "action": mapped_action,
            "raw_action": action,
            "state": self.replica_state,
            "iteration": self.iteration,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "target_cpu": self.target_cpu,
            "target_memory": self.target_memory,
            "unschedulable_replicas": unschedulable_replicas,
            "not_fetchable_replicas": not_fetchable_replicas,
            "metrics_reliability": metrics_reliability,
            "cpu_target_distance": observation[2],
            "memory_target_distance": observation[3],
            "replica_trend": observation[4],
            "time_since_last_scale": observation[5],
            "resource_pressure_score": observation[6],
            "replica_utilization": observation[7],
            "oscillation_risk": observation[8],
            "scaling_velocity": observation[9],
            "resource_balance": observation[10],
            "efficiency_score": observation[11],
            "stability_score": observation[12],
            "resource_saturation": observation[13],
            "load_trend": observation[14],
        }

        if self.verbose:
            logging.info(f"Reward: {reward}")
            logging.info(
                f"Step info: {info}, Reward: {reward}, Terminated: {terminated}, "
                f"Truncated: {truncated}",
            )
        logging.info("=" * 55)
        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def _calculate_total_reward(
        self,
        cpu_usage,
        memory_usage,
        ready_replicas,
        mapped_action,
        unschedulable_replicas,
        not_fetchable_replicas,
        metrics_reliability,
        ready,
    ):
        """Calculate the total reward for the current step"""
        reward = 0
        reward_breakdown = {
            "performance_score": 0,
            "resource_efficiency": 0,
            "scaling_appropriateness": 0,
            "cluster_health": 0,
            "stability_bonus": 0,
            "metrics_penalty": 0,
            "critical_override": 0,
        }

        if self._is_impossible_action(mapped_action, ready_replicas):
            reward = -25
            reward_breakdown["critical_override"] = -25
            logging.error(
                f"IMPOSSIBLE ACTION: Action {mapped_action} with {ready_replicas} "
                f"replicas = {reward}"
            )
            return reward

        performance_score = self._calculate_performance_score(
            cpu_usage, memory_usage, ready_replicas
        )
        reward += performance_score
        reward_breakdown["performance_score"] = performance_score

        efficiency_score = self._calculate_efficiency_score(
            cpu_usage, memory_usage, ready_replicas
        )
        reward += efficiency_score
        reward_breakdown["resource_efficiency"] = efficiency_score

        scaling_penalty = self._calculate_scaling_penalty(
            mapped_action, cpu_usage, memory_usage, ready_replicas
        )
        reward -= scaling_penalty
        reward_breakdown["scaling_appropriateness"] = -scaling_penalty

        if unschedulable_replicas > 0 and not ready:
            cluster_penalty = min(
                50, unschedulable_replicas * 5 + (unschedulable_replicas**1.5)
            )
            reward -= cluster_penalty
            reward_breakdown["cluster_health"] = -cluster_penalty
            logging.warning(
                f"CLUSTER PENALTY: {unschedulable_replicas} unschedulable "
                f"pods = -{cluster_penalty:.1f}"
            )
        elif not (
            cpu_usage > self.CPU_LIMIT_THRESHOLD
            or memory_usage > self.MEMORY_LIMIT_THRESHOLD
        ):
            reward += 5
            reward_breakdown["cluster_health"] = 5

        if not_fetchable_replicas > 0:
            metrics_penalty = min(5, not_fetchable_replicas * 1)
            reward -= metrics_penalty
            reward_breakdown["metrics_penalty"] = -metrics_penalty
            logging.warning(
                f"METRICS ISSUE: {not_fetchable_replicas} pods not fetchable "
                f"= -{metrics_penalty:.1f} (reliability: {metrics_reliability:.1%})"
            )
        if metrics_reliability >= self.MIN_METRICS_RELIABILITY:
            reward += 1
            reward_breakdown["metrics_penalty"] = 1

        stability_bonus = self._calculate_stability_bonus(mapped_action)
        reward += stability_bonus
        reward_breakdown["stability_bonus"] = stability_bonus

        critical_cpu = cpu_usage > self.CPU_LIMIT_THRESHOLD
        critical_memory = memory_usage > self.MEMORY_LIMIT_THRESHOLD

        if (critical_cpu or critical_memory) and reward > 0:
            critical_override = -reward
            reward = min(reward, -10)
            reward_breakdown["critical_override"] = critical_override
            logging.error(
                f"CRITICAL ZONE OVERRIDE: Reward capped at {reward:.1f} "
                f"(was {reward - critical_override:.1f}) - "
                f"CPU: {cpu_usage:.1f}%, Memory: {memory_usage:.1f}%"
            )

        return reward

    def _calculate_performance_score(self, cpu_usage, memory_usage, ready_replicas):
        """Calculate performance-based reward with incremental threshold penalties"""
        score = 0

        cpu_penalty = self._calculate_incremental_penalty(
            cpu_usage,
            [
                (self.target_cpu[1], 1.0),
                (self.CPU_WARNING_THRESHOLD, 5.0),
                (self.CPU_LIMIT_THRESHOLD, 10.0),
            ],
            "CPU",
        )
        score -= cpu_penalty

        memory_penalty = self._calculate_incremental_penalty(
            memory_usage,
            [
                (self.target_memory[1], 1.0),
                (self.MEMORY_WARNING_THRESHOLD, 5.0),
                (self.MEMORY_LIMIT_THRESHOLD, 10.0),
            ],
            "Memory",
        )
        score -= memory_penalty

        cpu_optimal = self.target_cpu[0] <= cpu_usage <= self.target_cpu[1]
        memory_optimal = self.target_memory[0] <= memory_usage <= self.target_memory[1]

        critical_threshold_check = (
            cpu_usage < self.CPU_LIMIT_THRESHOLD
            and memory_usage < self.MEMORY_LIMIT_THRESHOLD
        )
        if critical_threshold_check:
            if cpu_optimal and memory_optimal:
                score += 80
            elif cpu_optimal or memory_optimal:
                score += 40

            if cpu_usage > 0 and memory_usage > 0:
                resource_balance = 1 - abs(cpu_usage - memory_usage) / 100
                balance_bonus = resource_balance * 15
                score += balance_bonus

        return score

    def _calculate_incremental_penalty(self, usage, thresholds, resource_type):
        """Calculate incremental penalties based on threshold violations"""
        total_penalty = 0

        for threshold, penalty_rate in thresholds:
            if usage > threshold:
                range_violation = min(usage - threshold, 100 - threshold)
                range_penalty = range_violation * penalty_rate
                total_penalty += range_penalty

                logging.warning(
                    f"{resource_type} VIOLATION: {usage:.2f}% > {threshold}% "
                    f"= -{range_penalty:.2f} (rate: {penalty_rate}x)"
                )
            else:
                break

        return total_penalty

    def _calculate_efficiency_score(self, cpu_usage, memory_usage, ready_replicas):
        """Calculate resource efficiency score"""
        score = 0

        if ready_replicas > self.min_replicas:
            if cpu_usage < self.LOW_USAGE_CPU:
                waste_factor = (self.LOW_USAGE_CPU - cpu_usage) / self.LOW_USAGE_CPU
                score -= ready_replicas * waste_factor * 2

            if memory_usage < self.LOW_USAGE_MEMORY:
                waste_factor = (
                    self.LOW_USAGE_MEMORY - memory_usage
                ) / self.LOW_USAGE_MEMORY
                score -= ready_replicas * waste_factor * 1.5

        if (
            ready_replicas == 1
            and cpu_usage > self.MIN_EFFICIENT_CPU
            and memory_usage > self.MIN_EFFICIENT_MEMORY
        ):
            score += 15

        return score

    def _calculate_scaling_penalty(
        self, mapped_action, cpu_usage, memory_usage, ready_replicas
    ):
        """Penalize inappropriate scaling decisions with stricter rules"""
        penalty = 0

        if mapped_action < 0 and ready_replicas <= self.min_replicas:
            penalty += 20
            logging.error(
                f"CRITICAL PENALTY: Trying to scale below minimum replicas "
                f"(current: {ready_replicas}, min: {self.min_replicas}) = -20"
            )

        if mapped_action < 0 and abs(mapped_action) > ready_replicas:
            overshoot_penalty = abs(mapped_action) - ready_replicas
            penalty += overshoot_penalty * 5
            logging.error(
                f"OVERSHOOT PENALTY: Trying to scale down by {abs(mapped_action)} "
                f"when only {ready_replicas} replicas exist = -{overshoot_penalty * 50}"
            )

        if mapped_action < 0:
            if ready_replicas <= self.CRITICAL_REPLICA_THRESHOLD and (
                cpu_usage > self.HIGH_RESOURCE_THRESHOLD
                or memory_usage > self.HIGH_RESOURCE_THRESHOLD
            ):
                penalty += 30
                logging.error(
                    f"CRITICAL: Scaling down at minimum replicas with high usage "
                    f"(CPU={cpu_usage:.1f}%, MEM={memory_usage:.1f}%) = -30"
                )

            critical_zone = (
                cpu_usage > self.CPU_LIMIT_THRESHOLD
                or memory_usage > self.MEMORY_LIMIT_THRESHOLD
            )
            warning_zone = (
                cpu_usage > self.CPU_WARNING_THRESHOLD
                or memory_usage > self.MEMORY_WARNING_THRESHOLD
            )

            if critical_zone:
                penalty += abs(mapped_action) * 5
                logging.warning(
                    f"SEVERE INAPPROPRIATE SCALE DOWN: CPU={cpu_usage:.1f}%, "
                    f"MEM={memory_usage:.1f}% - CRITICAL ZONE"
                )
            elif warning_zone:
                penalty += abs(mapped_action) * 3
                logging.warning(
                    f"HIGH INAPPROPRIATE SCALE DOWN: CPU={cpu_usage:.1f}%, "
                    f"MEM={memory_usage:.1f}% - WARNING ZONE"
                )
            elif cpu_usage > self.target_cpu[1] or memory_usage > self.target_memory[1]:
                penalty += abs(mapped_action) * 2
                logging.warning(
                    f"MODERATE INAPPROPRIATE SCALE DOWN: CPU={cpu_usage:.1f}%, "
                    f"MEM={memory_usage:.1f}% - ABOVE TARGET"
                )

        elif (
            mapped_action > 0
            and cpu_usage < self.LOW_USAGE_CPU
            and memory_usage < self.LOW_USAGE_MEMORY
            and ready_replicas > self.waste_check_threshold
        ):
            underutilization_factor = (
                self.LOW_USAGE_CPU - cpu_usage
            ) / self.LOW_USAGE_CPU
            penalty += mapped_action * (5 + underutilization_factor * 10)
            logging.warning(
                f"INAPPROPRIATE SCALE UP: CPU={cpu_usage:.1f}%, "
                f"MEM={memory_usage:.1f}% - RESOURCES UNDERUTILIZED "
                f"(threshold: {self.waste_check_threshold})"
            )

        if abs(mapped_action) > self.LARGE_ACTION_THRESHOLD:
            penalty += abs(mapped_action) * 0.5
            logging.warning(f"LARGE ACTION PENALTY: {abs(mapped_action)} replicas")
        elif abs(mapped_action) > self.MEDIUM_ACTION_THRESHOLD:
            penalty += abs(mapped_action) * 0.25

        return penalty

    def _calculate_stability_bonus(self, mapped_action):
        """Reward stable, gradual scaling decisions and penalize oscillations"""
        bonus = 0

        self.action_history.append(mapped_action)

        if mapped_action == 0:
            bonus += 25
        elif 0 < abs(mapped_action) <= self.SMALL_ACTION_THRESHOLD:
            bonus += 15
        elif (
            self.SMALL_ACTION_THRESHOLD
            < abs(mapped_action)
            <= self.MEDIUM_ACTION_THRESHOLD
        ):
            bonus += 8

        oscillation_penalty = self._detect_oscillation()
        bonus -= oscillation_penalty

        if self._is_in_good_state() and mapped_action == 0:
            bonus += 30

        return bonus

    def _detect_oscillation(self):
        """Detect oscillating behavior in recent actions"""
        if len(self.action_history) < self.OSCILLATION_WINDOW:
            return 0

        recent_actions = list(self.action_history)[-self.OSCILLATION_WINDOW :]

        sign_changes = 0
        for i in range(1, len(recent_actions)):
            if recent_actions[i] * recent_actions[i - 1] < 0:
                sign_changes += 1

        if sign_changes >= self.MIN_SIGN_CHANGES:
            penalty = self.OSCILLATION_PENALTY_MULTIPLIER * sign_changes
            logging.warning(
                f"OSCILLATION DETECTED: {sign_changes} sign changes = -{penalty}"
            )
            return penalty

        if len(self.replica_history) >= self.OSCILLATION_WINDOW:
            recent_replicas = list(self.replica_history)[-self.OSCILLATION_WINDOW :]
            replica_range = max(recent_replicas) - min(recent_replicas)

            if replica_range > self.LARGE_REPLICA_SWING:
                penalty = replica_range * self.REPLICA_OSCILLATION_MULTIPLIER
                logging.warning(
                    f"REPLICA OSCILLATION: Range {replica_range} = -{penalty}"
                )
                return penalty

        return 0

    def _is_in_good_state(self):
        """Check if current state is in good operational range"""

        return len(self.replica_history) > 0

    def _is_impossible_action(self, mapped_action, ready_replicas):
        """Check if the action is impossible given current state"""

        if mapped_action < 0:
            new_replicas = ready_replicas + mapped_action
            if new_replicas < self.min_replicas:
                return True

        if mapped_action > 0:
            new_replicas = ready_replicas + mapped_action
            if new_replicas > self.max_replicas:
                return True

        return False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        logging.info("Resetting the environment")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.iteration = self.initial_iteration

        self.replica_history.clear()
        self.action_history.clear()
        self.last_scale_time = time.time()

        (
            cpu_usage,
            memory_usage,
            replica,
            _,
            unschedulable_replicas,
            not_fetchable_replicas,
            metrics_reliability,
            _,
        ) = self._handle_scaling_with_fallback()

        observation = self._get_observation(cpu_usage, memory_usage)

        info = {
            "current_replicas": self.replica_state,
            "actual_replicas": replica,
            "action": 0,
            "raw_action": -1,
            "state": self.replica_state,
            "iteration": self.iteration,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "target_cpu": self.target_cpu,
            "target_memory": self.target_memory,
            "unschedulable_replicas": unschedulable_replicas,
            "not_fetchable_replicas": not_fetchable_replicas,
            "metrics_reliability": metrics_reliability,
            "cpu_target_distance": observation[2],
            "memory_target_distance": observation[3],
            "replica_trend": observation[4],
            "time_since_last_scale": observation[5],
            "resource_pressure_score": observation[6],
            "replica_utilization": observation[7],
            "oscillation_risk": observation[8],
            "scaling_velocity": observation[9],
            "resource_balance": observation[10],
            "efficiency_score": observation[11],
            "stability_score": observation[12],
            "resource_saturation": observation[13],
            "load_trend": observation[14],
        }

        return (
            observation,
            info,
        )

    def _wait_for_pods_ready(self):
        """Wait for pods to be ready after scaling operation."""
        start_time = time.time()
        desired_replicas = 0
        ready_replicas = 0
        while time.time() - start_time < self.timeout:
            try:
                deployment = self.cluster.read_namespaced_deployment(
                    name=self.deployment_name, namespace=self.namespace
                )
                status = getattr(deployment, "status", None)
                spec = getattr(deployment, "spec", None)

                if status is not None:
                    ready_replicas = getattr(status, "ready_replicas", 0) or 0
                else:
                    ready_replicas = 0

                if spec is not None:
                    desired_replicas = getattr(spec, "replicas", 0) or 0
                else:
                    desired_replicas = 0

                if ready_replicas == desired_replicas > 0:
                    return True, desired_replicas, ready_replicas

                time.sleep(5)

            except Exception as e:
                logging.error(f"Error checking pod readiness: {e}")
                time.sleep(5)

        logging.warning(f"Timeout waiting for pods to be ready after {self.timeout}s")

        return False, desired_replicas, ready_replicas

    def _get_deployment_resource_limits(self):
        """
        Get CPU and memory resource limits from the deployment specification.
        Returns: tuple of (cpu_limit_cores, memory_limit_mb)
        """
        default_cpu = 0.2
        default_memory = 128

        try:
            deployment = self.cluster.read_namespaced_deployment(
                name=self.deployment_name, namespace=self.namespace
            )

            spec = getattr(deployment, "spec", None)
            template = getattr(spec, "template", None) if spec else None
            template_spec = getattr(template, "spec", None) if template else None
            containers = (
                getattr(template_spec, "containers", None) if template_spec else None
            )

            if not containers or len(containers) == 0:
                logging.warning("No containers found in deployment, using defaults")
                return default_cpu, default_memory

            container = containers[0]
            resources = getattr(container, "resources", None)
            limits = getattr(resources, "limits", None) if resources else None

            if not limits:
                logging.warning("No resource limits defined, using defaults")
                return default_cpu, default_memory

            cpu_limit_str = limits.get("cpu", "200m")
            memory_limit_str = limits.get("memory", "128Mi")

            cpu_limit_cores = self._parse_cpu_value(cpu_limit_str)
            memory_limit_mb = self._parse_memory_value(memory_limit_str)

            return cpu_limit_cores, memory_limit_mb

        except Exception as e:
            logging.warning(f"Could not get deployment resource limits: {e}")
            return default_cpu, default_memory

    def _parse_cpu_value(self, cpu_str):
        """Parse CPU value from kubernetes format to cores (float)"""
        try:
            if cpu_str.endswith("m"):
                return float(cpu_str[:-1]) / 1000
            if cpu_str.endswith("n"):
                return float(cpu_str[:-1]) / 1000000000
            if cpu_str.endswith("u"):
                return float(cpu_str[:-1]) / 1000000
            return float(cpu_str)
        except (ValueError, IndexError) as e:
            logging.warning(f"Could not parse CPU value '{cpu_str}': {e}")
            return 0.0

    def _parse_memory_value(self, memory_str):
        """Parse memory value from kubernetes format to MB (float)"""
        try:
            if memory_str.endswith("Ki"):
                return float(memory_str[:-2]) / 1024
            if memory_str.endswith("Mi"):
                return float(memory_str[:-2])
            if memory_str.endswith("Gi"):
                return float(memory_str[:-2]) * 1024
            if memory_str.endswith("Ti"):
                return float(memory_str[:-2]) * 1024 * 1024
            return float(memory_str) / (1024 * 1024)
        except (ValueError, IndexError) as e:
            logging.warning(f"Could not parse memory value '{memory_str}': {e}")
            return 0.0
