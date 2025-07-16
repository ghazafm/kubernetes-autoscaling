import collections
import logging
import random
import time

import numpy as np
import urllib3
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from kubernetes import client, config
from pyparsing import Any

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

    CRITICAL_REPLICA_THRESHOLD = 2
    HIGH_RESOURCE_THRESHOLD = 70.0

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
            low=np.array([0, 0, 0, 0, 1, 0, -1, 0, 0], dtype=np.float32),
            high=np.array([100, 100, 100, 100, 500, 100, 1, 1, 1], dtype=np.float32),
        )
        """
        cpu_usage,
        memory_usage,
        cpu_available,
        memory_available,
        current_replicas,
        unschedulable_replicas,
        replica_trend,
        time_since_last_scale,
        resource_pressure_score
        """

        self.replica_state = (
            min_replicas + 1 if min_replicas < max_replicas - 1 else min_replicas
        )
        self.target_cpu = [min_cpu, max_cpu]
        self.target_memory = [min_memory, max_memory]
        self.max_replicas = max_replicas
        self.min_replicas = min_replicas
        self.node_cpu_total, self.node_memory_total = self._get_node_capacity()

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
        self.last_action = 0

        self.init_iteration = iteration
        self.iteration = iteration

    def _calculate_waste_check_threshold(self, mode: str) -> int:
        """Calculate dynamic waste check threshold based on deployment"""

        # Mode-specific threshold calculation
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

        # Adjust for large deployments
        if self.max_replicas > self.LARGE_DEPLOYMENT_THRESHOLD:
            base_threshold = max(base_threshold, 5)
        elif self.max_replicas > self.MEDIUM_DEPLOYMENT_THRESHOLD:
            base_threshold = max(base_threshold, 3)

        # Adjust for resource-intensive applications
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
        """Get context-aware threshold"""
        base_threshold = max(2, self.min_replicas + 1)

        # High availability scenario
        if self.min_replicas >= self.HIGH_AVAILABILITY_MIN_REPLICAS:
            base_threshold = max(base_threshold, self.min_replicas + 2)

        # Very large deployments
        if self.max_replicas > self.VERY_LARGE_DEPLOYMENT_THRESHOLD:
            base_threshold = max(base_threshold, 8)
        elif self.max_replicas > self.LARGE_DEPLOYMENT_THRESHOLD:
            base_threshold = max(base_threshold, 5)

        return base_threshold

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

    def _get_observation(
        self,
        cpu_usage,
        memory_usage,
        cpu_available,
        memory_available,
        unschedulable_replicas,
    ):
        """Get enhanced observation with temporal and contextual features"""

        self.replica_history.append(self.replica_state)

        replica_trend = self._calculate_replica_trend()

        time_since_scale = min(1.0, (time.time() - self.last_scale_time) / 300)

        resource_pressure = self._calculate_resource_pressure(cpu_usage, memory_usage)

        return np.array(
            [
                cpu_usage,
                memory_usage,
                cpu_available,
                memory_available,
                self.replica_state,
                unschedulable_replicas,
                replica_trend,
                time_since_scale,
                resource_pressure,
            ],
            dtype=np.float32,
        )

    def _scale_deployment(self):
        logging.info(
            f"Scaling deployment {self.deployment_name} to {self.replica_state} "
            "replicas"
        )
        self.cluster.patch_namespaced_deployment_scale(
            name=self.deployment_name,
            body=client.V1Scale(
                spec=client.V1ScaleSpec(replicas=int(self.replica_state))
            ),
            namespace=self.namespace,
        )

    def get_metrics(self, replicas):
        counter = 0
        replica = 0
        cpu_usage = []
        memory_usage = []
        logging.info(f"Fetching metrics for {replicas} replicas.....")

        while True:
            if counter >= self.timeout:
                logging.warning(
                    f"Timeout reached while fetching metrics after {self.timeout}s"
                )
                logging.info(f"Fetched replica {replica}/{replicas}")
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
        logging.info(
            f"CPU usage: {cpu_usage_mean:.2f}% (limit: {self.cpu_limit_cores} cores)"
        )
        logging.info(
            f"Memory usage: {memory_usage_mean:.2f}% (limit: {self.memory_limit_mb} MB)"
        )
        return cpu_usage_mean, memory_usage_mean, replica

    def step(self, action):
        mapped_action = action - self.action_step

        self.replica_state = max(
            self.min_replicas,
            min(self.max_replicas, self.replica_state + mapped_action),
        )

        if mapped_action != 0:
            self.last_scale_time = time.time()
        logging.info(
            f"Action taken: {mapped_action} (raw: {action}), New replica state: "
            f"{self.replica_state}"
        )
        self.iteration -= 1
        self._scale_deployment()
        ready, desired_replicas, ready_replicas = self._wait_for_pods_ready()
        cpu_usage, memory_usage, replica = self.get_metrics(replicas=ready_replicas)

        cpu_available, memory_available = self.get_node_resource_usage()

        unschedulable_replicas = max(0, desired_replicas - ready_replicas)
        not_fetchable_replicas = max(0, ready_replicas - replica)

        metrics_reliability = replica / max(1, ready_replicas)

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

        observation = self._get_observation(
            cpu_usage,
            memory_usage,
            cpu_available,
            memory_available,
            unschedulable_replicas,
        )

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
            "cpu_available": cpu_available,
            "memory_available": memory_available,
            "unschedulable_replicas": unschedulable_replicas,
            "not_fetchable_replicas": not_fetchable_replicas,
            "metrics_reliability": metrics_reliability,
            "replica_trend": observation[6],
            "time_since_last_scale": observation[7],
            "resource_pressure_score": observation[8],
        }

        if self.verbose:
            logging.debug(
                f"Step info: {info}, Reward: {reward}, Terminated: {terminated}, "
                f"Truncated: {truncated}",
            )
            logging.info(f"Replicas: {replica}")
        logging.info(f"Reward: {reward}")
        logging.info("======================================================")
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
            reward = -500
            reward_breakdown["critical_override"] = -500
            logging.error(
                f"IMPOSSIBLE ACTION: Action {mapped_action} with {ready_replicas} "
                f"replicas = {reward}"
            )
            logging.info(f"Reward breakdown: {reward_breakdown}")
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
        else:
            critical_zone = (
                cpu_usage > self.CPU_LIMIT_THRESHOLD
                or memory_usage > self.MEMORY_LIMIT_THRESHOLD
            )
            if not critical_zone:
                reward += 5
                reward_breakdown["cluster_health"] = 5
                logging.info(
                    "CLUSTER HEALTH: All pods are ready: "
                    f"{ready_replicas}/{self.replica_state}"
                )

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

        logging.info(f"Reward breakdown: {reward_breakdown}")
        logging.info(
            f"CPU: {cpu_usage:.2f}% | Memory: {memory_usage:.2f}% | "
            f"Replicas: {ready_replicas}"
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
                (self.target_memory[1], 0.8),
                (self.MEMORY_WARNING_THRESHOLD, 4.0),
                (self.MEMORY_LIMIT_THRESHOLD, 8.0),
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
                score += 25
                logging.info("OPTIMAL: Both CPU and memory in target range")
            elif cpu_optimal or memory_optimal:
                score += 12
                if cpu_optimal:
                    logging.info("GOOD: CPU in optimal range")
                if memory_optimal:
                    logging.info("GOOD: Memory in optimal range")

            if cpu_usage > 0 and memory_usage > 0:
                resource_balance = 1 - abs(cpu_usage - memory_usage) / 100
                balance_bonus = resource_balance * 3
                score += balance_bonus
                logging.debug(f"Resource balance bonus: +{balance_bonus:.2f}")

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
            if cpu_usage < self.target_cpu[0]:
                waste_factor = (self.target_cpu[0] - cpu_usage) / self.target_cpu[0]
                score -= ready_replicas * waste_factor * 2

            if memory_usage < self.target_memory[0]:
                waste_factor = (
                    self.target_memory[0] - memory_usage
                ) / self.target_memory[0]
                score -= ready_replicas * waste_factor * 1.5

        if (
            ready_replicas == 1
            and cpu_usage > self.MIN_EFFICIENT_CPU
            and memory_usage > self.MIN_EFFICIENT_MEMORY
        ):
            score += 5

        return score

    def _calculate_scaling_penalty(
        self, mapped_action, cpu_usage, memory_usage, ready_replicas
    ):
        """Penalize inappropriate scaling decisions with stricter rules"""
        penalty = 0

        if mapped_action < 0 and ready_replicas <= self.min_replicas:
            penalty += 200
            logging.error(
                f"CRITICAL PENALTY: Trying to scale below minimum replicas "
                f"(current: {ready_replicas}, min: {self.min_replicas}) = -200"
            )

        if mapped_action < 0 and abs(mapped_action) > ready_replicas:
            overshoot_penalty = abs(mapped_action) - ready_replicas
            penalty += overshoot_penalty * 50
            logging.error(
                f"OVERSHOOT PENALTY: Trying to scale down by {abs(mapped_action)} "
                f"when only {ready_replicas} replicas exist = -{overshoot_penalty * 50}"
            )

        if mapped_action < 0:
            if ready_replicas <= self.CRITICAL_REPLICA_THRESHOLD and (
                cpu_usage > self.HIGH_RESOURCE_THRESHOLD
                or memory_usage > self.HIGH_RESOURCE_THRESHOLD
            ):
                penalty += 300
                logging.error(
                    f"CRITICAL: Scaling down at minimum replicas with high usage "
                    f"(CPU={cpu_usage:.1f}%, MEM={memory_usage:.1f}%) = -300"
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
                penalty += abs(mapped_action) * 20
                logging.warning(
                    f"SEVERE INAPPROPRIATE SCALE DOWN: CPU={cpu_usage:.1f}%, "
                    f"MEM={memory_usage:.1f}% - CRITICAL ZONE"
                )
            elif warning_zone:
                penalty += abs(mapped_action) * 10
                logging.warning(
                    f"HIGH INAPPROPRIATE SCALE DOWN: CPU={cpu_usage:.1f}%, "
                    f"MEM={memory_usage:.1f}% - WARNING ZONE"
                )
            elif cpu_usage > self.target_cpu[1] or memory_usage > self.target_memory[1]:
                penalty += abs(mapped_action) * 5
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
            penalty += abs(mapped_action) * 2.0
            logging.warning(f"LARGE ACTION PENALTY: {abs(mapped_action)} replicas")
        elif abs(mapped_action) > self.MEDIUM_ACTION_THRESHOLD:
            penalty += abs(mapped_action) * 1.0
            logging.info(f"Medium action penalty: {abs(mapped_action)} replicas")

        return penalty

    def _calculate_stability_bonus(self, mapped_action):
        """Reward stable, gradual scaling decisions and penalize oscillations"""
        bonus = 0

        self.action_history.append(mapped_action)

        if mapped_action == 0:
            bonus += 5
        elif 0 < abs(mapped_action) <= self.SMALL_ACTION_THRESHOLD:
            bonus += 3
        elif (
            self.SMALL_ACTION_THRESHOLD
            < abs(mapped_action)
            <= self.MEDIUM_ACTION_THRESHOLD
        ):
            bonus += 1

        oscillation_penalty = self._detect_oscillation()
        bonus -= oscillation_penalty

        if self._is_in_good_state() and mapped_action == 0:
            bonus += 10

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

        if mapped_action < 0 and ready_replicas <= self.min_replicas:
            return True

        if mapped_action < 0 and abs(mapped_action) > ready_replicas:
            return True

        return mapped_action > 0 and ready_replicas + mapped_action > self.max_replicas

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

        self.iteration = self.init_iteration

        self.replica_history.clear()
        self.action_history.clear()
        self.last_scale_time = time.time()
        self.last_action = 0

        self._scale_deployment()
        _, _, ready_replicas = self._wait_for_pods_ready()
        cpu_usage, memory_usage, replica = self.get_metrics(replicas=ready_replicas)

        cpu_available, memory_available = self.get_node_resource_usage()

        unschedulable_replicas = 0
        observation = self._get_observation(
            cpu_usage,
            memory_usage,
            cpu_available,
            memory_available,
            unschedulable_replicas,
        )

        info = {
            "current_replicas": self.replica_state,
            "actual_replicas": replica,
            "action": 0,
            "state": self.replica_state,
            "iteration": self.iteration,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "target_cpu": self.target_cpu,
            "target_memory": self.target_memory,
            "cpu_available": cpu_available,
            "memory_available": memory_available,
            "unschedulable_replicas": unschedulable_replicas,
            "replica_trend": observation[6],
            "time_since_last_scale": observation[7],
            "resource_pressure_score": observation[8],
        }

        logging.info(f"Reset observation: {observation}")
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

                logging.debug(
                    f"Waiting for pods: {ready_replicas}/{desired_replicas} ready"
                )
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

    def _get_node_capacity(self):
        """Get total CPU and memory capacity across all nodes"""
        try:
            nodes = self.core.list_node()
            total_cpu = 0
            total_memory = 0
            schedulable_nodes = 0

            for node in nodes.items:
                if getattr(node.spec, "unschedulable", False):
                    logging.debug(f"Node {node.metadata.name}: SKIPPED (unschedulable)")
                    continue

                taints = getattr(node.spec, "taints", []) or []
                has_no_schedule_taint = any(
                    taint.effect == "NoSchedule"
                    and taint.key == "node-role.kubernetes.io/control-plane"
                    for taint in taints
                )

                if has_no_schedule_taint:
                    logging.info(
                        f"Node {node.metadata.name}: SKIPPED (control plane taint)"
                    )
                    continue

                allocatable = node.status.allocatable
                cpu_str = allocatable.get("cpu", "0")
                memory_str = allocatable.get("memory", "0")

                cpu_cores = self._parse_cpu_value(cpu_str)
                memory_mb = self._parse_memory_value(memory_str)

                total_cpu += cpu_cores
                total_memory += memory_mb
                schedulable_nodes += 1

                logging.info(
                    f"Node {node.metadata.name}: CPU={cpu_cores:.2f} cores, "
                    f"Memory={memory_mb:.2f} MB"
                )

            logging.info(f"Total schedulable nodes: {schedulable_nodes}")
            logging.info(
                f"Total capacity: CPU={total_cpu:.2f} cores, "
                f"Memory={total_memory:.2f} MB"
            )
            return total_cpu, total_memory

        except Exception as e:
            logging.warning(f"Could not get node capacity: {e}")
            return 6.0, 5820.0

    def get_node_resource_usage(self):
        """Get current resource usage across schedulable nodes only"""
        try:
            node_metrics = self.api.list_cluster_custom_object(
                group="metrics.k8s.io", version="v1beta1", plural="nodes"
            )

            nodes = self.core.list_node()
            schedulable_node_names = set()

            for node in nodes.items:
                if getattr(node.spec, "unschedulable", False):
                    continue

                taints = getattr(node.spec, "taints", []) or []
                has_no_schedule_taint = any(
                    taint.effect == "NoSchedule"
                    and taint.key == "node-role.kubernetes.io/control-plane"
                    for taint in taints
                )

                if not has_no_schedule_taint:
                    schedulable_node_names.add(node.metadata.name)

            total_cpu_used = 0
            total_memory_used = 0

            if node_metrics and "items" in node_metrics:
                for node in node_metrics["items"]:
                    node_name = node["metadata"]["name"]

                    if node_name not in schedulable_node_names:
                        logging.debug(
                            f"Node {node_name}: SKIPPED from metrics (not schedulable)"
                        )
                        continue

                    cpu_str = node["usage"]["cpu"]
                    memory_str = node["usage"]["memory"]

                    cpu_cores = self._parse_cpu_value(cpu_str)
                    memory_mb = self._parse_memory_value(memory_str)

                    total_cpu_used += cpu_cores
                    total_memory_used += memory_mb

                    logging.debug(
                        f"Node {node_name}: CPU usage={cpu_cores:.3f} cores, Memory "
                        f"usage={memory_mb:.2f} MB"
                    )

            cpu_available_percent = max(
                0, 100 - (total_cpu_used / self.node_cpu_total * 100)
            )
            memory_available_percent = max(
                0, 100 - (total_memory_used / self.node_memory_total * 100)
            )

            logging.info(
                f"Node availability - CPU: {cpu_available_percent:.1f}%, "
                f"Memory: {memory_available_percent:.1f}%"
            )

            return cpu_available_percent, memory_available_percent

        except Exception as e:
            logging.warning(f"Could not get node resource usage: {e}")
            return 50.0, 50.0

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
