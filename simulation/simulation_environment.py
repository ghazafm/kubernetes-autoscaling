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
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class K8sAutoscalerEnv(Env):
    LOW_USAGE_THRESHOLD = 5.0
    OPTIMAL_USAGE_MIN = 20.0
    OPTIMAL_USAGE_MAX = 70.0
    HIGH_USAGE_THRESHOLD = 80.0
    CRITICAL_USAGE_THRESHOLD = 85.0

    def __init__(
        self,
        min_replicas: int = 1,
        max_replicas: int = 50,
        iteration: int = 100,
        namespace: str = "default",
        deployment_name: str = "default",
        min_cpu: float = 20,
        min_memory: float = 20,
        max_cpu: float = 100,
        max_memory: float = 100,
        verbose: bool = False,
        action_step: int = 1,
        timeout: int = 60,
    ):
        config.load_kube_config()
        self.verbose = verbose
        self.cluster = client.AppsV1Api()
        self.api = client.CustomObjectsApi()
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.timeout = timeout

        action_range = action_step * 2 + 1
        self.action_step = action_step
        if action_range <= 0:
            raise ValueError("action_step must be a positive integer")
        # Use regular Discrete space for SB3 compatibility
        self.action_space = Discrete(action_range)

        # [cpu, memory, replicas]
        self.observation_space = Box(
            low=np.array([0, 0, 1], dtype=np.float32),
            high=np.array([100, 100, 50], dtype=np.float32),
            dtype=np.float32,
        )

        self.replica_state = (
            min_replicas + 1 if min_replicas < max_replicas - 1 else min_replicas
        )
        self.target_cpu = [min_cpu, max_cpu]
        self.target_memory = [min_memory, max_memory]
        self.max_replicas = max_replicas
        self.min_replicas = min_replicas
        self.init_iteration = iteration
        self.iteration = iteration

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
        while True:
            metric_data = self.api.list_namespaced_custom_object(
                group="metrics.k8s.io",
                version="v1beta1",
                namespace="default",
                plural="pods",
            )
            replica = 0
            cpu_usage = []
            memory_usage = []

            if not metric_data or "items" not in metric_data:
                logging.warning("No metric data found.")
                return 0, 0, 0

            for item in metric_data["items"]:
                if self.deployment_name in item["metadata"]["name"]:
                    cpu_str = item["containers"][0]["usage"]["cpu"]
                    if cpu_str.endswith("n"):
                        cpu_value = float(cpu_str[:-1]) / 1000000000
                    elif cpu_str.endswith("m"):
                        cpu_value = float(cpu_str[:-1]) / 1000
                    else:
                        cpu_value = float(cpu_str)

                    cpu_percentage = (cpu_value / 0.5) * 100

                    memory_str = item["containers"][0]["usage"]["memory"]
                    if memory_str.endswith("Ki"):
                        memory_mb = float(memory_str[:-2]) / 1024
                    elif memory_str.endswith("Mi"):
                        memory_mb = float(memory_str[:-2])
                    elif memory_str.endswith("Gi"):
                        memory_mb = float(memory_str[:-2]) * 1024
                    else:
                        memory_mb = float(memory_str) / (1024 * 1024)

                    memory_percentage = (memory_mb / 128) * 100
                    cpu_usage.append(cpu_percentage)
                    memory_usage.append(memory_percentage)
                    replica += 1

            if replica >= replicas:
                break

        logging.info(f"Fetched metrics for {replica} replicas.")
        cpu_usage_mean = np.mean(cpu_usage) if cpu_usage else 0
        memory_usage_mean = np.mean(memory_usage) if memory_usage else 0
        logging.info(f"CPU usage: {cpu_usage_mean}, Memory usage: {memory_usage_mean}")
        return cpu_usage_mean, memory_usage_mean, replica

    def step(self, action):
        mapped_action = action - self.action_step
        logging.info(
            f"Action taken: {mapped_action} (raw: {action}), New replica state: "
            f"{self.replica_state}"
        )

        self.replica_state = max(
            self.min_replicas,
            min(self.max_replicas, self.replica_state + mapped_action),
        )
        self.iteration -= 1
        self._scale_deployment()
        ready, desired_replicas, ready_replicas = self._wait_for_pods_ready()
        cpu_usage, memory_usage, replica = self.get_metrics(replicas=ready_replicas)

        reward = 0
        reward_breakdown = {
            "resource_efficiency": 0,
            "scaling_penalty": 0,
            "cluster_health": 0,
            "stability_bonus": 0,
        }

        # 1. CLUSTER HEALTH PENALTY
        unschedulable_pods = max(0, desired_replicas - ready_replicas)
        if unschedulable_pods > 0 and not ready:
            cluster_penalty = unschedulable_pods * 25
            reward -= cluster_penalty
            reward_breakdown["cluster_health"] = -cluster_penalty
            logging.warning(
                f"CLUSTER PENALTY: {unschedulable_pods} unschedulable pods ="
                f" -{cluster_penalty}"
            )
        else:
            logging.info("CLUSTER HEALTH: All pods are ready")

        cpu_reward = 0
        if ready_replicas == self.min_replicas:
            if cpu_usage < self.target_cpu[0]:
                cpu_reward = 10 - (self.target_cpu[0] - cpu_usage) * 0.1
            elif cpu_usage > self.target_cpu[1]:
                cpu_reward = -5 - (cpu_usage - self.target_cpu[1]) * 0.2
            else:
                mid_point = (self.target_cpu[0] + self.target_cpu[1]) / 2
                distance_from_mid = abs(cpu_usage - mid_point)
                cpu_reward = 12 - distance_from_mid * 0.05
        elif cpu_usage < self.target_cpu[0]:
            waste_factor = (self.target_cpu[0] - cpu_usage) / self.target_cpu[0]
            cpu_reward = -ready_replicas * waste_factor * 3
        elif cpu_usage > self.target_cpu[1]:
            overload_factor = (cpu_usage - self.target_cpu[1]) / (
                100 - self.target_cpu[1]
            )
            cpu_reward = -8 - overload_factor * 8
        else:
            mid_point = (self.target_cpu[0] + self.target_cpu[1]) / 2
            distance_from_mid = abs(cpu_usage - mid_point)
            cpu_reward = 8 - distance_from_mid * 0.03

        memory_reward = 0
        if ready_replicas == 1:
            if memory_usage < self.target_memory[0]:
                memory_reward = 10 - (self.target_memory[0] - memory_usage) * 0.1
            elif memory_usage > self.target_memory[1]:
                memory_reward = -5 - (memory_usage - self.target_memory[1]) * 0.2
            else:
                mid_point = (self.target_memory[0] + self.target_memory[1]) / 2
                distance_from_mid = abs(memory_usage - mid_point)
                memory_reward = 12 - distance_from_mid * 0.05
        elif memory_usage < self.target_memory[0]:
            waste_factor = (self.target_memory[0] - memory_usage) / self.target_memory[
                0
            ]
            memory_reward = -ready_replicas * waste_factor * 3
        elif memory_usage > self.target_memory[1]:
            overload_factor = (memory_usage - self.target_memory[1]) / (
                100 - self.target_memory[1]
            )
            memory_reward = -8 - overload_factor * 8
        else:
            mid_point = (self.target_memory[0] + self.target_memory[1]) / 2
            distance_from_mid = abs(memory_usage - mid_point)
            memory_reward = 8 - distance_from_mid * 0.03

        reward += cpu_reward + memory_reward
        if self.iteration <= 0:
            terminated = True
            truncated = False
        else:
            terminated = False
            truncated = False

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
        }
        if self.verbose:
            logging.debug(
                f"Step info: {info}, Reward: {reward}, Terminated: {terminated}, "
                f"Truncated: {truncated}",
            )
            logging.info(
                f"CPU usage: {cpu_usage}, Memory usage: {memory_usage}, "
                f"Replicas: {replica}"
            )
        logging.info(f"Reward: {reward}")
        logging.info("===========================")
        return (
            np.array([cpu_usage, memory_usage, self.replica_state], dtype=np.float32),
            reward,
            terminated,
            truncated,
            info,
        )

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

        max_initial_replicas = min(self.max_replicas, 10)
        self.replica_state = np.random.randint(
            self.min_replicas, max_initial_replicas + 1
        )

        self._scale_deployment()
        _, _, ready_replicas = self._wait_for_pods_ready()
        cpu_usage, memory_usage, replica = self.get_metrics(replicas=ready_replicas)

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
        }
        return (
            self.replica_state,
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

                if ready_replicas >= desired_replicas > 0:
                    logging.info(f"All {desired_replicas} pods are ready")
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
