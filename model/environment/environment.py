from logging import Logger
from typing import Optional

from kubernetes import client, config

from .utils import get_metrics, get_response_time, wait_for_pods_ready


class KubernetesEnv:
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
        timeout: int = 60,
        verbose: bool = False,
        logger: Optional[Logger] = None,
    ):
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
        self.verbose = verbose
        self.timeout = timeout

        self.action_space = list(range(101))

        self.observation_space = {
            "cpu_usage": (0, 100.0),
            "memory_usage": (0, 100.0),
            "response_time": (0, 1000.0),
            "last_action": (1, 100),
        }

    def scale(self):
        http_timeout = 30
        self.cluster.patch_namespaced_deployment_scale(
            name=self.deployment_name,
            body=client.V1Scale(
                spec=client.V1ScaleSpec(replicas=int(self.replica_state))
            ),
            namespace=self.namespace,
            _request_timeout=http_timeout,
        )

    def scale_and_get_metrics(self):
        self.scale()
        ready, desired_replicas, ready_replicas = wait_for_pods_ready(
            cluster=self.cluster,
            deployment_name=self.deployment_name,
            namespace=self.namespace,
            timeout=self.timeout,
        )
        self.cpu_usage, self.memory_usage, self.replica = get_metrics(
            replicas=ready_replicas,
            timeout=self.timeout,
            namespace=self.namespace,
            deployment_name=self.deployment_name,
            api=self.api,
            core=self.core,
        )

        self.response_time = get_response_time()

        if not ready:
            self.logger.warning(
                f"Pods are not ready, {ready_replicas}/{desired_replicas} ready"
            )

    def get_observation(self):
        return {
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "response_time": self.response_time,
            "last_action": self.last_action,
        }

    def step(self, action: int):
        self.last_action = action
        ratio = action / 100.0
        self.replica_state = round(self.min_replicas + ratio * self.range_replicas)
        self.replica_state = max(
            self.min_replicas, min(self.replica_state, self.max_replicas)
        )

        self.scale_and_get_metrics()

        reward = self.calculate_reward()

        self.iteration -= 1
        terminated = bool(self.iteration <= 0)

        observation = self.get_observation()
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
        return observation, reward, terminated, info

    def calculate_reward(self):
        SLA = 200.0  # ms

        # Penalti latency hanya jika melebihi SLA (0..∞), dinormalisasi ke ~0..1
        resp_pen = min(
            1.0, max(0.0, (self.response_time - SLA) / SLA)
        )  # Menjaga agar penalti tidak melebihi 1
        # Selain itu max() untuk memastikan penalti tidak negatif jika di bawah SLA/RELU

        # Penalti biner: 0 jika dalam batas, 1 jika di luar
        cpu_pen = 0.0 if self.min_cpu <= self.cpu_usage <= self.max_cpu else 1.0
        mem_pen = (
            0.0 if self.min_memory <= self.memory_usage <= self.max_memory else 1.0
        )
        cost_pen = (
            0.1 * (self.replica_state - self.min_replicas) / self.range_replicas
        )  # Agar menambahkan bias ke minimum pods untuk efisiensi biaya

        # Reward sederhana: mulai dari 1, kurangi penalti
        reward = 1.0 - resp_pen - 0.5 * (cpu_pen + mem_pen) - cost_pen

        # Clamp agar stabil
        return float(max(min(reward, 1.0), -1.0))

    def reset(self):
        self.iteration = self.initial_iteration
        self.replica_state = self.min_replicas
        self.scale_and_get_metrics()
        self.last_action = 1
        return self.get_observation()
