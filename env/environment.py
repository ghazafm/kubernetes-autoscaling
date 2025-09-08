import logging
import time

from kubernetes import client, config
from utils.cluster import wait_for_pods_ready
from utils.metrics import get_metrics, get_response_time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=f"simulation_environment_{time.strftime('%Y%m%d_%H%M%S')}.log",
    filemode="a",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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
        verbose: bool = False,
        timeout: int = 60,
    ):
        config.load_kube_config()
        self.cluster = client.AppsV1Api()
        self.api = client.CustomObjectsApi()
        self.core = client.CoreV1Api()
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.range_replicas = self.max_replicas - self.min_replicas
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
            "replicas": (1, 100),
            "cpu_usage": (0, 100),
            "memory_usage": (0, 100),
            "response_time": (0, 1000),
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
            logging.warning(
                f"Pods are not ready, {ready_replicas}/{desired_replicas} ready"
            )

    def get_observation(self):
        return {
            "replicas": self.replica_state,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "response_time": self.response_time,
            "last_action": self.last_action,
        }

    def step(self, action: int):
        self.last_action = action
        self.replica_state = min(
            self.min_replicas
            + (self.range_replicas * action / 100 + self.min_replicas),
            self.max_replicas,
        )

        self.scale_and_get_metrics()

        reward = self.calculate_reward()

        terminated = bool(self.iteration <= 0)

        observation = self.get_observation()
        info = {
            "iteration": self.iteration,
            "action": action,
            "reward": reward,
            "terminated": terminated,
        }
        return observation, reward, terminated, info

    def calculate_reward(self):
        """
        Simple reward function for Kubernetes autoscaling.
        Rewards efficient resource usage while maintaining good response times.
        """
        cpu_penalty = max(0, self.cpu_usage - self.max_cpu) * -0.1
        memory_penalty = max(0, self.memory_usage - self.max_memory) * -0.1

        cpu_underuse_penalty = max(0, self.min_cpu - self.cpu_usage) * -0.05
        memory_underuse_penalty = max(0, self.min_memory - self.memory_usage) * -0.05

        response_penalty = max(0, self.response_time - 200) * -0.01

        base_reward = 1.0

        total_reward = (
            base_reward
            + cpu_penalty
            + memory_penalty
            + cpu_underuse_penalty
            + memory_underuse_penalty
            + response_penalty
        )

        return max(total_reward, -1.0)

    def reset(self):
        self.iteration = self.initial_iteration
        self.replica_state = self.min_replicas
        self.scale()
        return self.get_observation()
