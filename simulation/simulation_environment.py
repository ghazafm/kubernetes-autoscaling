import random
from typing import Any
from venv import logger

# helpers
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete


class K8sAutoscalerEnv(Env):
    def __init__(self):
        self.action_space = Discrete(7, start=-3)  # Actions: -3, -2, -1, 0, +1, +2, +3
        self.state = float(38 + random.randint(-3, 3))  # noqa: S311
        self.shower_length = 60

        # [cpu, memory, replicas, pending]
        self.observation_space = Box(
            low=np.array([0, 0, 1, 0], dtype=np.float32),
            high=np.array([100, 100, 50, 1000], dtype=np.float32),
            dtype=np.float32,
        )

        self.current_replicas = 1
        self.target_cpu = [50.0, 70.0]
        self.target_memory = [50.0, 70.0]
        self.max_replicas = 50
        self.min_replicas = 1
        self.episode_length = 100
        self.current_step = 0

        self.cpu_utilization = 50.0
        self.memory_utilization = 40.0
        self.pending_requests = 0

        self.HIGH_UTILIZATION_THRESHOLD = 90
        self.LOW_UTILIZATION_THRESHOLD = 30

    def step(self, action):
        scale_change = action
        new_replicas = max(
            self.min_replicas,
            min(self.max_replicas, self.current_replicas + scale_change),
        )

        self._simulate_system(new_replicas)
        self.current_replicas = new_replicas

        reward = self._calculate_reward()

        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        truncated = False

        # Prepare observation
        observation = self._get_observation()

        info = {
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "current_replicas": self.current_replicas,
            "pending_requests": self.pending_requests,
        }

        return observation, reward, terminated, truncated, info

    def _simulate_system(self, new_replicas):
        load_factor = random.uniform(0.8, 1.2)  # noqa: S311

        base_load = 200
        self.cpu_utilization = min(100, (base_load * load_factor) / new_replicas)
        self.memory_utilization = min(
            100, (base_load * load_factor * 0.8) / new_replicas
        )
        if (
            self.cpu_utilization > self.HIGH_UTILIZATION_THRESHOLD
            or self.memory_utilization > self.HIGH_UTILIZATION_THRESHOLD
        ):
            self.pending_requests = random.randint(10, 50)
        elif self.cpu_utilization > 80 or self.memory_utilization > 80:
            self.pending_requests = random.randint(0, 10)
        else:
            self.pending_requests = 0

        self.cpu_utilization += random.uniform(-5, 5)
        self.memory_utilization += random.uniform(-5, 5)

        self.cpu_utilization = max(0, min(100, self.cpu_utilization))
        self.memory_utilization = max(0, min(100, self.memory_utilization))

    def _calculate_reward(self):
        reward = 0

        cpu_distance = self._distance_from_range(self.cpu_utilization, self.target_cpu)
        memory_distance = self._distance_from_range(
            self.memory_utilization, self.target_memory
        )

        # Reward semakin tinggi jika mendekati target
        reward += max(0, 20 - cpu_distance / 2)
        reward += max(0, 20 - memory_distance / 2)

        # Pending requests penalty
        reward -= self.pending_requests * 0.5

        # High utilization penalty
        if (
            self.cpu_utilization > self.HIGH_UTILIZATION_THRESHOLD
            or self.memory_utilization > self.HIGH_UTILIZATION_THRESHOLD
        ):
            reward -= 20

        # Low utilization penalty
        if (
            self.cpu_utilization < self.LOW_UTILIZATION_THRESHOLD
            and self.memory_utilization < self.LOW_UTILIZATION_THRESHOLD
        ):
            reward -= 10

        # Efisiensi Cost
        if self.current_replicas > 10:
            reward -= (self.current_replicas - 10) * 0.2

        return reward

    def _distance_from_range(self, value, target_range):
        min_target, max_target = target_range

        if min_target <= value <= max_target:
            return 0
        if value < min_target:
            return min_target - value
        return value - max_target

    def _get_observation(self):
        """Get current observation state"""
        return np.array(
            [
                self.cpu_utilization,
                self.memory_utilization,
                float(self.current_replicas),
                float(self.pending_requests),
            ],
            dtype=np.float32,
        )

    def render(self, mode="human"):
        """Render the current state"""
        if mode == "human":
            logger.info(f"Step: {self.current_step}")
            logger.info(f"Replicas: {self.current_replicas}")
            logger.info(f"CPU Utilization: {self.cpu_utilization:.1f}%")
            logger.info(f"Memory Utilization: {self.memory_utilization:.1f}%")
            logger.info(f"Pending Requests: {self.pending_requests}")
            logger.info("-" * 40)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """Reset the environment to initial state"""
        super().reset(seed=seed, options=options)

        self.current_replicas = 3
        self.current_step = 0
        self.cpu_utilization = 50.0 + random.uniform(-10, 10)
        self.memory_utilization = 40.0 + random.uniform(-10, 10)
        self.pending_requests = 0

        observation = self._get_observation()
        info = {}

        return observation, info
