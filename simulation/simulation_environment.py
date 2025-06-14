# Gym
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

# helpers
import numpy as np
import random
import os

# Stable-baseline
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from typing import Any


class K8sAutoscalerEnv(Env):
    def __init__(self):
        self.action_space = Discrete(7, start=-3)  # Actions: -3, -2, -1, 0, +1, +2, +3
        self.observation_space = Box(low=0, high=100)
        self.state = float(38 + random.randint(-3, 3))
        self.shower_length = 60

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
        load_factor = random.uniform(0.8, 1.2)

        base_load = 200
        self.cpu_utilization = min(100, (base_load * load_factor) / new_replicas)
        self.memory_utilization = min(100, (base_load * load_factor * 0.8) / new_replicas)

        if self.cpu_utilization > 90 or self.memory_utilization > 90:
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
        memory_distance = self._distance_from_range(self.memory_utilization, self.target_memory)
        
        reward += max(0, 20 - cpu_distance/2)      
        reward += max(0, 20 - memory_distance/2)  
        
        reward -= self.pending_requests * 0.5
        
        if self.cpu_utilization > 90 or self.memory_utilization > 90:
            reward -= 20
        
        if self.cpu_utilization < 30 and self.memory_utilization < 30:
            reward -= 10
        
        if self.current_replicas > 10:
            reward -= (self.current_replicas - 10) * 0.2
        
        return reward

    def _distance_from_range(self, value, target_range):
        min_target, max_target = target_range
        
        if min_target <= value <= max_target:
            return 0
        elif value < min_target:
            return min_target - value
        else:
            return value - max_target
        
    def render(self):
        pass

    def reset(self):
        pass
