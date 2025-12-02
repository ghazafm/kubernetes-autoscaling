import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # maksimum pengalaman yang disimpan

    def push(self, state, action, reward, next_state, done):
        # Menyimpan satu tuple pengalaman (s, a, r, s', done)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Sampling random batch dari buffer
        batch = random.sample(self.buffer, batch_size)
        # Pisahkan masing-masing komponen pengalaman menjadi batch tensor
        states, actions, rewards, next_states, dones = zip(*batch)
        # Konversi ke numpy arrays terlebih dahulu untuk efisiensi
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        # Konversi ke tensor PyTorch
        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions)
        rewards = torch.from_numpy(rewards)
        next_states = torch.from_numpy(next_states)
        dones = torch.from_numpy(dones)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
