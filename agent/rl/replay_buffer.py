import random
from collections import deque

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
        # Konversi ke tensor PyTorch
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
