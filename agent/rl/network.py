import torch.nn.functional as F
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Increased capacity for better Q-value approximation
        # Input: [cpu, memory, response_time, last_action] = 4 features
        # Hidden layers: 128 → 128 → 64
        # Output: 100 Q-values (one per action)
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, action_dim)

    def forward(self, x):
        # x: tensor state, shape [batch, state_dim]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # output without activation (Q values can be negative/positive)
        return self.fc_out(x)
