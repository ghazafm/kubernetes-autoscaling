import torch.nn.functional as F
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Layer input -> hidden1
        self.fc1 = nn.Linear(state_dim, 64)
        # Layer hidden1 -> hidden2
        self.fc2 = nn.Linear(64, 64)
        # Layer hidden2 -> output (Q values for each action)
        self.fc_out = nn.Linear(64, action_dim)

    def forward(self, x):
        # x: tensor state, shape [batch, state_dim]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # output tanpa aktivasi (nilai Q bisa bernilai negatif/positif sebarang)
        return self.fc_out(x)
