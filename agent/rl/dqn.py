import numpy as np
import torch
import torch.nn.functional as F
import urllib3
from torch import optim

from .network import QNetwork
from .q import Q
from .replay_buffer import ReplayBuffer

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class DQN(Q):
    def __init__(
        self,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        epsilon_start: float = 0.1,
        epsilon_decay: float = 0.0,
        epsilon_min: float = 0.01,
        device: str = "cpu",
        buffer_size: int = 50000,
        batch_size: int = 64,
        target_update_freq: int = 200,
        grad_clip_norm: float = 10.0,
    ):
        super().__init__(
            learning_rate,
            discount_factor,
            epsilon_start,
            epsilon_decay,
            epsilon_min,
        )

        self.train_step = 0

        self.device = device
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.target_update_freq = target_update_freq
        self.grad_clip_norm = grad_clip_norm
        self._state_dim = 4  # [cpu%, mem%, response_time_norm, last_action%]

        self.policy_net = QNetwork(self._state_dim, self.n_actions).to(self.device)
        self.target_net = QNetwork(self._state_dim, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer + Replay Buffer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(self.buffer_size)

    def get_state_key(self, observation: dict) -> np.ndarray:
        """Convert observation to a numpy array state for DQN"""
        cpu = observation["cpu_usage"] / 100.0  # Normalize to 0-1
        memory = observation["memory_usage"] / 100.0  # Normalize to 0-1
        last_action = observation["last_action"] / 100.0  # Normalize to 0-1

        response_time_raw = observation["response_time"]
        if np.isnan(response_time_raw) or response_time_raw is None:
            response_time = 0.0
        else:
            response_time = min(
                response_time_raw / 1000.0, 1.0
            )  # Normalize and cap at 1.0

        return np.array([cpu, memory, response_time, last_action], dtype=np.float32)

    def get_action(self, observation: dict) -> int:
        """Choose action using epsilon-greedy strategy"""
        state = self.get_state_key(observation)

        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_actions)

        with torch.no_grad():
            s = torch.from_numpy(state).unsqueeze(0).to(self.device)  # [1, state_dim]
            q = self.policy_net(s)  # [1, n_actions]
        return int(torch.argmax(q, dim=1).item())

    def update_q_table(
        self, observation: dict, action: int, reward: float, next_observation: dict
    ):
        """Update Q-table using Q-learning algorithm and DQN if enabled"""

        state_key = self.get_state_key(observation)
        next_state_key = self.get_state_key(next_observation)

        done = bool(next_observation.get("done", False))

        # simpan pengalaman
        self.replay_buffer.push(
            state_key, int(action), float(reward), next_state_key, float(done)
        )

        # belajar kalau buffer cukup
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Q(s,a)
        q_values = self.policy_net(states)  # [B, A]
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

        # target: r + gamma * max_a' Q_target(s',a') * (1 - done)
        with torch.no_grad():
            q_next = self.target_net(next_states)  # [B, A]
            max_q_next, _ = torch.max(q_next, dim=1)  # [B]
            target = rewards + (1.0 - dones) * self.discount_factor * max_q_next

        # Huber loss lebih stabil
        loss = F.smooth_l1_loss(q_sa, target)

        # step
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.grad_clip_norm
        )
        self.optimizer.step()

        # decay epsilon
        if self.epsilon_decay and self.epsilon > self.epsilon_min:
            self.epsilon = max(
                self.epsilon_min, self.epsilon * (1.0 - self.epsilon_decay)
            )

        # hard update target network
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return
