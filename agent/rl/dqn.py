from logging import Logger
from pathlib import Path
from typing import Optional

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
        target_update_freq: int = 100,
        grad_clip_norm: float = 10.0,
        created_at: int = 0,
        logger: Optional[Logger] = None,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon_start=epsilon_start,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            created_at=created_at,
            logger=logger,
        )

        self.train_step = 0

        self.agent_type = "DQN"

        self.device = device
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.target_update_freq = target_update_freq
        self.grad_clip_norm = grad_clip_norm
        # State: [cpu%, mem%, rt%, replica%, action%,
        #         cpu_Δ, mem_Δ, rt_Δ, time_in_state, scaling_dir]
        self._state_dim = 10

        self.policy_net = QNetwork(self._state_dim, self.n_actions).to(self.device)
        self.target_net = QNetwork(self._state_dim, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer + Replay Buffer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.logger.info("Initialized DQN agent")
        self.logger.info(f"Agent parameters: {self.__dict__}")

    def get_state_key(self, observation: dict) -> np.ndarray:
        """Convert observation to a numpy array state for DQN"""
        cpu = observation["cpu_usage"] / 100.0  # Normalize to 0-1
        memory = observation["memory_usage"] / 100.0  # Normalize to 0-1
        last_action = observation["last_action"] / 99.0  # Normalize to 0-1

        response_time_raw = observation["response_time"]
        if np.isnan(response_time_raw) or response_time_raw is None:
            response_time = 0.0
        else:
            response_time = min(
                response_time_raw / 100.0, 1.0
            )  # Normalize and cap at 1.0

        # NEW: Current replica percentage (already 0-100, normalize to 0-1)
        current_replica_pct = observation.get("current_replica_pct", 0.0) / 100.0

        # NEW: Metric deltas (already -100 to +100, normalize to -1 to +1)
        cpu_delta = np.clip(observation.get("cpu_delta", 0.0) / 100.0, -1.0, 1.0)
        memory_delta = np.clip(observation.get("memory_delta", 0.0) / 100.0, -1.0, 1.0)
        rt_delta = np.clip(observation.get("rt_delta", 0.0) / 100.0, -1.0, 1.0)

        # NEW: Time in state (already 0-1)
        time_in_state = observation.get("time_in_state", 0.0)

        # NEW: Scaling direction (already 0, 0.5, or 1)
        scaling_direction = observation.get("scaling_direction", 0.5)

        return np.array(
            [
                cpu,
                memory,
                response_time,
                current_replica_pct,
                last_action,
                cpu_delta,
                memory_delta,
                rt_delta,
                time_in_state,
                scaling_direction,
            ],
            dtype=np.float32,
        )

    def get_action(self, observation: dict) -> int:
        """Choose action using epsilon-greedy strategy"""
        state = self.get_state_key(observation)

        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            with torch.no_grad():
                s = (
                    torch.from_numpy(state).unsqueeze(0).to(self.device)
                )  # [1, state_dim]
                q = self.policy_net(s)  # [1, n_actions]
            action = int(torch.argmax(q, dim=1).item())

        return action

    def update_q_table(
        self, observation: dict, action: int, reward: float, next_observation: dict
    ) -> None:
        """Update Q-table using Q-learning algorithm and DQN if enabled"""

        try:
            state_key = self.get_state_key(observation)
            next_state_key = self.get_state_key(next_observation)

            done = bool(next_observation.get("terminated", False))

            # simpan pengalaman
            self.replay_buffer.push(
                state_key, int(action), float(reward), next_state_key, float(done)
            )

            if self.epsilon_decay and self.epsilon > self.epsilon_min:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            self.train_step += 1

            self.logger.debug(
                f"Buffer: {len(self.replay_buffer)}/{self.batch_size}, "
                f"Train step: {self.train_step}, "
                f"Epsilon: {self.epsilon:.4f}"
            )

            if len(self.replay_buffer) < self.batch_size:
                if self.train_step % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    self.logger.debug(
                        f"Target network updated at step {self.train_step} (buffer not "
                        "ready for learning)"
                    )
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

            if self.train_step % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.logger.debug(
                    f"Target network updated at step {self.train_step} (after learning)"
                )

        except Exception as e:
            self.logger.error(
                f"Error during DQN update (step {self.train_step}): {e}. "
                "Skipping this update to prevent training crash."
            )
            # Continue training even if one update fails
            return

    def save_model(self, filepath: str, episode_count: int = 0) -> None:
        """Save DQN model and parameters to file"""
        try:
            model_data = {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "n_actions": self.n_actions,
                "train_step": self.train_step,
                "device": self.device,
                "batch_size": self.batch_size,
                "buffer_size": self.buffer_size,
                "target_update_freq": self.target_update_freq,
                "grad_clip_norm": self.grad_clip_norm,
                "created_at": self.created_at,
                "episodes_trained": episode_count,
            }

            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            torch.save(model_data, filepath)
            self.logger.info(f"DQN model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save DQN model to {filepath}: {e}")
            raise

    def load_model(self, filepath: str) -> None:
        """Load DQN model and parameters from file"""
        try:
            if not Path(filepath).exists():
                raise FileNotFoundError(f"Model file not found: {filepath}")

            model_data = torch.load(
                filepath, map_location=self.device, weights_only=False
            )

            # Load network states
            self.policy_net.load_state_dict(model_data["policy_net_state_dict"])
            self.target_net.load_state_dict(model_data["target_net_state_dict"])
            self.optimizer.load_state_dict(model_data["optimizer_state_dict"])

            # Load hyperparameters
            self.learning_rate = model_data["learning_rate"]
            self.discount_factor = model_data["discount_factor"]
            self.epsilon = model_data["epsilon"]
            self.epsilon_min = model_data["epsilon_min"]
            self.epsilon_decay = model_data["epsilon_decay"]
            self.n_actions = model_data["n_actions"]
            self.train_step = model_data["train_step"]
            self.batch_size = model_data["batch_size"]
            self.buffer_size = model_data["buffer_size"]
            self.target_update_freq = model_data["target_update_freq"]
            self.grad_clip_norm = model_data["grad_clip_norm"]
            self.created_at = model_data["created_at"]
            self.episodes_trained = model_data["episodes_trained"]

            self.logger.info(f"DQN model loaded from {filepath}")
            self.logger.info(f"Training step: {self.train_step}")
            self.logger.info(f"Current epsilon: {self.epsilon:.4f}")
        except Exception as e:
            self.logger.error(f"Failed to load DQN model from {filepath}: {e}")
            raise
