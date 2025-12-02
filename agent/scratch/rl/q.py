import pickle
from logging import Logger
from pathlib import Path
from typing import Optional

import numpy as np
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class Q:
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 0.1,
        epsilon_decay: float = 0.0,
        epsilon_min: float = 0.01,
        created_at: int = 0,
        logger: Optional[Logger] = None,
    ):
        self.n_actions = 100  # Action dari 0-99 (100 total actions)
        self.agent_type = "Q"
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        # Epsilon: probabilitas eksplorasi awal
        # Contoh: epsilon_start=0.1 -> 10% pilih acak
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.created_at = created_at
        self.episodes_trained = 0
        self.q_table = {}
        self.logger = logger or Logger(__name__)

        self.logger.info("Initialized Q-learning agent")
        self.logger.info(f"Agent parameters: {self.__dict__}")

    def add_episode_count(self, count: int = 1):
        """Increment the episode count"""
        self.episodes_trained += count

    def reset_epsilon(self, value: float) -> None:
        """Reset epsilon to a specific value for periodic re-exploration"""
        old_epsilon = self.epsilon
        self.epsilon = max(self.epsilon_min, min(value, 1.0))
        self.logger.info(f"🔄 Epsilon reset: {old_epsilon:.4f} → {self.epsilon:.4f}")

    def get_state_key(
        self, observation: dict
    ) -> tuple[int, int, int, int, int, int, int, int, int, int]:
        """Convert observation to a hashable state key with discretization"""
        cpu = int(observation["cpu_usage"])
        memory = int(observation["memory_usage"])
        action = int(observation["last_action"])

        response_time_raw = observation["response_time"]
        if np.isnan(response_time_raw) or response_time_raw is None:
            response_time = 0
        else:
            response_time = int(response_time_raw)

        # NEW: Discretize additional state components
        replica_pct = int(observation.get("current_replica_pct", 0))

        # Discretize deltas to -100 to +100 range, then shift to 0-200
        cpu_delta = int(observation.get("cpu_delta", 0) + 100)
        memory_delta = int(observation.get("memory_delta", 0) + 100)
        rt_delta = int(observation.get("rt_delta", 0) + 100)

        # Discretize time_in_state (0-1) to bins (0-10)
        time_in_state = int(observation.get("time_in_state", 0) * 10)

        # Discretize scaling_direction (0, 0.5, 1) to (0, 1, 2)
        scaling_dir_raw = observation.get("scaling_direction", 0.5)
        SCALING_DIRECTION_THRESHOLD = 0.25
        SCALING_DIRECTION_UP_THRESHOLD = 0.75
        if scaling_dir_raw <= SCALING_DIRECTION_THRESHOLD:
            scaling_direction = 0  # Down
        elif scaling_dir_raw >= SCALING_DIRECTION_UP_THRESHOLD:
            scaling_direction = 2  # Up
        else:
            scaling_direction = 1  # Same

        return (
            cpu,
            memory,
            response_time,
            replica_pct,
            action,
            cpu_delta,
            memory_delta,
            rt_delta,
            time_in_state,
            scaling_direction,
        )

    def get_action(self, observation: dict) -> int:
        """Choose action using epsilon-greedy strategy"""
        state_key = self.get_state_key(observation)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)

        # Choose action based on epsilon-greedy
        # Eksplorasi vs Eksploitasi
        # Jika random < epsilon -> eksplorasi (pilih acak)
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            action = np.argmax(self.q_table[state_key])

        # REMOVED: Epsilon decay moved to update() to avoid double decay
        return action

    def update(
        self, observation: dict, action: int, reward: float, next_observation: dict
    ):
        """Update Q-table using Q-learning algorithm"""
        state_key = self.get_state_key(observation)
        next_state_key = self.get_state_key(next_observation)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.n_actions)

        best_next_action = np.max(
            self.q_table[next_state_key]
        )  # Ini rumus Q Learning, jika sarsa akan memanggil fungsi get_action()
        self.q_table[state_key][action] += self.learning_rate * (
            reward
            + self.discount_factor * best_next_action
            - self.q_table[state_key][action]
        )
        if self.epsilon > self.epsilon_min:
            # Decay epsilon agar seiring waktu agent lebih mengeksploitasi
            # Contoh: epsilon=0.1, epsilon_decay=0.99 -> epsilon->0.099
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # Q(S,A)←Q(S,A)+α(R+γQ(S′,A′)−Q(S,A))  # noqa: RUF003

    """
    # Q(S,A) [Reward sebelumnya yang sudah tercatat di Q table]
    # R [Reward terbaru yang didapat dari action yang diambil]

    # @ [Learning rate, seberapa besar pengaruh reward terbaru terhadap Q(S,A)]

    # Y [Discount factor, seberapa besar pengaruh reward di masa depan terhadap Q(S,A) Jadi
    di Q Learning juga memperhitungkan reward kemungkinan pada step berikutnya dari perolehan state(observation) yang sekarang]

    # (S`,A`) [State dan action pada step berikutnya dari state(observation) yang sekarang]

    # -Q(S,A) [Untuk mengurangi pengaruh reward sebelumnya yang sudah tercatat di Q table] lebih stabil
    """  # noqa: E501

    def save_model(self, filepath: str, episode_count: int = 0):
        """Save Q-table and parameters to file"""
        try:
            model_data = {
                "q_table": self.q_table,
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "n_actions": self.n_actions,
                "created_at": self.created_at,
                "episodes_trained": episode_count,
            }
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with Path(filepath).open("wb") as f:
                pickle.dump(model_data, f)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save model to {filepath}: {e}")
            raise

    def load_model(self, filepath: str):
        """Load Q-table and parameters from file"""
        try:
            if not Path(filepath).exists():
                raise FileNotFoundError(f"Model file not found: {filepath}")

            with Path(filepath).open("rb") as f:
                model_data = pickle.load(f)  # noqa: S301

            self.q_table = model_data["q_table"]
            self.learning_rate = model_data["learning_rate"]
            self.discount_factor = model_data["discount_factor"]
            self.epsilon = model_data["epsilon"]
            self.epsilon_min = model_data["epsilon_min"]
            self.epsilon_decay = model_data["epsilon_decay"]
            self.n_actions = model_data["n_actions"]
            self.created_at = model_data.get("created_at", None)
            self.episodes_trained = model_data.get("episodes_trained", 0)

            self.logger.info(f"Model loaded from {filepath}")
            self.logger.info(f"Q-table size: {len(self.q_table)} states")
        except Exception as e:
            self.logger.error(f"Failed to load model from {filepath}: {e}")
            raise
