import json
import time
from pathlib import Path

import numpy as np
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class QLearningAgent:
    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1,
        epsilon_decay=0.0,
        epsilon_min=0.01,
    ):
        self.n_actions = 101  # Action dari 0-100 persentase
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # probabilitas untuk eksplorasi
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}

    def get_state_key(self, observation):
        """Convert observation to a hashable state key"""
        # cpu = int(observation["cpu_usage"] // 10)
        # memory = int(observation["memory_usage"] // 10)
        # response_time = int(observation["response_time"] // 100)
        cpu = int(observation["cpu_usage"])
        memory = int(observation["memory_usage"])
        response_time = int(observation["response_time"])
        action = int(observation["last_action"])

        return (cpu, memory, response_time, action)

    def get_action(self, observation):
        """Choose action using epsilon-greedy strategy"""
        state_key = self.get_state_key(observation)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)

        if np.random.rand() < self.epsilon:
            return np.random.randint(
                1, self.n_actions
            )  # Perlu adanya random untuk mencoba action lain
        return np.argmax(self.q_table[state_key])

    def update_q_table(self, observation, action, reward, next_observation):
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

    def save_model(self, path: str):
        """
        Simpan Q-table (npz) + metadata (json).
        Contoh: agent.save_model('checkpoints/agent_k8s')
        Akan membuat:
        - checkpoints/agent_k8s.npz
        - checkpoints/agent_k8s.json
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Kemas Q-table jadi dua array: states (N,4) dan q (N,n_actions)
        if len(self.q_table) == 0:
            states = np.empty((0, 4), dtype=np.int32)
            qvals = np.empty((0, self.n_actions), dtype=np.float32)
        else:
            states = np.array(
                list(self.q_table.keys()), dtype=np.int32
            )  # (replicas, cpu, mem, rtime)
            qvals = np.vstack([self.q_table[s] for s in self.q_table]).astype(
                np.float32
            )

        np.savez_compressed(p.with_suffix(".npz"), states=states, q=qvals)

        meta = {
            "n_actions": self.n_actions,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": getattr(self, "epsilon", None),
            "epsilon_decay": getattr(self, "epsilon_decay", None),
            "epsilon_min": getattr(self, "epsilon_min", None),
        }
        with p.with_suffix(".json").open("w") as f:
            json.dump(meta, f, indent=2)

    def load_model(self, path: str):
        """
        Muat kembali Q-table + metadata.
        Contoh: agent.load_model('checkpoints/agent_k8s')
        """
        p = Path(path)

        # Load Q-table
        data = np.load(p.with_suffix(".npz"), allow_pickle=False)
        states = data["states"]
        qvals = data["q"]

        self.q_table = {
            tuple(int(x) for x in state_row): qvals[i].astype(np.float32)
            for i, state_row in enumerate(states)
        }

        # (Opsional) Muat metadata jika ada
        try:
            with p.with_suffix(".json").open() as f:
                meta = json.load(f)
            # Sinkronkan parameter dasar (jika ingin)
            self.n_actions = int(meta.get("n_actions", self.n_actions))
            self.learning_rate = float(meta.get("learning_rate", self.learning_rate))
            self.discount_factor = float(
                meta.get("discount_factor", self.discount_factor)
            )
            if meta.get("epsilon") is not None:
                self.epsilon = float(meta["epsilon"])
            if meta.get("epsilon_decay") is not None:
                self.epsilon_decay = float(meta["epsilon_decay"])
            if meta.get("epsilon_min") is not None:
                self.epsilon_min = float(meta["epsilon_min"])
        except FileNotFoundError:
            pass

    def save_checkpoint(
        self,
        dir_path: str = "checkpoints",
        episode: int | None = None,
        iteration: int | None = None,
        prefix: str = "qlearning",
    ):
        """
        Simpan checkpoint dengan nama berisi episode/iteration & timestamp.
        Menggunakan save_model() yang sudah ada.
        Return: path file .npz yang tersimpan.
        """
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = f"{prefix}"
        if episode is not None:
            name += f"_ep{episode}"
        if iteration is not None:
            name += f"_it{iteration}"
        name += f"_{ts}"
        base = str(Path(dir_path) / name)
        self.save_model(base)
        return f"{base}.npz"


# Q(S,A)←Q(S,A)+α(R+γQ(S′,A′)−Q(S,A))

"""
# Q(S,A) [Reward sebelumnya yang sudah tercatat di Q table]
# R [Reward terbaru yang didapat dari action yang diambil]

# @ [Learning rate, seberapa besar pengaruh reward terbaru terhadap Q(S,A)]

# Y [Discount factor, seberapa besar pengaruh reward di masa depan terhadap Q(S,A) Jadi
  di Q Learning juga memperhitungkan reward kemungkinan pada step berikutnya dari perolehan state(observation) yang sekarang]

# (S`,A`) [State dan action pada step berikutnya dari state(observation) yang sekarang]

# -Q(S,A) [Untuk mengurangi pengaruh reward sebelumnya yang sudah tercatat di Q table] lebih stabil
"""
