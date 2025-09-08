import logging

import numpy as np
from environment import KubernetesEnv


class QLearningAgent:
    def __init__(
        self,
        n_actions=101,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_prob=0.1,
    ):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.q_table = {}

    def get_state_key(self, observation):
        """Convert observation to a hashable state key"""
        replicas = int(observation["replicas"])
        replicas = int(observation["replicas"])
        # cpu = int(observation["cpu_usage"] // 10)
        # memory = int(observation["memory_usage"] // 10)
        # response_time = int(observation["response_time"] // 100)
        cpu = int(observation["cpu_usage"])
        memory = int(observation["memory_usage"])
        response_time = int(observation["response_time"])

        return (replicas, cpu, memory, response_time)

    def get_action(self, observation):
        """Choose action using epsilon-greedy strategy"""
        state_key = self.get_state_key(observation)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)

        if np.random.rand() < self.exploration_prob:
            return np.random.randint(0, self.n_actions)
        return np.argmax(self.q_table[state_key])

    def update_q_table(self, observation, action, reward, next_observation):
        """Update Q-table using Q-learning algorithm"""
        state_key = self.get_state_key(observation)
        next_state_key = self.get_state_key(next_observation)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.n_actions)

        best_next_action = np.max(self.q_table[next_state_key])
        self.q_table[state_key][action] += self.learning_rate * (
            reward
            + self.discount_factor * best_next_action
            - self.q_table[state_key][action]
        )


def train_agent(episodes=10):
    """Train the Q-learning agent on the Kubernetes environment"""
    env = KubernetesEnv(
        min_replicas=1,
        max_replicas=10,
        iteration=50,
        namespace="default",
        deployment_name="your-deployment-name",
        verbose=True,
    )

    agent = QLearningAgent()

    logging.info(f"Starting training for {episodes} episodes...")

    for episode in range(episodes):
        logging.info(f"\nEpisode {episode + 1}/{episodes}")
        observation = env.reset()
        total_reward = 0

        while True:
            action = agent.get_action(observation)
            next_observation, reward, terminated, info = env.step(action)

            agent.update_q_table(observation, action, reward, next_observation)

            total_reward += reward
            observation = next_observation

            logging.info(
                f"Action: {action}, Reward: {reward}, Total Reward: {total_reward}"
            )

            if terminated:
                break

        logging.info(f"Episode {episode + 1} completed. Total reward: {total_reward}")

    logging.info("Training completed!")
    return agent, env


if __name__ == "__main__":
    trained_agent, environment = train_agent(episodes=5)

    logging.info(f"\nQ-table size: {len(trained_agent.q_table)} states")
    logging.info("Sample Q-values:")
    for _, (state, q_values) in enumerate(list(trained_agent.q_table.items())[:5]):
        max_q = np.max(q_values)
        best_action = np.argmax(q_values)
        logging.info(
            f"State {state}: Best action = {best_action}, Max Q-value = {max_q:.3f}"
        )
