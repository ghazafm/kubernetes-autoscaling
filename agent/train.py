import time

import numpy as np
from database import InfluxDB
from environment import (
    KubernetesEnv,
)
from model import QLearningAgent
from utils import (
    log_verbose_details,
    setup_interruption_handlers,
    setup_logger,
)

logger = setup_logger("kubernetes_agent", log_level="INFO", log_to_file=True)


def _run_training_episode(
    env: KubernetesEnv,
    agent: QLearningAgent,
    episode: int,
    episodes: int,
    stop_requested: dict,
    current_episode: list[int],
    current_iteration: list[int],
    checkpoint_dir: str,
    verbose: bool = False,
):
    """Run a single training episode"""
    current_episode[0] = episode + 1
    logger.info(f"\nEpisode {current_episode[0]}/{episodes}")
    observation = env.reset()
    total_reward = 0

    while True:
        action = agent.get_action(observation)
        next_observation, reward, terminated, info = env.step(action)
        agent.update_q_table(observation, action, reward, next_observation)

        total_reward += reward
        observation = next_observation

        logger.info(
            f"Action: {action}, Reward: {reward}, Total Reward: {total_reward} "
            f"| Iteration: {info['iteration']}"
        )

        log_verbose_details(
            observation=observation, agent=agent, verbose=verbose, logger=logger
        )

        if stop_requested["flag"]:
            path = agent.save_checkpoint(
                checkpoint_dir,
                episode=current_episode[0],
                iteration=current_iteration[0],
                prefix="interrupt",
            )
            logger.warning(f"💾 Checkpoint saved due to Ctrl+C: {path}")
            return True, total_reward  # signal to stop training

        if terminated:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
            break

    return False, total_reward


def train_agent(
    min_replicas: int = 1,
    max_replicas: int = 10,
    iteration: int = 50,
    episodes: int = 10,
    namespace: str = "default",
    deployment_name: str = "app-deployment",
    min_cpu: int = 10,
    min_memory: int = 10,
    max_cpu: int = 90,
    max_memory: int = 90,
    timeout: int = 120,
    wait_time: int = 30,
    verbose: bool = False,
    checkpoint_dir: str = "checkpoints",
    save_on_interrupt: bool = True,
    checkpoint_interval: int = 5,
    influxdb: InfluxDB = None,
):
    """Train the Q-learning agent on the Kubernetes environment"""
    env = KubernetesEnv(
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        iteration=iteration,
        namespace=namespace,
        deployment_name=deployment_name,
        min_cpu=min_cpu,
        min_memory=min_memory,
        max_cpu=max_cpu,
        max_memory=max_memory,
        timeout=timeout,
        wait_time=wait_time,
        verbose=verbose,
        logger=logger,
        influxdb=influxdb,
    )

    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.95,
    )

    current_episode = [0]
    current_iteration = [0]

    stop_requested = setup_interruption_handlers(
        agent=agent,
        current_episode=current_episode,
        current_iteration=current_iteration,
        checkpoint_dir=checkpoint_dir,
        save_on_interrupt=save_on_interrupt,
        logger=logger,
    )

    logger.info(f"Starting training for {episodes} episodes...")

    try:
        for episode in range(episodes):
            should_stop, total_reward = _run_training_episode(
                env,
                agent,
                episode,
                episodes,
                stop_requested,
                current_episode,
                current_iteration,
                checkpoint_dir,
                verbose,
            )

            if should_stop:
                return agent, env

            logger.info(
                f"Episode {episode + 1} completed. Total reward: {total_reward}"
            )

            if checkpoint_interval and (current_episode[0] % checkpoint_interval == 0):
                path = agent.save_checkpoint(
                    checkpoint_dir,
                    episode=current_episode[0],
                    iteration=current_iteration[0],
                )
                logger.info(f"💾 Checkpoint saved: {path}")

    except KeyboardInterrupt:
        path = agent.save_checkpoint(
            checkpoint_dir,
            episode=current_episode[0],
            iteration=current_iteration[0],
            prefix="interrupt",
        )
        logger.warning(f"💾 Checkpoint saved due to KeyboardInterrupt: {path}")
    except Exception:
        path = agent.save_checkpoint(
            checkpoint_dir,
            episode=current_episode[0],
            iteration=current_iteration[0],
            prefix="error",
        )
        logger.exception(f"Error during training. Checkpoint saved: {path}")
        raise
    finally:
        logger.info("Training completed!")

    return agent, env


if __name__ == "__main__":
    trained_agent, environment = train_agent(
        min_replicas=1,
        max_replicas=15,
        episodes=10,
        iteration=10,
        namespace="default",
        deployment_name="ecom-api",
        min_cpu=10,
        min_memory=10,
        max_cpu=90,
        max_memory=90,
        timeout=120,
        wait_time=1,
        verbose=True,
    )

    logger.info(f"\nQ-table size: {len(trained_agent.q_table)} states")
    logger.info("Sample Q-values:")
    for _, (state, q_values) in enumerate(list(trained_agent.q_table.items())[:5]):
        max_q = np.max(q_values)
        best_action = np.argmax(q_values)
        logger.info(
            f"State {state}: Best action = {best_action}, Max Q-value = {max_q:.3f}"
        )
    # Save the trained Q-table
    trained_agent.save_model(f"model/{time.time()}.npz")
