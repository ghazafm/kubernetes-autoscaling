import logging

from environment import KubernetesEnv
from rl import Q
from utils import (
    log_verbose_details,
    normalize_endpoints,
)


def _run_training_episode(
    environment: KubernetesEnv,
    agent: Q,
    episode: int,
    episodes: int,
    verbose: bool = False,
    logger: logging.Logger = logging.getLogger(__name__),  # noqa: B008
):
    """Run a single training episode"""
    logger.info(f"\nEpisode {episode + 1}/{episodes}")
    observation = environment.reset()
    total_reward = 0

    while True:
        action = agent.get_action(observation)
        next_observation, reward, terminated, info = environment.step(action)
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

        if terminated:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
            break

    return False, total_reward


def train_agent(
    agent: Q,
    environment: KubernetesEnv,
    episodes: int = 10,
    verbose: bool = False,
    metrics_endpoints_method: list[tuple[str, str]] = (("/", "GET"), ("/docs", "GET")),
    logger: logging.Logger = logging.getLogger(__name__),  # noqa: B008
):
    """Train the Q-learning agent on the Kubernetes environment"""
    metrics_endpoints_method = normalize_endpoints(metrics_endpoints_method)

    logger.info(f"Starting training for {episodes} episodes...")

    try:
        for episode in range(episodes):
            should_stop, total_reward = _run_training_episode(
                environment,
                agent,
                episode,
                episodes,
                verbose,
                logger,
            )

            if should_stop:
                return agent, environment

            logger.info(
                f"Episode {episode + 1} completed. Total reward: {total_reward}"
            )

    except Exception:
        logger.exception("Error during training.")
        raise
    finally:
        logger.info("Training completed!")

    return agent, environment
