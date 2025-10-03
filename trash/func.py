import logging
import sys
import time
from pathlib import Path

from environment import KubernetesEnv
from rl import Q
from utils import (
    log_verbose_details,
    normalize_endpoints,
    setup_logger,
)

trained_agent = None
start_time = None
note = None
logger = setup_logger("kubernetes_agent", log_level="INFO", log_to_file=False)


def signal_handler(sig, frame):
    """Handle interrupt signals (Ctrl+C)"""
    logger.warning("\nInterrupt signal received. Saving model...")

    current_agent = trained_agent
    current_start_time = globals().get("start_time")
    current_note = globals().get("note")

    if current_agent and current_start_time and current_note:
        try:
            ext = ".pth" if current_agent.agent_type.upper() == "DQN" else ".pkl"
            model_type = (
                "dqn" if current_agent.agent_type.upper() == "DQN" else "qlearning"
            )

            interrupted_dir = Path(
                f"model/{model_type}/{current_note}_{current_start_time}/interrupted"
            )
            interrupted_dir.mkdir(parents=True, exist_ok=True)

            timestamp = int(time.time())
            episode_count = current_agent.episodes_trained
            filename = f"interrupted_episode_{episode_count}_{timestamp}{ext}"
            interrupted_path = interrupted_dir / filename

            current_agent.save_model(
                str(interrupted_path), current_agent.episodes_trained
            )
            logger.info(f"✅ Model saved to: {interrupted_path}")
            logger.info(f"Episodes completed: {current_agent.episodes_trained}")
        except Exception as e:
            logger.error(f"Failed to save model on interrupt: {e}")
    else:
        logger.warning(
            "⚠️  No model to save (training not started or variables not set)"
        )

    logger.info("Exiting...")
    sys.exit(0)


def _run_training_episode(
    environment: KubernetesEnv,
    agent: Q,
    episode: int,
    episodes: int,
    verbose: bool = False,
    logger: logging.Logger = logging.getLogger(__name__),  # noqa: B008
) -> tuple[bool, float]:
    """Run a single training episode"""
    agent.add_episode_count()
    logger.info(f"\nEpisode {episode + 1}/{episodes}")
    logger.info(f"Total episodes trained: {agent.episodes_trained}")
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
    note: str = "default",
    start_time: int = int(time.time()),
    resume: bool = False,
    resume_path: str = "",
    reset_epsilon: bool = True,
    logger: logging.Logger = logging.getLogger(__name__),  # noqa: B008
) -> tuple[Q, KubernetesEnv]:
    """Train the Q-learning agent on the Kubernetes environment"""

    # Set global variables for signal handler
    global trained_agent
    trained_agent = agent
    globals()["start_time"] = start_time
    globals()["note"] = note

    metrics_endpoints_method = normalize_endpoints(metrics_endpoints_method)

    logger.info(f"Starting training for {episodes} episodes...")
    if resume and resume_path:
        try:
            epsilon_start = agent.epsilon
            agent.load_model(resume_path)
            logger.info(f"Resumed training from model at: {resume_path}")
            if reset_epsilon:
                agent.epsilon = epsilon_start
                logger.info("Epsilon reset to starting value.")
        except Exception as e:
            logger.error(f"Failed to load model from {resume_path}: {e}")
            raise

    try:
        total_reward_init = 0
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
            if total_reward > total_reward_init:
                total_reward_init = total_reward
                if hasattr(agent, "save_model"):
                    # Determine correct file extension
                    ext = ".pth" if agent.agent_type.upper() == "DQN" else ".pkl"
                    model_type = (
                        "dqn" if agent.agent_type.upper() == "DQN" else "qlearning"
                    )
                    checkpoint_path = (
                        f"model/{model_type}/{note}_{start_time}/checkpoints/"
                        f"episode_{episode}_total_{total_reward_init}{ext}"
                    )
                    agent.save_model(checkpoint_path, episode + 1)
                    logger.info(
                        f"New best model saved with total reward: {total_reward_init}"
                    )

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user (KeyboardInterrupt)!")
        logger.info("Saving current model state...")

        if hasattr(agent, "save_model"):
            ext = ".pth" if agent.agent_type.upper() == "DQN" else ".pkl"
            model_type = "dqn" if agent.agent_type.upper() == "DQN" else "qlearning"
            interrupted_dir = Path(
                f"model/{model_type}/{note}_{start_time}/interrupted"
            )
            interrupted_dir.mkdir(parents=True, exist_ok=True)

            timestamp = int(time.time())
            interrupted_path = (
                interrupted_dir
                / f"interrupted_episode_{agent.episodes_trained}_{timestamp}{ext}"
            )

            agent.save_model(str(interrupted_path), agent.episodes_trained)
            logger.info(f"✅ Model saved to: {interrupted_path}")
            logger.info(f"Episodes completed: {agent.episodes_trained}")
        raise
    except Exception:
        logger.exception("Error during training.")
        raise
    finally:
        logger.info("Training completed!")

    return agent, environment
