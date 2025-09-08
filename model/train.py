import atexit
import signal

import numpy as np
from agent import QLearningAgent
from environment import KubernetesEnv, setup_logger

logger = setup_logger("kubernetes_agent", log_level="INFO", log_to_file=True)


def train_agent(
    min_replicas=1,
    max_replicas=10,
    iteration=50,
    episodes=10,
    namespace="default",
    deployment_name="app-deployment",
    min_cpu=10,
    min_memory=10,
    max_cpu=90,
    max_memory=90,
    timeout=60,
    verbose=False,
    checkpoint_dir="checkpoints",
    save_on_interrupt=True,
    checkpoint_interval=5,
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
        verbose=verbose,
        logger=logger,
    )

    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.95,
    )

    current_episode = 0
    current_iteration = 0

    # atexit: simpan akhir kalau program berhenti normal
    def _final_save(checkpoint_dir: str):
        try:
            agent.save_checkpoint(
                checkpoint_dir,
                episode=current_episode,
                iteration=current_iteration,
                prefix="final",
            )
            logger.info("✅ Final checkpoint saved on exit.")
        except Exception as e:
            logger.exception(f"Failed to save final checkpoint: {e}")

    atexit.register(_final_save, checkpoint_dir=checkpoint_dir)

    # (opsional) handler SIGINT: tandai stop & biarkan loop menyelesaikan step
    stop_requested = {"flag": False}

    def _handle_sigint(signum, frame):
        stop_requested["flag"] = True
        logger.warning(
            "⚠️  Ctrl+C detected. Will checkpoint and stop at next safe point..."
        )

    if save_on_interrupt:
        signal.signal(signal.SIGINT, _handle_sigint)

    logger.info(f"Starting training for {episodes} episodes...")

    try:
        for episode in range(episodes):
            current_episode = episode + 1
            logger.info(f"\nEpisode {current_episode}/{episodes}")
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
                if verbose:
                    # Add observation details
                    logger.info("  🔍 Observation:")
                    logger.info(f"     CPU: {observation.get('cpu_usage', 0):.1f}%")
                    logger.info(
                        f"     Memory: {observation.get('memory_usage', 0):.1f}%"
                    )
                    logger.info(
                        f"     Response Time: {observation.get('response_time', 0):.1f}ms"
                    )
                    logger.info(
                        f"     Last Action: {observation.get('last_action', 'N/A')}"
                    )

                    # State key for debugging
                    state_key = agent.get_state_key(observation)
                    logger.info(f"  🗝️  State Key: {state_key}")

                    # Q-values for current state
                    if state_key in agent.q_table:
                        q_values = agent.q_table[state_key]
                        max_q = np.max(q_values)
                        best_action = np.argmax(q_values)
                        logger.info(
                            f"  🧠 Q-Values: Max={max_q:.3f}, Best Action={best_action}"
                        )

                    logger.info("----------------------------------------")

                if stop_requested["flag"]:
                    path = agent.save_checkpoint(
                        checkpoint_dir,
                        episode=current_episode,
                        iteration=current_iteration,
                        prefix="interrupt",
                    )
                    logger.warning(f"💾 Checkpoint saved due to Ctrl+C: {path}")
                    return agent, env  # stop training segera

                if terminated:
                    # decay epsilon di akhir episode
                    agent.epsilon = max(
                        agent.epsilon_min, agent.epsilon * agent.epsilon_decay
                    )
                    break

            logger.info(
                f"Episode {episode + 1} completed. Total reward: {total_reward}"
            )
            if checkpoint_interval and (current_episode % checkpoint_interval == 0):
                path = agent.save_checkpoint(
                    checkpoint_dir, episode=current_episode, iteration=current_iteration
                )
                logger.info(f"💾 Checkpoint saved: {path}")

    except KeyboardInterrupt:
        # fallback kalau SIGINT tidak tertangkap di loop
        path = agent.save_checkpoint(
            checkpoint_dir,
            episode=current_episode,
            iteration=current_iteration,
            prefix="interrupt",
        )
        logger.warning(f"💾 Checkpoint saved due to KeyboardInterrupt: {path}")
        # (opsional) re-raise jika ingin exit code 130
        # raise
    except Exception:
        # Simpan checkpoint kalau terjadi error lain
        path = agent.save_checkpoint(
            checkpoint_dir,
            episode=current_episode,
            iteration=current_iteration,
            prefix="error",
        )
        logger.exception(f"Error during training. Checkpoint saved: {path}")
        raise
    finally:
        logger.info("Training completed!")

    logger.info("Training completed!")
    return agent, env


if __name__ == "__main__":
    trained_agent, environment = train_agent(
        min_replicas=1,
        max_replicas=20,
        episodes=10,
        iteration=100,
        namespace="default",
        deployment_name="ecom-api",
        min_cpu=10,
        min_memory=10,
        max_cpu=90,
        max_memory=90,
        timeout=120,
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
    trained_agent.save_model("model/q_learning_model.npz")
