import ast
import os
import time
from pathlib import Path

import numpy as np
from database import InfluxDB
from dotenv import load_dotenv
from environment import (
    KubernetesEnv,
)
from rl import DQN, Q
from trainer import Trainer
from utils import (
    setup_logger,
)

load_dotenv()


def find_latest_checkpoint(model_dir: Path) -> Path | None:
    """Find the most recent checkpoint in the model directory."""
    if not model_dir.exists():
        return None

    checkpoints = list(model_dir.glob("**/*.pth")) + list(model_dir.glob("**/*.pkl"))
    if not checkpoints:
        return None

    # Sort by modification time
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]


if __name__ == "__main__":
    # Configuration
    max_training_retries = int(os.getenv("MAX_TRAINING_RETRIES", "3"))
    auto_resume = ast.literal_eval(os.getenv("AUTO_RESUME", "True"))

    start_time = int(time.time())
    logger = setup_logger(
        "kubernetes_agent", log_level=os.getenv("LOG_LEVEL", "INFO"), log_to_file=True
    )

    logger.info("=" * 80)
    logger.info("KUBERNETES AUTOSCALING RL TRAINING")
    logger.info(f"Training Start Time: {start_time}")
    logger.info(f"Auto-resume enabled: {auto_resume}")
    logger.info(f"Max retries on failure: {max_training_retries}")
    logger.info("=" * 80)

    Influxdb = InfluxDB(
        logger=logger,
        url=os.getenv("INFLUXDB_URL", "http://localhost:8086"),
        token=os.getenv("INFLUXDB_TOKEN", "my-token"),
        org=os.getenv("INFLUXDB_ORG", "my-org"),
        bucket=os.getenv("INFLUXDB_BUCKET", "my-bucket"),
    )
    try:
        metrics_endpoints_str = os.getenv(
            "METRICS_ENDPOINTS_METHOD", "[['/', 'GET'], ['/docs', 'GET']]"
        )
        metrics_endpoints_method = ast.literal_eval(metrics_endpoints_str)
    except (ValueError, SyntaxError):
        logger.warning("Invalid METRICS_ENDPOINTS_METHOD format, using default")
        metrics_endpoints_method = [["/", "GET"], ["/docs", "GET"]]

    env = KubernetesEnv(
        min_replicas=int(os.getenv("MIN_REPLICAS", "1")),
        max_replicas=int(os.getenv("MAX_REPLICAS", "12")),
        iteration=int(os.getenv("ITERATION", "10")),
        namespace=os.getenv("NAMESPACE", "default"),
        deployment_name=os.getenv("DEPLOYMENT_NAME", "ecom-api"),
        min_cpu=int(os.getenv("MIN_CPU", "10")),
        min_memory=int(os.getenv("MIN_MEMORY", "10")),
        max_cpu=int(os.getenv("MAX_CPU", "90")),
        max_memory=int(os.getenv("MAX_MEMORY", "90")),
        max_response_time=float(os.getenv("MAX_RESPONSE_TIME", "100.0")),
        timeout=int(os.getenv("TIMEOUT", "120")),
        wait_time=int(os.getenv("WAIT_TIME", "1")),
        verbose=True,
        logger=logger,
        influxdb=Influxdb,
        prometheus_url=os.getenv("PROMETHEUS_URL", "http://localhost:1234/prom"),
        metrics_endpoints_method=metrics_endpoints_method,
        metrics_interval=int(os.getenv("METRICS_INTERVAL", "15")),
        metrics_quantile=float(os.getenv("METRICS_QUANTILE", "0.90")),
        max_scaling_retries=int(os.getenv("MAX_SCALING_RETRIES", "1000")),
        response_time_weight=float(os.getenv("RESPONSE_TIME_WEIGHT", "1.0")),
        error_rate_weight=float(os.getenv("ERROR_RATE_WEIGHT", "1.0")),
        cpu_memory_weight=float(os.getenv("CPU_MEMORY_WEIGHT", "0.5")),
        cost_weight=float(os.getenv("COST_WEIGHT", "0.3")),
    )

    choose_algorithm = os.getenv("ALGORITHM", "Q").upper()
    note = os.getenv("NOTE", "default")

    # Check for existing checkpoints if auto-resume is enabled
    resume_from_checkpoint = ast.literal_eval(os.getenv("RESUME", "False"))
    resume_path = os.getenv("RESUME_PATH", "")

    if auto_resume and not resume_from_checkpoint:
        model_dir = Path(f"model/{choose_algorithm.lower()}")
        latest_checkpoint = find_latest_checkpoint(model_dir)
        if latest_checkpoint:
            resume_from_checkpoint = True
            resume_path = str(latest_checkpoint)
            logger.info(f"🔄 Auto-resume: Found checkpoint at {resume_path}")

    # Training loop with retry mechanism
    training_attempt = 0
    training_successful = False

    while training_attempt < max_training_retries and not training_successful:
        training_attempt += 1

        if training_attempt > 1:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"RETRY ATTEMPT {training_attempt}/{max_training_retries}")
            logger.info(f"{'=' * 80}\n")

            # Re-check for checkpoints after failed attempt
            if auto_resume:
                model_dir = Path(f"model/{choose_algorithm.lower()}")
                latest_checkpoint = find_latest_checkpoint(model_dir)
                if latest_checkpoint:
                    resume_from_checkpoint = True
                    resume_path = str(latest_checkpoint)
                    logger.info(f"🔄 Resuming from: {resume_path}")

        try:
            # Initialize agent
            if choose_algorithm == "Q":
                algorithm = Q(
                    learning_rate=float(os.getenv("LEARNING_RATE", None)),
                    discount_factor=float(os.getenv("DISCOUNT_FACTOR", None)),
                    epsilon_start=float(os.getenv("EPSILON_START", None)),
                    epsilon_decay=float(os.getenv("EPSILON_DECAY", None)),
                    epsilon_min=float(os.getenv("EPSILON_MIN", None)),
                    created_at=start_time,
                    logger=logger,
                )
            elif choose_algorithm == "DQN":
                algorithm = DQN(
                    learning_rate=float(os.getenv("LEARNING_RATE", None)),
                    discount_factor=float(os.getenv("DISCOUNT_FACTOR", None)),
                    epsilon_start=float(os.getenv("EPSILON_START", None)),
                    epsilon_decay=float(os.getenv("EPSILON_DECAY", None)),
                    epsilon_min=float(os.getenv("EPSILON_MIN", None)),
                    device=os.getenv("DEVICE", None),
                    buffer_size=int(os.getenv("BUFFER_SIZE", None)),
                    batch_size=int(os.getenv("BATCH_SIZE", None)),
                    target_update_freq=int(os.getenv("TARGET_UPDATE_FREQ", None)),
                    grad_clip_norm=float(os.getenv("GRAD_CLIP_NORM", None)),
                    created_at=start_time,
                    logger=logger,
                )
            else:
                raise ValueError(f"Unsupported algorithm: {choose_algorithm}")

            trainer = Trainer(
                agent=algorithm,
                env=env,
                logger=logger,
                resume=resume_from_checkpoint,
                resume_path=resume_path,
                reset_epsilon=ast.literal_eval(os.getenv("RESET_EPSILON", "True")),
                change_epsilon_decay=float(os.getenv("EPSILON_DECAY", None)),
                periodic_epsilon_reset=ast.literal_eval(
                    os.getenv("PERIODIC_EPSILON_RESET", "False")
                ),
                epsilon_reset_interval=int(os.getenv("EPSILON_RESET_INTERVAL", "200")),
                epsilon_reset_value=float(os.getenv("EPSILON_RESET_VALUE", "0.3")),
            )

            # Start training
            logger.info("🚀 Starting training...")
            trainer.train(
                episodes=int(os.getenv("EPISODES", None)),
                note=note,
                start_time=start_time,
            )

            training_successful = True
            logger.info("✅ Training completed successfully!")

        except KeyboardInterrupt:
            logger.warning("⚠️  Training interrupted by user.")
            raise

        except Exception as e:
            logger.error(
                f"❌ Training failed on attempt {training_attempt}/"
                f"{max_training_retries}"
            )
            logger.error(f"Error: {type(e).__name__}: {e}")

            if training_attempt < max_training_retries:
                wait_time = min(60 * training_attempt, 300)  # Max 5 minutes
                logger.info(f"⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"❌ All {max_training_retries} attempts failed.")
                raise

    # Training completed - save final model and display stats
    if training_successful and hasattr(trainer, "agent"):
        if hasattr(trainer.agent, "q_table"):
            logger.info(f"\nQ-table size: {len(trainer.agent.q_table)} states")
            logger.info("Sample Q-values:")
            for _, (state, q_values) in enumerate(
                list(trainer.agent.q_table.items())[:5]
            ):
                max_q = np.max(q_values)
                best_action = np.argmax(q_values)
                logger.info(
                    f"State {state}: Best action = {best_action}, "
                    f"Max Q-value = {max_q:.3f}"
                )
        else:
            logger.info("\nDQN model trained (no Q-table to display)")

        model_type = "dqn" if trainer.agent.agent_type.upper() == "DQN" else "qlearning"
        model_dir = Path(f"model/{model_type}/{start_time}_{note}/final")
        model_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        if trainer.agent.agent_type.upper() == "DQN":
            model_file = model_dir / f"dqn_{timestamp}.pth"
        else:
            model_file = model_dir / f"qlearning_{timestamp}.pkl"

        trainer.agent.save_model(str(model_file), trainer.agent.episodes_trained)
        logger.info(f"✅ Final model saved to: {model_file}")
        logger.info("=" * 80)
        logger.info("TRAINING SESSION COMPLETE")
        logger.info("=" * 80)
