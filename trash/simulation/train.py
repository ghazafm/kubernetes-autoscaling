#!/usr/bin/env python3
"""
Kubernetes Autoscaling DQN Training Script

This script trains a Deep Q-Network (DQN) model for Kubernetes autoscaling.
It supports resuming training from previously saved models.

Usage Examples:
    # Start new training
    python train.py

    # Resume training from interrupted model
    python train.py --resume training/interrupted_model/model

    # Start new training with custom parameters
    python train.py --timesteps 100000 --action-step 10 --max-replicas 30

    # Resume training with different timesteps
    python train.py -r training/interrupted_model/model -t 25000

    # Full configuration with custom hyperparameters
    python train.py --timesteps 500000 --action-step 10 --max-replicas 70 \\
                   --min-cpu 20 --max-cpu 70 --min-memory 20 --max-memory 70 \\
                   --learning-rate 3e-4 --buffer-size 100000 --batch-size 64 \\
                   --gamma 0.95 --exploration-fraction 0.3 --verbose

    # Production training with context-aware waste checking
    python train.py --deployment-name nodejs-deployment --namespace production \\
                   --timesteps 500000 --iterations 200 --timeout 180 \\
                   --waste-check-mode context_aware --max-replicas 70 \\
                   --learning-rate 1e-4 --exploration-initial-eps 0.8
"""

import argparse
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

import urllib3
from simulation_environment import K8sAutoscalerEnv
from stable_baselines3 import DQN

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def parse_arguments():
    """Parse command line arguments for training configuration"""
    parser = argparse.ArgumentParser(
        description="Train or resume DQN model for Kubernetes autoscaling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--resume",
        "-r",
        type=str,
        help="Path to saved model to resume training from "
        "(e.g., 'training/interrupted_model/model')",
    )

    parser.add_argument(
        "--timesteps",
        "-t",
        type=int,
        default=50000,
        help="Number of timesteps to train for",
    )

    parser.add_argument(
        "--action-step",
        "-a",
        type=int,
        default=5,
        help="Action step size for replica scaling",
    )

    parser.add_argument(
        "--max-replicas", "-m", type=int, default=30, help="Maximum number of replicas"
    )

    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=100,
        help="Number of iterations per episode",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout for Kubernetes operations in seconds",
    )

    parser.add_argument(
        "--deployment-name",
        "-d",
        type=str,
        default="nodejs-deployment",
        help="Name of the Kubernetes deployment to scale",
    )

    parser.add_argument(
        "--namespace", "-n", type=str, default="default", help="Kubernetes namespace"
    )

    parser.add_argument(
        "--waste-check-mode",
        "-w",
        type=str,
        default="adaptive",
        choices=["fixed", "min_plus_one", "adaptive", "percentage", "context_aware"],
        help="Waste check threshold mode",
    )

    # CPU/Memory threshold arguments
    parser.add_argument(
        "--min-cpu",
        type=float,
        default=20.0,
        help="Minimum CPU target percentage (default: 20.0)",
    )

    parser.add_argument(
        "--max-cpu",
        type=float,
        default=85.0,
        help="Maximum CPU target percentage (default: 85.0)",
    )

    parser.add_argument(
        "--min-memory",
        type=float,
        default=20.0,
        help="Minimum memory target percentage (default: 20.0)",
    )

    parser.add_argument(
        "--max-memory",
        type=float,
        default=85.0,
        help="Maximum memory target percentage (default: 85.0)",
    )

    # Environment behavior arguments
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging and detailed output",
    )

    parser.add_argument(
        "--min-replicas",
        type=int,
        default=2,
        help="Minimum number of replicas (default: 2)",
    )

    # DQN hyperparameter arguments
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for DQN (default: 1e-4)",
    )

    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1_000_000,
        help="Experience replay buffer size (default: 1_000_000)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for future rewards (default: 0.99)",
    )

    parser.add_argument(
        "--exploration-fraction",
        type=float,
        default=0.1,
        help="Fraction of training for exploration (default: 0.1)",
    )

    parser.add_argument(
        "--exploration-initial-eps",
        type=float,
        default=1.0,
        help="Initial exploration rate (default: 1.0)",
    )

    parser.add_argument(
        "--exploration-final-eps",
        type=float,
        default=0.05,
        help="Final exploration rate (default: 0.05)",
    )

    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=10000,
        help="Target network update interval (default: 10000)",
    )

    parser.add_argument(
        "--learning-starts",
        type=int,
        default=100,
        help="Number of steps before learning starts (default: 100)",
    )

    parser.add_argument(
        "--train-freq",
        type=int,
        default=4,
        help="Training frequency (default: 4)",
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="Soft update coefficient for target network (default: 1.0)",
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=10.0,
        help="Maximum gradient norm for clipping (default: 10.0)",
    )

    return parser.parse_args()


model = None
env = None


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    logging.info("\n\n🛑 Training interrupted by user (Ctrl+C)")
    if model is not None:
        logging.info("💾 Saving model before exit...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path("training") / f"interrupted_model_{timestamp}"
        save_path.mkdir(parents=True, exist_ok=True)
        model.save(save_path / "model")
        logging.info(f"✅ Model saved to: {save_path}")

    if env is not None:
        logging.info("🔒 Closing environment...")
        env.close()

    logging.info("👋 Goodbye!")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def create_environment(args):
    """Create the K8s autoscaler environment"""
    return K8sAutoscalerEnv(
        min_replicas=args.min_replicas,
        max_replicas=args.max_replicas,
        iteration=args.iterations,
        namespace=args.namespace,
        deployment_name=args.deployment_name,
        min_cpu=args.min_cpu,
        min_memory=args.min_memory,
        max_cpu=args.max_cpu,
        max_memory=args.max_memory,
        verbose=args.verbose,
        action_step=args.action_step,
        timeout=args.timeout,
        waste_check_mode=args.waste_check_mode,
    )


def load_or_create_model(args, env):
    """Load existing model or create new one"""
    if args.resume:
        model_path = Path(args.resume)
        if not model_path.exists():
            logging.error(f"❌ Model file not found: {model_path}")
            sys.exit(1)

        logging.info(f"🔄 Resuming training from: {model_path}")
        try:
            model = DQN.load(
                model_path,
                env=env,
                verbose=1,
                tensorboard_log=str(Path("training") / "logs" / "DQN"),
            )
            logging.info("✅ Model loaded successfully!")
            return model
        except Exception as e:
            logging.error(f"❌ Failed to load model: {e}")
            sys.exit(1)
    else:
        logging.info("🆕 Creating new model...")
        return DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=args.train_freq,
            gradient_steps=1,
            target_update_interval=args.target_update_interval,
            exploration_fraction=args.exploration_fraction,
            exploration_initial_eps=args.exploration_initial_eps,
            exploration_final_eps=args.exploration_final_eps,
            max_grad_norm=args.max_grad_norm,
            verbose=1,
            tensorboard_log=str(Path("training") / "logs" / "DQN"),
        )


def display_training_info(args, env):
    """Display training configuration information"""
    logging.info("🚀 Starting training...")
    logging.info(f"📊 Action space: {env.action_space}")
    logging.info(f"⏱️  Timesteps: {args.timesteps:,}")
    logging.info(f"🔧 Action step: {args.action_step}")
    logging.info(f"📏 Replicas: {args.min_replicas}-{args.max_replicas}")
    logging.info(f"🎯 CPU target: {args.min_cpu}%-{args.max_cpu}%")
    logging.info(f"🧠 Memory target: {args.min_memory}%-{args.max_memory}%")
    logging.info(f"🔄 Iterations per episode: {args.iterations}")
    logging.info(f"⏰ Timeout: {args.timeout}s")
    logging.info(f"🎯 Deployment: {args.deployment_name} (namespace: {args.namespace})")
    logging.info(
        f"🎛️  Waste check mode: {args.waste_check_mode} "
        f"(threshold: {env.waste_check_threshold})"
    )
    logging.info(f"🔬 Learning rate: {args.learning_rate}")
    logging.info(f"💾 Buffer size: {args.buffer_size:,}")
    logging.info(f"📦 Batch size: {args.batch_size}")
    logging.info(f"🎲 Gamma: {args.gamma}")
    logging.info(
        f"🔍 Exploration: {args.exploration_initial_eps} → "
        f"{args.exploration_final_eps} (fraction: {args.exploration_fraction})"
    )
    if args.resume:
        logging.info(f"🔄 Resuming from: {args.resume}")
    logging.info("⏹️  Press Ctrl+C to stop training and save model")
    logging.info("=" * 55)


def train_model(model, args):
    """Train the model and save it"""
    try:
        model.learn(total_timesteps=args.timesteps, progress_bar=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path("training") / f"completed_model_{timestamp}"
        save_path.mkdir(parents=True, exist_ok=True)
        model.save(save_path / "model")
        logging.info(f"✅ Training completed! Model saved to: {save_path}")
    except KeyboardInterrupt:
        signal_handler(None, None)


def test_model(model, env, args):
    """Test the trained model"""
    logging.info("\n🧪 Testing the trained model for 10 steps...")
    obs, _ = env.reset()

    for step in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        logging.info(
            f"Step {step + 1}: Action={action - args.action_step:+d}, "
            f"Reward={reward:.1f}, "
            f"Replicas={info['current_replicas']}, "
            f"CPU={info['cpu_usage']:.1f}%, Mem={info['memory_usage']:.1f}%"
        )

        if terminated or truncated:
            break


def main():
    """Main training function"""

    args = parse_arguments()
    env = create_environment(args)
    model = load_or_create_model(args, env)
    display_training_info(args, env)
    train_model(model, args)
    test_model(model, env, args)
    env.close()
    logging.info("🎉 Done!")


if __name__ == "__main__":
    main()
