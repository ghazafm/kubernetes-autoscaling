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

    # Full configuration
    python train.py --resume training/interrupted_model/model \
                   --timesteps 75000 \
                   --action-step 15 \
                   --max-replicas 40 \
                   --deployment-name my-app \
                   --namespace production
"""

import argparse
import logging
import signal
import sys
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
        default=25,
        help="Action step size for replica scaling",
    )

    parser.add_argument(
        "--max-replicas", "-m", type=int, default=50, help="Maximum number of replicas"
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

    return parser.parse_args()


model = None
env = None


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global model, env
    logging.info("\n\n🛑 Training interrupted by user (Ctrl+C)")
    if model is not None:
        logging.info("💾 Saving model before exit...")
        save_path = Path("training") / "interrupted_model"
        save_path.mkdir(parents=True, exist_ok=True)
        model.save(save_path / "model")
        logging.info(f"✅ Model saved to: {save_path}")

    if env is not None:
        logging.info("🔒 Closing environment...")
        env.close()

    logging.info("👋 Goodbye!")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def main():
    """Main training function"""
    global model, env

    args = parse_arguments()

    env = K8sAutoscalerEnv(
        min_replicas=1,
        max_replicas=args.max_replicas,
        iteration=args.iterations,
        namespace=args.namespace,
        deployment_name=args.deployment_name,
        min_cpu=20,
        min_memory=20,
        max_cpu=85,
        max_memory=85,
        verbose=True,
        action_step=args.action_step,
        timeout=args.timeout,
        waste_check_mode=args.waste_check_mode,
    )

    # Create or load model
    if args.resume:
        # Resume training from saved model
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
        except Exception as e:
            logging.error(f"❌ Failed to load model: {e}")
            sys.exit(1)
    else:
        # Create new model
        logging.info("🆕 Creating new model...")
        model = DQN(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log=str(Path("training") / "logs" / "DQN"),
        )

    # Display training information
    logging.info("🚀 Starting training...")
    logging.info(f"📊 Action space: {env.action_space}")
    logging.info(f"⏱️  Timesteps: {args.timesteps:,}")
    logging.info(f"🔧 Action step: {args.action_step}")
    logging.info(f"📏 Max replicas: {args.max_replicas}")
    logging.info(f"🔄 Iterations per episode: {args.iterations}")
    logging.info(f"🎯 Deployment: {args.deployment_name} (namespace: {args.namespace})")
    logging.info(
        f"🎛️  Waste check mode: {args.waste_check_mode} "
        f"(threshold: {env.waste_check_threshold})"
    )
    if args.resume:
        logging.info(f"🔄 Resuming from: {args.resume}")
    logging.info("⏹️  Press Ctrl+C to stop training and save model")
    logging.info("=" * 60)

    try:
        model.learn(total_timesteps=args.timesteps, progress_bar=True)

        save_path = Path("training") / "completed_model"
        save_path.mkdir(parents=True, exist_ok=True)
        model.save(save_path / "model")
        logging.info(f"✅ Training completed! Model saved to: {save_path}")

    except KeyboardInterrupt:
        signal_handler(None, None)

    # Test the trained model
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

    env.close()
    logging.info("🎉 Done!")


if __name__ == "__main__":
    main()
