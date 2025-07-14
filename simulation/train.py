import logging
import signal
import sys
from pathlib import Path

import urllib3
from simulation_environment import K8sAutoscalerEnv
from stable_baselines3 import DQN

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


model = None
env = None


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
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


env = K8sAutoscalerEnv(
    min_replicas=1,
    max_replicas=50,
    iteration=100,
    namespace="default",
    deployment_name="nodejs-deployment",
    min_cpu=20,
    min_memory=20,
    max_cpu=85,
    max_memory=85,
    verbose=True,
    action_step=25,
    timeout=120,
)


model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    verbose=1,
)

logging.info("🚀 Starting training...")
logging.info(
    f"📊 Action space: {env.action_space} (5 actions: -2, -1, 0, +1, +2 replicas)"
)
logging.info("⏹️  Press Ctrl+C to stop training and save model")
logging.info("=" * 60)

try:
    model.learn(total_timesteps=50000, progress_bar=True)

    save_path = Path("training") / "completed_model"
    save_path.mkdir(parents=True, exist_ok=True)
    model.save(save_path / "model")
    logging.info(f"✅ Training completed! Model saved to: {save_path}")

except KeyboardInterrupt:
    signal_handler(None, None)


logging.info("\n🧪 Testing the trained model for 10 steps...")
obs, _ = env.reset()

for step in range(10):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    logging.info(
        f"Step {step + 1}: Action={action - 25:+d}, Reward={reward:.1f}, "
        f"Replicas={info['current_replicas']}, "
        f"CPU={info['cpu_usage']:.1f}%, Mem={info['memory_usage']:.1f}%"
    )

    if terminated or truncated:
        break

env.close()
logging.info("🎉 Done!")
