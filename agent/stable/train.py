import ast
import os
from datetime import datetime
from pathlib import Path

from database import InfluxDB
from dotenv import load_dotenv
from environment import KubernetesEnv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from utils import setup_logger

load_dotenv()

if __name__ == "__main__":
    now = datetime.now().strftime("%Y-%m-%d-%H-%M")
    logger, log_dir = setup_logger(
        "kubernetes_agent", log_level=os.getenv("LOG_LEVEL", "INFO"), log_to_file=True
    )

    influxdb = InfluxDB(
        logger=logger,
        url=os.getenv("INFLUXDB_URL", "http://localhost:8086"),
        token=os.getenv("INFLUXDB_TOKEN", "my-token"),
        org=os.getenv("INFLUXDB_ORG", "my-org"),
        bucket=os.getenv("INFLUXDB_BUCKET", "my-bucket"),
    )
    metrics_endpoints_method = ast.literal_eval(os.getenv("METRICS_ENDPOINTS_METHOD"))

    # Environment configuration
    iteration = int(os.getenv("ITERATION"))
    env = KubernetesEnv(
        min_replicas=int(os.getenv("MIN_REPLICAS")),
        max_replicas=int(os.getenv("MAX_REPLICAS")),
        iteration=iteration,
        namespace=os.getenv("NAMESPACE"),
        deployment_name=os.getenv("DEPLOYMENT_NAME"),
        min_cpu=float(os.getenv("MIN_CPU")),
        min_memory=float(os.getenv("MIN_MEMORY")),
        max_cpu=float(os.getenv("MAX_CPU")),
        max_memory=float(os.getenv("MAX_MEMORY")),
        max_response_time=float(os.getenv("MAX_RESPONSE_TIME")),
        timeout=int(os.getenv("TIMEOUT")),
        wait_time=int(os.getenv("WAIT_TIME")),
        logger=logger,
        influxdb=influxdb,
        prometheus_url=os.getenv("PROMETHEUS_URL"),
        metrics_endpoints_method=metrics_endpoints_method,
        metrics_interval=int(os.getenv("METRICS_INTERVAL")),
        metrics_quantile=float(os.getenv("METRICS_QUANTILE")),
        max_scaling_retries=int(os.getenv("MAX_SCALING_RETRIES")),
        weight_response_time=float(os.getenv("WEIGHT_RESPONSE_TIME")),
        weight_cost=float(os.getenv("WEIGHT_COST")),
        weight_error_rate=float(os.getenv("WEIGHT_ERROR_RATE", "0.3")),
        render_mode="human",
    )

    eval_env = KubernetesEnv(
        min_replicas=int(os.getenv("MIN_REPLICAS")),
        max_replicas=int(os.getenv("MAX_REPLICAS")) - 10,
        iteration=iteration,
        namespace=os.getenv("NAMESPACE"),
        deployment_name=os.getenv("DEPLOYMENT_NAME"),
        min_cpu=float(os.getenv("MIN_CPU")),
        min_memory=float(os.getenv("MIN_MEMORY")),
        max_cpu=float(os.getenv("MAX_CPU")),
        max_memory=float(os.getenv("MAX_MEMORY")),
        max_response_time=float(os.getenv("MAX_RESPONSE_TIME")),
        timeout=int(os.getenv("TIMEOUT")),
        wait_time=int(os.getenv("WAIT_TIME")),
        logger=logger,
        influxdb=influxdb,
        prometheus_url=os.getenv("PROMETHEUS_URL"),
        metrics_endpoints_method=[("/", "GET"), ("/docs", "GET")],
        metrics_interval=int(os.getenv("METRICS_INTERVAL")),
        metrics_quantile=float(os.getenv("METRICS_QUANTILE")),
        max_scaling_retries=int(os.getenv("MAX_SCALING_RETRIES")),
        weight_response_time=float(os.getenv("WEIGHT_RESPONSE_TIME")),
        weight_cost=float(os.getenv("WEIGHT_COST")),
        weight_error_rate=float(os.getenv("WEIGHT_ERROR_RATE", "0.3")),
        render_mode="human",
    )

    # Training configuration
    BASE_EPISODES = 10
    num_episodes = int(os.getenv("EPISODE", BASE_EPISODES))
    total_timesteps = num_episodes * iteration

    note = os.getenv("NOTE", "default")
    model_dir = Path(f"model/{now}_{note}")
    model_dir.mkdir(parents=True, exist_ok=True)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        policy_kwargs={"net_arch": [256, 256, 128]},
        learning_rate=1e-4,
        gamma=0.99,
        buffer_size=100_000,
        learning_starts=iteration * 3,
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=iteration,
        exploration_fraction=0.4,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        max_grad_norm=1,
        verbose=1,
        tensorboard_log=log_dir,
        seed=42,
        device="auto",
    )

    logger.info(
        f"Starting training: {num_episodes} episodes x {iteration} steps "
        f"= {total_timesteps} timesteps"
    )
    logger.info(f"Model directory: {model_dir}")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best_model"),
        log_path=str(model_dir / "eval_logs"),
        eval_freq=iteration,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=iteration * 2,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="dqn_autoscaler",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=1,
    )
    callback = CallbackList([checkpoint_callback, eval_callback])

    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        reset_num_timesteps=True,
        progress_bar=True,
        tb_log_name="DQN",
    )

    final_model_path = model_dir / "final" / "model"
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    env.close()
    influxdb.close()
