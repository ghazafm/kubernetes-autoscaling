import argparse
import ast
import os
import pickle
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from environment import KubernetesEnv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from utils import setup_logger

from database import InfluxDB

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--second", action="store_true", help="Use .env.second file")
    args, _ = parser.parse_known_args()

    if args.second:
        load_dotenv(".env.second")
    else:
        load_dotenv()

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

    iteration = int(os.getenv("ITERATION"))
    csv_log_dir = os.getenv("CSV_LOG_DIR", "data")
    note = os.getenv("NOTE", "default")

    env = KubernetesEnv(
        min_replicas=int(os.getenv("MIN_REPLICAS")),
        max_replicas=int(os.getenv("MAX_REPLICAS")),
        iteration=iteration,
        namespace=os.getenv("NAMESPACE"),
        deployment_name=os.getenv("DEPLOYMENT_NAME"),
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
        render_mode="human",
        csv_log_dir=csv_log_dir,
        csv_log_prefix=f"{now}_{note}",
        mode="dev",
    )

    eval_env = KubernetesEnv(
        min_replicas=int(os.getenv("MIN_REPLICAS")),
        max_replicas=max(int(os.getenv("MAX_REPLICAS")) // 2, 1),
        iteration=iteration,
        namespace=os.getenv("NAMESPACE"),
        deployment_name=os.getenv("DEPLOYMENT_NAME"),
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
        render_mode="human",
        csv_log_dir=csv_log_dir,
        csv_log_prefix=f"{now}_{note}_eval",
        mode="prod",
    )
    eval_env = Monitor(eval_env)

    # Training configuration
    BASE_EPISODES = 10
    num_episodes = int(os.getenv("EPISODE", BASE_EPISODES))

    resume_path = os.getenv("RESUME_PATH", "")

    if resume_path:
        resume_model_path = Path(resume_path)
        if not resume_model_path.exists():
            raise FileNotFoundError(f"Resume model not found: {resume_path}")

        model_dir = resume_model_path.parent.parent / f"resume_{now}_{note}"
        logger.info(f"Resuming training from: {resume_path}")

        model = DQN.load(
            resume_path,
            env=env,
            tensorboard_log=log_dir,
            device="auto",
        )

        try:
            checkpoints_dir = resume_model_path.parent.parent / "checkpoints"
            if checkpoints_dir.exists():
                vec_files = list(checkpoints_dir.glob("*vecnormalize*"))
                if vec_files:
                    vec_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    vec_path = str(vec_files[0])
                    venv = DummyVecEnv([lambda: env])
                    vecnorm = VecNormalize.load(vec_path, venv)
                    model.set_env(vecnorm)
                    env = vecnorm
                    logger.info(f"Loaded VecNormalize from {vec_path}")
        except Exception as e:
            logger.warning(f"Could not restore VecNormalize: {e}")

        try:
            if checkpoints_dir.exists():
                replay_candidates = list(checkpoints_dir.glob("*replay*"))
            else:
                replay_candidates = []

            if replay_candidates:
                replay_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                replay_path = replay_candidates[0]

                try:
                    model.load_replay_buffer(str(replay_path))
                    logger.info(f"Loaded replay buffer from {replay_path}")
                except AttributeError:
                    with Path(replay_path).open("rb") as fh:
                        buf = pickle.load(fh)  # noqa: S301
                    model.replay_buffer = buf
                    logger.info(f"Assigned replay buffer from {replay_path}")
        except Exception as e:
            logger.warning(f"Could not load replay buffer: {e}")

        old_timesteps = model.num_timesteps
        logger.info(f"Loaded model from timestep: {old_timesteps}")

        model.exploration_fraction = 0.4
        model.exploration_initial_eps = 1.0
        model.exploration_final_eps = 0.1

        model.exploration_schedule = LinearSchedule(
            model.exploration_initial_eps,
            model.exploration_final_eps,
            model.exploration_fraction,
        )

        model.num_timesteps = 0
        model._n_calls = 0
        model.exploration_rate = model.exploration_initial_eps

        logger.info(
            f"Reset exploration schedule: "
            f"fraction={model.exploration_fraction}, "
            f"initial_eps={model.exploration_initial_eps}, "
            f"final_eps={model.exploration_final_eps}"
        )
        logger.info(f"Current exploration_rate: {model.exploration_rate:.3f}")

        additional_timesteps = num_episodes * iteration
        total_timesteps = additional_timesteps

        logger.info(f"Will train for {additional_timesteps} steps (fresh exploration)")
        logger.info(f"Model weights from {old_timesteps} offline steps are preserved")

    else:
        model_dir = Path(f"model/{now}_{note}")
        model_dir.mkdir(parents=True, exist_ok=True)
        total_timesteps = num_episodes * iteration

        model = DQN(
            policy="MlpPolicy",
            env=env,
            target_update_interval=iteration,  # Nilai default 10000, yang mana terlalu besar untuk jumlah pelatihan yang lebih sedikit
            exploration_fraction=0.4,  # Eksplorasi lebih lama dibandingkan dengan defaultnya yaitu 0.1, hal ini dilakukan agar agen mendapatkan pengalaman yang lebih beragam, dan tidak terjebak dalam eksploitasi awal.
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
        eval_freq=iteration * 10,
        n_eval_episodes=1,
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
        reset_num_timesteps=False,
        progress_bar=True,
        tb_log_name="DQN",
    )

    final_model_path = model_dir / "final" / "model"
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    # Log CSV stats
    csv_stats = env.csv_logger.get_stats()
    if csv_stats.get("enabled"):
        logger.info(f"CSV transitions saved to: {csv_stats['filepath']}")
        logger.info(f"Total transitions logged: {csv_stats['total_steps']}")

    env.close()
    influxdb.close()
