from pathlib import Path

import urllib3
from simulation_environment import K8sAutoscalerEnv
from stable_baselines3 import DQN

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

env = K8sAutoscalerEnv(
    min_replicas=1,
    max_replicas=100,
    iteration=100,
    namespace="default",
    deployment_name="nodejs-deployment",
    min_cpu=20,
    min_memory=20,
    max_cpu=85,
    max_memory=85,
    verbose=False,
    action_step=100,
    timeout=60,
)
log_path = Path("training") / "logs"
model = DQN(policy="MlpPolicy", env=env, verbose=1, tensorboard_log=str(log_path))
model.learn(total_timesteps=10000)
simulation_path = Path("training") / "model" / "Nodejs_Model_DQN"
model.save(simulation_path)
