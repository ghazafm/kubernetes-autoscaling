from pathlib import Path

import urllib3
from simulation_environment import K8sAutoscalerEnv
from stable_baselines3 import DQN

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

env = K8sAutoscalerEnv(
    deployment_name="nginx-deployment",
    iteration=3,
    verbose=True,
    action_step=50,
    timeout=30,
)
model = DQN(policy="MlpPolicy", env=env, verbose=1, buffer_size=10000)
model.learn(total_timesteps=10000)
simulation_path = Path("training") / "model" / "Shower_Model_DQN"
model.save(simulation_path)
