from swarm_rl.env_wrappers.sb3_quad_env import SB3QuadrotorEnv

env = SB3QuadrotorEnv()
obs, info = env.reset()
print("Observation shape:", getattr(obs, "shape", type(obs)))
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print("Step OK:", reward, terminated, truncated)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from swarm_rl.env_wrappers.sb3_quad_env import SB3QuadrotorEnv

env = DummyVecEnv([lambda: SB3QuadrotorEnv()])
model = PPO("MlpPolicy", env, verbose=1)
model.learn(10000)