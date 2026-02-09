import gymnasium as gym
import torch
import cv2
import numpy as np
from stable_baselines3 import PPO
from swarm_rl.env_wrappers.sb3_quad_env import SB3QuadrotorEnv  # your env wrapper

# ----------------------------
# Configuration
# ----------------------------
# MODEL_PATH = "PPO/best_model/best_model.zip"  # path to your trained model
# MODEL_PATH = "PPO_4_ang/best_model/best_model.zip"  # path to your trained model
# MODEL_PATH = "PPO_4_repulsive/best_model/best_model.zip"  # path to your trained model
# MODEL_PATH = "PPO_4_ang/curriculum_checkpoint/0_049.zip"  # path to your trained model
MODEL_PATH = "PPO_4_repulsive/curriculum_checkpoint/1_075.zip"  # path to your trained model
# MODEL_PATH = "PPO_4_controller/checkpoints/quad_swarm_5199168_steps.zip"  # path to your trained model
NUM_EPISODES = 5
MAX_FRAMES = 600  # maximum frames per episode
episode_duration = 60.0
VIDEO_PATH = "test4_curr_repulsive_sparse.mp4"
FPS = 30

# ----------------------------
# Create environment
# ----------------------------
num_of_agents = 4
# env = SB3QuadrotorEnv(quads_render=True, num_agents=num_of_agents, quads_mode="static_diff_goal")
# env = SB3QuadrotorEnv(seed=3, quads_render=True, episode_duration=episode_duration, num_agents=num_of_agents, quads_mode="static_diff_goal")
# env = SB3QuadrotorEnv(seed=0, quads_render=True, episode_duration=episode_duration, num_agents=num_of_agents, quads_mode="dynamic_same_goal_trajectory")
env = SB3QuadrotorEnv(seed=0, quads_render=True, episode_duration=episode_duration, num_agents=num_of_agents, quads_mode="dynamic_repulsive")

# ----------------------------
# Load trained model
# ----------------------------
model = PPO.load(MODEL_PATH, env=env, device="cpu")

# ----------------------------
# Get frame size from first render
# ----------------------------
obs, info = env.reset()
frame = env.render()
frame_height, frame_width, _ = frame.shape

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(VIDEO_PATH, fourcc, FPS, (frame_width, frame_height))

# ----------------------------
# Run test episodes
# ----------------------------
for episode in range(NUM_EPISODES):
    obs, info = env.reset()
    done = False
    terminated = False
    truncated = False
    episode_reward = 0.0
    frame_count = 0

    while frame_count < MAX_FRAMES:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += np.array(reward).sum()

        # Render and save frame
        frame = env.render()  # ensure we get an image array
        if frame is not None:
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # convert RGB to BGR for OpenCV
        frame_count += 1
        if any(terminated):
            break

    print(f"Episode {episode + 1}: Reward = {episode_reward}, Frames = {frame_count}")

# ----------------------------
# Release resources
# ----------------------------
video_writer.release()
env.close()
print(f"Video saved to {VIDEO_PATH}")