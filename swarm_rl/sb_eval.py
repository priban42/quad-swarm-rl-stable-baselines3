import gymnasium as gym
import torch
import numpy as np
from numpy.ma.extras import average
from stable_baselines3 import PPO
from swarm_rl.env_wrappers.sb3_quad_env import SB3QuadrotorEnv  # your env wrapper
import matplotlib.pyplot as plt

def eval_model(model_path, num_of_agents, max_frames, num_episodes, episode_duration):
    env = SB3QuadrotorEnv(seed=1, quads_render=False, episode_duration=episode_duration, num_agents=num_of_agents, quads_mode="dynamic_same_goal_trajectory")
    model = PPO.load(model_path, env=env, device="cpu")
    obs, info = env.reset()

    episode_lengths = []
    average_distances = []
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        terminated = False
        truncated = False
        episode_reward = 0.0
        frame_count = 0
        average_distance = 0
        while frame_count < max_frames:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += np.array(reward).sum()
            average_distance += sum(d['goal_dist'] for d in info)/num_of_agents
            frame_count += 1
            if any(terminated):
                break
        episode_lengths.append(frame_count)
        average_distances.append(average_distance / frame_count)
        env.reset()
        print(f"Episode {episode + 1}: Reward = {episode_reward}, Frames = {frame_count}")
    env.close()
    return episode_lengths, average_distances

if __name__ == "__main__":
    MODEL_PATH = "PPO_4_ang/curriculum_checkpoint/0_049.zip"  # path to your trained model
    NUM_EPISODES = 10
    MAX_FRAMES = 3000  # maximum frames per episode
    episode_duration = 60.0 # s
    num_of_agents = 4
    episode_lengths_curr, average_distances_curr = eval_model("PPO_4_ang/curriculum_checkpoint/0_049.zip", num_of_agents, MAX_FRAMES, NUM_EPISODES, episode_duration)
    episode_lengths, average_distances = eval_model("PPO_4_ang/best_model/best_model.zip", num_of_agents, MAX_FRAMES, NUM_EPISODES, episode_duration)
    labels = ["curriculum", "no curriculum"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(episode_lengths_curr, label=labels[0])
    ax[1].plot(average_distances_curr, label=labels[0])
    ax[0].plot(episode_lengths, label=labels[1])
    ax[1].plot(average_distances, label=labels[1])
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel('episode')
    ax[0].set_ylabel('episode length')
    ax[1].set_xlabel('episode')
    ax[1].set_ylabel('average distance from goal')
    plt.tight_layout();
    plt.show()
    pass