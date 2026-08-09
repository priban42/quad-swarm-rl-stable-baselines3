import gymnasium as gym
import torch
import cv2
import numpy as np
from stable_baselines3 import PPO
from swarm_rl.env_wrappers.sb3_quad_env import SB3QuadrotorEnv
from swarm_rl.global_cfg import QuadrotorEnvConfig
from pathlib import Path
from copy import deepcopy
import os
import pickle
from dataclasses import dataclass, field
from swarm_rl.analytic_models import Janosov, Angelani


def load_model_env(cfg, MODEL_PATH=None, model_type=None):
    env = SB3QuadrotorEnv(cfg)
    if model_type == "Jasonov":
        print(f"loaded model: Jasonov")
        model = Janosov(cfg)
    elif model_type == "Angelani":
        print(f"loaded model: Angelani")
        model=Angelani(cfg)
    else:
        model = PPO.load(MODEL_PATH, env=env, device="cpu")
        print(f"loaded model: PPO")
    return env, model

def eval_single_config(env, model, NUM_EPISODES=50, MAX_FRAMES=600):
    obs, info = env.reset()
    episode_lengths = []
    successes = []
    for episode in range(NUM_EPISODES):
        obs, info = env.reset()
        done = False
        terminated = False
        truncated = False
        episode_reward = 0.0
        frame_count = 0
        success = False
        while frame_count < MAX_FRAMES:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += np.array(reward).sum()
            frame_count += 1
            if any(terminated):
                success = True
                break
        episode_lengths.append(frame_count)
        successes.append(success)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.02f}, Frames = {frame_count}")
    return successes, episode_lengths


def eval(cfg_, EVAL_DIR, attribute_name, attribute_values, MODEL_PATH=None):
    cfg = deepcopy(cfg_)
    eval_logs = {"cfgs":[], "sucesses":[], "episode_lengths":[], "attribute_name":attribute_name, "attribute_values":attribute_values}
    os.makedirs(EVAL_DIR, exist_ok=True)
    for attr_val in attribute_values:
        setattr(cfg, attribute_name, attr_val)
        print(f"setting {attribute_name}={attr_val}")
        env, model = load_model_env(cfg, MODEL_PATH, model_type=cfg.model_type)
        successes, episode_lengths = eval_single_config(env, model)
        env.close()
        eval_logs["cfgs"].append(cfg)
        eval_logs["sucesses"].append(successes)
        eval_logs["episode_lengths"].append(episode_lengths)
    with open(EVAL_DIR/f"{attribute_name}.p", "wb") as f:
        pickle.dump(eval_logs, f)
    print("EVAL DONE")

def viz_eval(EVAL_PATHS, attribute_name, invert_x=False):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10.colors

    for idx, eval_path in enumerate(EVAL_PATHS):
        pickle_file = Path(eval_path) / f"{attribute_name}.p"
        with open(pickle_file, "rb") as f:
            logs = pickle.load(f)

        attr_values = logs["attribute_values"]
        model_type = logs["cfgs"][0].model_type if logs["cfgs"][0].model_type is not None else f"PPO"
        color = colors[idx % len(colors)]

        avg_success_rates = [np.mean(s) for s in logs["sucesses"]]
        avg_episode_lengths = [np.mean(e) for e in logs["episode_lengths"]]

        ax1.plot(attr_values, avg_success_rates, marker="o", label=model_type, color=color)
        ax2.plot(attr_values, avg_episode_lengths, marker="o", label=model_type, color=color)

    ax1.set_xlabel(attribute_name)
    ax1.set_ylabel("Average Success Rate")
    ax1.set_title("Success Rate")
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel(attribute_name)
    ax2.set_ylabel("Average Episode Length (frames)")
    ax2.set_title("Episode Length")
    ax2.legend()
    ax2.grid(True)

    fig.suptitle(f"Evaluation over {attribute_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if invert_x:
        ax1.invert_xaxis()
        ax2.invert_xaxis()
    plt.show()

if __name__ == "__main__":
    MODEL_BASE_PATH = "quad_experiment3/final_models"
    MODEL_NAME = "ppo_128_128_full_3_36"
    EVAL_BASE_PATH = "eval"
    MODEL_PATH = Path(MODEL_BASE_PATH) / f"{MODEL_NAME}.zip"
    with open(Path(MODEL_BASE_PATH)/f"{MODEL_NAME}.p", "rb") as f:
        cfg = pickle.load(f)
    cfg.model_type = None
    # cfg.model_type = "Jasonov"
    # cfg.model_type = "Angelani"
    cfg.initial_capture_radius = 0.2
    cfg.episode_duration = 60.0
    if cfg.model_type is not None:
        MODEL_NAME = cfg.model_type
    EVAL_PATH = Path(EVAL_BASE_PATH) / MODEL_NAME
    # eval(cfg, EVAL_PATH, MODEL_PATH=MODEL_PATH, attribute_name="initial_capture_radius", attribute_values=[1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    # eval(cfg, EVAL_PATH, attribute_name="initial_capture_radius", attribute_values=[0.1])
    viz_eval([EVAL_PATH, f"{EVAL_BASE_PATH}/Jasonov", f"{EVAL_BASE_PATH}/Angelani"], "initial_capture_radius", invert_x=True)