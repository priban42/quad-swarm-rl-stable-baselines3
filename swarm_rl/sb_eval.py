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
from scipy.optimize import minimize


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
    min_distances = []
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
        min_distances.append(env.env.env.min_distance)
        episode_lengths.append(frame_count)
        successes.append(success)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.02f}, Frames = {frame_count}")
    return successes, episode_lengths, min_distances


import copy
from concurrent.futures import ProcessPoolExecutor
import numpy as np


def _eval_worker(args):
    """Worker function that runs a subset of episodes in its own isolated environment."""
    cfg, model_path, num_episodes, max_frames, x_candidate = args

    # 1. Each worker safely builds its own environment and model from scratch
    env, model = load_model_env(cfg, model_path, model_type=cfg.model_type)

    # 2. Apply the current optimization parameters
    model.set(x_candidate)

    local_successes = []
    local_episode_lengths = []
    local_min_distances = []

    try:
        for episode in range(num_episodes):
            obs, info = env.reset()
            frame_count = 0
            success = False

            while frame_count < max_frames:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                frame_count += 1

                if any(np.atleast_1d(terminated)):
                    success = True
                    break

            local_min_distances.append(env.env.env.min_distance)
            local_episode_lengths.append(frame_count)
            local_successes.append(success)
    finally:
        env.close()

    return local_successes, local_episode_lengths, local_min_distances


def eval_single_config_parallel(cfg, model_path, x_candidate, NUM_EPISODES=50, MAX_FRAMES=600, num_workers=8):
    """Splits NUM_EPISODES across multiple parallel worker processes safely."""

    # Create a picklable copy of cfg by removing/neutralizing the unpicklable OUNoiseNumba object
    cfg_clean = copy.deepcopy(cfg)
    if hasattr(cfg_clean, 'ounoise'):
        setattr(cfg_clean, 'ounoise', None)

    episodes_per_worker = NUM_EPISODES // num_workers
    remainder = NUM_EPISODES % num_workers

    tasks = []
    for i in range(num_workers):
        n_eps = episodes_per_worker + (remainder if i == num_workers - 1 else 0)
        if n_eps > 0:
            # Pass only serializable data (cfg_clean, strings, ints, numpy arrays)
            tasks.append((cfg_clean, MODEL_PATH, n_eps, MAX_FRAMES, x_candidate))

    successes = []
    episode_lengths = []
    min_distances = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(_eval_worker, tasks)

        for worker_successes, worker_lengths, worker_dists in results:
            successes.extend(worker_successes)
            episode_lengths.extend(worker_lengths)
            min_distances.extend(worker_dists)

    return successes, episode_lengths, min_distances


def eval(cfg_, EVAL_DIR, attribute_name, attribute_values, MODEL_PATH=None):
    cfg = deepcopy(cfg_)
    eval_logs = {"cfgs":[], "sucesses":[], "episode_lengths":[], "attribute_name":attribute_name, "attribute_values":attribute_values}
    os.makedirs(EVAL_DIR, exist_ok=True)
    for attr_val in attribute_values:
        setattr(cfg, attribute_name, attr_val)
        print(f"setting {attribute_name}={attr_val}")
        env, model = load_model_env(cfg, MODEL_PATH, model_type=cfg.model_type)
        successes, episode_lengths, min_distances = eval_single_config(env, model)
        env.close()
        eval_logs["cfgs"].append(cfg)
        eval_logs["sucesses"].append(successes)
        eval_logs["episode_lengths"].append(episode_lengths)
    with open(EVAL_DIR/f"{attribute_name}.p", "wb") as f:
        pickle.dump(eval_logs, f)
    print("EVAL DONE")

def tune(cfg_):
    cfg = deepcopy(cfg_)
    cfg.initial_capture_radius = 0.01
    cfg.seed = 0
    env, model = load_model_env(cfg, MODEL_PATH, model_type=cfg.model_type)
    x0 = model.get()
    env.close()
    def objective(x):
        # env, model = load_model_env(cfg, MODEL_PATH, model_type=cfg.model_type)
        # model.set(x)
        successes, episode_lengths, min_distances = eval_single_config(env, model, NUM_EPISODES=5)
        # successes, episode_lengths, min_distances = eval_single_config_parallel(cfg, MODEL_PATH, NUM_EPISODES=20, num_workers=8, x_candidate=x)
        mean_min_dist = np.mean(min_distances)
        print(model)
        print(f"{mean_min_dist=}")
        print(f"{min_distances=}")
        env.close()
        return mean_min_dist

# mean_min_dist=0.2514253612616709
# min_distances=[0.2867881131755944, 0.2612691999813549, 0.21526269025527056, 0.13668563936068664, 0.3571211635354479]

    result = minimize(
        objective,
        x0,
        method='Nelder-Mead',
        options={'maxiter': 100, 'disp': True}
    )
    print(result)

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
    # MODEL_NAME = "ppo_128_128_full_3_36"
    MODEL_NAME = "big_gamma_0.995_1"
    EVAL_BASE_PATH = "eval"
    MODEL_PATH = Path(MODEL_BASE_PATH) / f"{MODEL_NAME}.zip"
    with open(Path(MODEL_BASE_PATH)/f"{MODEL_NAME}.p", "rb") as f:
        cfg = pickle.load(f)
    cfg.model_type = None
    # cfg.model_type = "Jasonov"
    # cfg.model_type = "Angelani"
    # cfg.initial_capture_radius = 0.2
    cfg.episode_duration = 60.0
    if cfg.model_type is not None:
        MODEL_NAME = cfg.model_type
    EVAL_PATH = Path(EVAL_BASE_PATH) / MODEL_NAME
    # eval(cfg, EVAL_PATH, MODEL_PATH=MODEL_PATH, attribute_name="initial_capture_radius", attribute_values=[1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    # tune(cfg)
    # eval(cfg, EVAL_PATH, attribute_name="initial_capture_radius", attribute_values=[0.1])
    viz_eval([EVAL_PATH, f"{EVAL_BASE_PATH}/Jasonov", f"{EVAL_BASE_PATH}/Angelani"], "initial_capture_radius", invert_x=True)