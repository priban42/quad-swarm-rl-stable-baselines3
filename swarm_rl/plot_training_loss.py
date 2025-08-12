import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import re

def moving_average(arr, window_size):
    """
    Compute the moving average of a 1D NumPy array.
    Output is always the same length as the input.

    Parameters
    ----------
    arr : array-like
        Input data array.
    window_size : int
        Number of elements to include in each average.

    Returns
    -------
    np.ndarray
        Array of moving averages (same length as input).
    """
    arr = np.asarray(arr)
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if window_size > len(arr):
        raise ValueError("window_size must be <= length of arr")

    weights = np.ones(window_size) / window_size
    return np.convolve(arr, weights, mode='same')

def main():
    BASE_PATH = Path('./train_dir/quads_multi_mix_baseline_8a_local_v116')
    experiments = {"pozice + rychlost": 'quad_baseline_4_/00_quad_baseline_4_q.c.rew_5.0',
                   "pozice + rychlost + orientace":'quad_neighbor_Rz_4_/00_quad_neighbor_Rz_4_q.c.rew_5.0',
                   "pozice + orientace":'quad_neighbor_pos_Rz_4_/00_quad_neighbor_pos_Rz_4_q.c.rew_5.0',
                   # "pozice":'quad_neighbor_pos_4_/00_quad_neighbor_pos_4_q.c.rew_5.0'}
                   "pozice":'quad_neighbor_TEST_4_/00_quad_neighbor_TEST_4_q.c.rew_5.0'}
    experiments_data = {}
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    for name in experiments:
        plot_training_loss(axs, BASE_PATH/experiments[name], name)
    plt.show()

    # plot_distances(experiments_data)
# train_dir/quads_multi_mix_baseline_8a_local_v116/quad_baseline_4_/00_quad_baseline_4_q.c.rew_5.0/sf_log.txt
# train_dir/quads_multi_mix_baseline_8a_local_v116/quad_baseline_4_/00_quad_baseline_4_q.c.rew_5.0/sf_log.txt
def plot_training_loss(ax, path, label):
    reward_pattern = re.compile(r"Avg episode reward:\s*\[\(0,\s*'(-?\d+\.\d+)'\)\]")
    frames_pattern = re.compile(r"Total num frames:\s*(\d+)")
    policy_pattern = re.compile(r"Updated weights for policy 0, policy_version \s*(\d+)")

    rewards = []
    frames = []
    policy_versions = []
    last_frame = 0
    last_policy_version = 0
    with open(path/"sf_log.txt", "r") as f:
        for line in f:
            reward_match = reward_pattern.search(line)
            frames_match = frames_pattern.search(line)
            policy_match = policy_pattern.search(line)
            if frames_match:
                last_frame = int(frames_match.group(1))
            if policy_match:
                last_policy_version = int(policy_match.group(1))
            if reward_match:
                rewards.append(float(reward_match.group(1)))
                frames.append(last_frame)
                policy_versions.append(last_policy_version)
    # ax.plot(policy_versions, rewards, label=label)
    # ax.plot(frames, rewards, label=label, linewidth=0.8, alpha=1.0)
    ax.plot(frames, moving_average(rewards, 20), label=label, linewidth=1, alpha=1.0)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Avg Episode Reward")
    ax.set_title("Training curves")
    ax.grid(True)
    ax.legend()

if __name__ == "__main__":
    main()