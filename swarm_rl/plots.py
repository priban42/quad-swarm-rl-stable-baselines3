import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def main():
    BASE_PATH = Path('./train_dir/quads_multi_mix_baseline_8a_local_v116')
    experiments = {"pozice + rychlost": 'quad_baseline_4_/00_quad_baseline_4_q.c.rew_5.0',
                   "pozice + rychlost + orientace":'quad_neighbor_Rz_4_/00_quad_neighbor_Rz_4_q.c.rew_5.0',
                   "pozice + orientace":'quad_neighbor_pos_Rz_4_/00_quad_neighbor_pos_Rz_4_q.c.rew_5.0',
                   "pozice":'quad_neighbor_pos_4_/00_quad_neighbor_pos_4_q.c.rew_5.0'}
    experiments_data = {}
    for name in experiments:
        path = BASE_PATH / experiments[name]
        # with open(path/'plot_data.p', 'rb') as f:
        with open(path/'global_data.p', 'rb') as f:
            data = pickle.load(f)
        experiments_data[name] = data
    plot_success(experiments_data)
    plot_collisions(experiments_data)
    # plot_distances(experiments_data)

def plot_distances(data_dict):
    data_entry_name = "distance_to_goal"
    plt.figure(figsize=(10, 6))
    for exp_name, entries in data_dict.items():
        if data_entry_name not in entries:
            print(f"Warning: '{data_entry_name}' not found in experiment '{exp_name}'. Skipping.")
            continue

        # Concatenate the list of ndarrays into one 1D array for plotting
        data_list = entries[data_entry_name]
        # combined = np.concatenate(data_list)
        combined = np.mean(data_list, axis=0)
        # combined = data_list[0]
        plt.plot(combined, label=exp_name)

    plt.title(f"mean distance to goal")
    plt.xlabel("Step")
    plt.ylabel("meters")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_success(data_dict):
    data_entry_name = "avg_dist"
    plt.figure(figsize=(10, 6))
    for exp_name, entries in data_dict.items():
        if data_entry_name not in entries:
            print(f"Warning: '{data_entry_name}' not found in experiment '{exp_name}'. Skipping.")
            continue

        # Concatenate the list of ndarrays into one 1D array for plotting
        data_list = entries[data_entry_name]
        y = np.array([v for v in data_list.values()])
        # y = np.array([sum(v) for v in data_list.values()])
        x = np.array([v for v in data_list])
        # combined = np.concatenate(data_list)
        # combined = data_list[0]
        plt.plot(x, y, label=exp_name)

    plt.title(f"mean distance to goal")
    plt.xlabel("Number of agents")
    plt.ylabel("meters")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_collisions(data_dict):
    data_entry_name = "collisions"
    plt.figure(figsize=(10, 6))
    for exp_name, entries in data_dict.items():
        if data_entry_name not in entries:
            print(f"Warning: '{data_entry_name}' not found in experiment '{exp_name}'. Skipping.")
            continue

        # Concatenate the list of ndarrays into one 1D array for plotting
        data_list = entries[data_entry_name]
        # y = np.array([v for v in data_list.values()])
        y = np.array([sum(v) for v in data_list.values()])
        x = np.array([v for v in data_list])
        # combined = np.concatenate(data_list)
        # combined = data_list[0]
        plt.plot(x, y, label=exp_name)

    plt.title(f"Collisions")
    plt.xlabel("Number of agents")
    plt.ylabel("collisions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()