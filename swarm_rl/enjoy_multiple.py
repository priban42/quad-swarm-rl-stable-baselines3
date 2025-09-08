import subprocess
import sys
import cv2
import numpy as np
from pathlib import Path
import pickle
import os

def merge_videos(list_of_args, speed=1):
    vid_captures = []
    for args in list_of_args:
        path = args[-4][14:] + "/" + args[-3][13:]
        cap = cv2.VideoCapture(path + "/" + 'replay.mp4')
        vid_captures.append(cap)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_size = (width * len(list_of_args), height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_stacked.mp4', fourcc, fps, output_size)

    while True:
        for i in range(speed):
            rets, frames = [], []

            for cap in vid_captures:
                ret, frame = cap.read()
                rets.append(ret)
                frames.append(frame)

        if not np.all(rets):
            break

        # Resize all frames to the same size (if needed)
        for i in range(len(frames)-1):
            frames[i+1] = cv2.resize(frames[i+1], (width, height))

        # Stack frames horizontally
        stacked_frame = np.hstack(frames)

        # Write to output
        out.write(stacked_frame)
    for cap in vid_captures:
        cap.release()
    out.release()
    # cv2.destroyAllWindows()

def read_data(experiment, name="plot_data.p"):
    path = Path(experiment[-4][14:] + "/" + experiment[-3][13:])
    with open(path/name, 'rb') as f:
        data = pickle.load(f)
    return data
    # plot_distances(experiments_data)

def save_data(data, experiment, name="data.p"):
    path = Path(experiment[-4][14:] + "/" + experiment[-3][13:])
    with open(path / name, 'wb') as f:
        pickle.dump(data, f)

def launch_script():
    quads_render=False
    pos_Rz_4 = [
        sys.executable, "swarm_rl/enjoy.py",  # Replace with your actual script filename
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--replay_buffer_sample_prob=0",
        "--quads_use_numba=False",
        f"--quads_render={quads_render}",
        "--train_dir=./train_dir/quads_multi_mix_baseline_8a_local_v116/quad_neighbor_pos_Rz_4_",
        "--experiment=00_quad_neighbor_pos_Rz_4_q.c.rew_5.0",
        "--quads_view_mode", "global"
    ]
    pos_Rz_vel_4 = [
        sys.executable, "swarm_rl/enjoy.py",  # Replace with your actual script filename
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--replay_buffer_sample_prob=0",
        "--quads_use_numba=False",
        f"--quads_render={quads_render}",
        "--train_dir=./train_dir/quads_multi_mix_baseline_8a_local_v116/quad_neighbor_Rz_4_",
        "--experiment=00_quad_neighbor_Rz_4_q.c.rew_5.0",
        "--quads_view_mode", "global"
    ]
    baseline_4 = [
        sys.executable, "swarm_rl/enjoy.py",  # Replace with your actual script filename
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--replay_buffer_sample_prob=0",
        "--quads_use_numba=False",
        f"--quads_render={quads_render}",
        "--train_dir=./train_dir/quads_multi_mix_baseline_8a_local_v116/quad_baseline_4_",
        "--experiment=00_quad_baseline_4_q.c.rew_5.0",
        "--quads_view_mode", "global"
    ]
    pos = [
        sys.executable, "swarm_rl/enjoy.py",  # Replace with your actual script filename
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--replay_buffer_sample_prob=0",
        "--quads_use_numba=False",
        f"--quads_render={quads_render}",
        "--train_dir=./train_dir/quads_multi_mix_baseline_8a_local_v116/quad_neighbor_pos_4_",
        "--experiment=00_quad_neighbor_pos_4_q.c.rew_5.0",
        "--quads_view_mode", "global"
    ]
    npos = [
        sys.executable, "swarm_rl/enjoy.py",  # Replace with your actual script filename
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--replay_buffer_sample_prob=0",
        "--quads_use_numba=False",
        f"--quads_render={quads_render}",
        "--train_dir=./train_dir/quads_multi_mix_baseline_8a_local_v116/npos_4_",
        "--experiment=00_npos_4_q.c.rew_5.0",
        "--quads_view_mode", "global"
    ]
    #######
    pos_4_ = [
        sys.executable, "swarm_rl/enjoy.py",
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--replay_buffer_sample_prob=0",
        "--quads_use_numba=False",
        f"--quads_render={quads_render}",
        "--train_dir=./train_dir/quads_multi_mix_baseline_8a_local_v116/pos_4_",
        "--experiment=00_pos_4_q.c.rew_5.0",
        "--quads_view_mode", "global"
    ]
    pos_Rz_4_ = [
        sys.executable, "swarm_rl/enjoy.py",
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--replay_buffer_sample_prob=0",
        "--quads_use_numba=False",
        f"--quads_render={quads_render}",
        "--train_dir=./train_dir/quads_multi_mix_baseline_8a_local_v116/pos_Rz_4_",
        "--experiment=00_pos_Rz_4_q.c.rew_5.0",
        "--quads_view_mode", "global"
    ]
    pos_vel_4_ = [
        sys.executable, "swarm_rl/enjoy.py",
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--replay_buffer_sample_prob=0",
        "--quads_use_numba=False",
        f"--quads_render={quads_render}",
        "--train_dir=./train_dir/quads_multi_mix_baseline_8a_local_v116/pos_vel_4_",
        "--experiment=00_pos_vel_4_q.c.rew_5.0",
        "--quads_view_mode", "global"
    ]
    pos_vel_Rz_4_ = [
        sys.executable, "swarm_rl/enjoy.py",
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--replay_buffer_sample_prob=0",
        "--quads_use_numba=False",
        f"--quads_render={quads_render}",
        "--train_dir=./train_dir/quads_multi_mix_baseline_8a_local_v116/pos_vel_Rz_4_",
        "--experiment=00_pos_vel_Rz_4_q.c.rew_5.0",
        "--quads_view_mode", "global"
    ]
    pos_8_ = [
        sys.executable, "swarm_rl/enjoy.py",
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--replay_buffer_sample_prob=0",
        "--quads_use_numba=False",
        f"--quads_render={quads_render}",
        "--train_dir=./train_dir/quads_multi_mix_baseline_8a_local_v116/pos_8_",
        "--experiment=00_pos_8_q.c.rew_5.0",
        "--quads_view_mode", "global"
    ]
    pos_Rz_8_ = [
        sys.executable, "swarm_rl/enjoy.py",
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--replay_buffer_sample_prob=0",
        "--quads_use_numba=False",
        f"--quads_render={quads_render}",
        "--train_dir=./train_dir/quads_multi_mix_baseline_8a_local_v116/pos_Rz_8_",
        "--experiment=00_pos_Rz_8_q.c.rew_5.0",
        "--quads_view_mode", "global"
    ]
    pos_vel_8_ = [
        sys.executable, "swarm_rl/enjoy.py",
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--replay_buffer_sample_prob=0",
        "--quads_use_numba=False",
        f"--quads_render={quads_render}",
        "--train_dir=./train_dir/quads_multi_mix_baseline_8a_local_v116/pos_vel_8_",
        "--experiment=00_pos_vel_8_q.c.rew_5.0",
        "--quads_view_mode", "global"
    ]
    pos_vel_Rz_8_ = [
        sys.executable, "swarm_rl/enjoy.py",
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--replay_buffer_sample_prob=0",
        "--quads_use_numba=False",
        f"--quads_render={quads_render}",
        "--train_dir=./train_dir/quads_multi_mix_baseline_8a_local_v116/pos_vel_Rz_8_",
        "--experiment=00_pos_vel_Rz_8_q.c.rew_5.0",
        "--quads_view_mode", "global"
    ]
    #######
    # experiments = [baseline_4, npos]
    # experiments = [baseline_4]
    experiments = [pos_4_, pos_Rz_4_, pos_vel_4_, pos_vel_Rz_4_, pos_8_, pos_Rz_8_, pos_vel_8_, pos_vel_Rz_8_]

    for experiment in experiments:
        global_data = {"avg_dist":{}, "distance_to_goal":{}, "collisions":{}}
        for i in range(1, 16):
            num_agents = i*4
            print(f"number of quads:{num_agents}")
            subprocess.run(experiment + [f"--quads_num_agents={num_agents}"])
            data = read_data(experiment)
            global_data["avg_dist"][num_agents] = (np.mean(data["distance_to_goal"]))
            global_data["distance_to_goal"][num_agents] = data["distance_to_goal"]
            global_data["collisions"][num_agents] = data["collisions"]
        save_data(global_data, experiment, "global_data.p")
        print(f"saved global data for {num_agents=}")

        # subprocess.run(pos_Rz_4)
        # subprocess.run(pos_Rz_vel_4 + ["--quads_num_agents=8"])
        # read_data(pos_Rz_vel_4)
        # subprocess.run(baseline_4)

    # merge_videos(experiments)

if __name__ == "__main__":
    launch_script()