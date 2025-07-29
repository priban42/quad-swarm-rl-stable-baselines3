import subprocess
import sys
import cv2
import numpy as np
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
    output_size = (width * 3, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_stacked.mp4', fourcc, fps, output_size)

    while True:
        for i in range(speed):
            rets, frames = [], []

            for cap in vid_captures:
                ret, frame = cap.read()
                rets.append(ret)
                frames.append(frame)

        if not (rets[0] and rets[1] and rets[2]):
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
    cv2.destroyAllWindows()


def launch_script():
    pos_Rz_4 = [
        sys.executable, "swarm_rl/enjoy.py",  # Replace with your actual script filename
        "--algo=APPO",
        "--env=quadrotor_multi",
        "--replay_buffer_sample_prob=0",
        "--quads_use_numba=False",
        "--quads_render=True",
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
        "--quads_render=True",
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
        "--quads_render=True",
        "--train_dir=./train_dir/quads_multi_mix_baseline_8a_local_v116/quad_baseline_4_",
        "--experiment=00_quad_baseline_4_q.c.rew_5.0",
        "--quads_view_mode", "global"
    ]
    pass

    # subprocess.run(pos_Rz_4)
    # subprocess.run(pos_Rz_vel_4)
    # subprocess.run(baseline_4)

    merge_videos([baseline_4, pos_Rz_vel_4, pos_Rz_4])

if __name__ == "__main__":
    launch_script()