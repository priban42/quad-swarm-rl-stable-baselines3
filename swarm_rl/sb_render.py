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

def parse_attention(attention_softmax):
    tensor = attention_softmax.reshape(-1)  # flatten to (12,)
    matrix = np.zeros((4, 4))
    for agent in range(4):
        values = tensor[agent * 3: (agent + 1) * 3]  # 3 values for this agent
        # Fill the row, skipping the diagonal
        cols = [c for c in range(4) if c != agent]
        matrix[agent, cols] = values
    return matrix

def render_attention_matrix(matrix, img_width=640, img_height=480):
    matrix = np.array(matrix)
    n = matrix.shape[0]

    AGENT_COLORS = [
        (0, 0, 255),  # red
        (0, 128, 255),  # orange
        (0, 255, 255), # yellow
        (255, 255, 0),  # cyan
        (128, 255, 255),  # magenta
        (255, 0, 0),  # blue
        (120, 51, 56),  # purple
        (255, 0, 255),  # violet
    ]

    label_size = min(img_width, img_height) // 10
    cell_w = (img_width - label_size) // n
    cell_h = (img_height - label_size) // n

    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    # Draw cells
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = (
                int(255 * (1 - val)),
                int(255 * (1 - val)),
                255,
            )
            x1 = label_size + j * cell_w
            y1 = label_size + i * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (180, 180, 180), 1)

            text = f"{val:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = min(cell_w, cell_h) / 150
            text_size = cv2.getTextSize(text, font, font_scale, 1)[0]
            tx = x1 + (cell_w - text_size[0]) // 2
            ty = y1 + (cell_h + text_size[1]) // 2
            cv2.putText(img, text, (tx, ty), font, font_scale, (80, 80, 80), 1)

    # Draw row labels (left)
    for i in range(n):
        color = AGENT_COLORS[i] if i < len(AGENT_COLORS) else (0, 0, 0)
        x1, y1 = 0, label_size + i * cell_h
        cv2.rectangle(img, (x1, y1), (label_size, y1 + cell_h), color, -1)
        text = str(i)
        font_scale = label_size / 60
        text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
        tx = (label_size - text_size[0]) // 2
        ty = y1 + (cell_h + text_size[1]) // 2
        cv2.putText(img, text, (tx, ty), font, font_scale, (255, 255, 255), 2)

    # Draw column labels (top)
    for j in range(n):
        color = AGENT_COLORS[j] if j < len(AGENT_COLORS) else (0, 0, 0)
        x1, y1 = label_size + j * cell_w, 0
        cv2.rectangle(img, (x1, y1), (x1 + cell_w, label_size), color, -1)
        text = str(j)
        font_scale = label_size / 60
        text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
        tx = x1 + (cell_w - text_size[0]) // 2
        ty = (label_size + text_size[1]) // 2
        cv2.putText(img, text, (tx, ty), font, font_scale, (255, 255, 255), 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def render(cfg_, MODEL_PATH, VIDEO_DIR = None, VIDEO_NAME="viAGENT_COLORSdeo"):
    cfg = deepcopy(cfg_)
    NUM_EPISODES = 5
    MAX_FRAMES = 600  # maximum frames per episode
    episode_duration = 60.0
    FPS = 30

    cfg.quads_render = True
    cfg.initial_capture_radius = 0.2
    cfg.episode_duration = 60.0

    env = SB3QuadrotorEnv(cfg)

    model = PPO.load(MODEL_PATH, env=env, device="cpu")

    obs, info = env.reset()
    frame = env.render()
    frame_height, frame_width, _ = frame.shape

    os.makedirs(VIDEO_DIR, exist_ok=True)
    # Initialize video writer
    video_path = Path(VIDEO_DIR)/f"{VIDEO_NAME}_vid.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (frame_width*2, frame_height))

    for episode in range(NUM_EPISODES):
        obs, info = env.reset()
        done = False
        terminated = False
        truncated = False
        episode_reward = 0.0
        frame_count = 0

        while frame_count < MAX_FRAMES:
            action, _states = model.predict(obs, deterministic=True)
            attention_matrix = parse_attention(model.policy.actor_encoder.neighbor_encoder.last_attention)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += np.array(reward).sum()

            # Render and save frame
            frame = env.render()  # ensure we get an image array
            if frame is not None:
                attention_frame = render_attention_matrix(attention_matrix, frame_width, frame_height)
                video_writer.write(cv2.cvtColor(np.hstack([frame, attention_frame]), cv2.COLOR_RGB2BGR))  # convert RGB to BGR for OpenCV
            frame_count += 1
            if any(terminated):
                break

        print(f"Episode {episode + 1}: Reward = {episode_reward}, Frames = {frame_count}")

    video_writer.release()
    env.close()
    print(f"Video saved to {video_path}")

if __name__ == "__main__":
    MODEL_BASE_PATH = "quad_experiment3/final_models"
    MODEL_NAME = "ppo_128_128_full_3_2"
    VIDEO_DIR = "quad_experiment3/videos"
    VIDEO_NAME = MODEL_NAME

    with open(Path(MODEL_BASE_PATH)/f"{MODEL_NAME}.p", "rb") as f:
        cfg = pickle.load(f)
    render(cfg, MODEL_PATH=Path(MODEL_BASE_PATH)/f"{MODEL_NAME}.zip", VIDEO_DIR=VIDEO_DIR, VIDEO_NAME=VIDEO_NAME)