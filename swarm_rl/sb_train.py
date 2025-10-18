"""
Main script for training a swarm of quadrotors with Stable-Baselines3.
This replaces Sample Factory's training loop with SB3.
"""

import sys
import argparse
import os
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

# from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.env_wrappers.sb3_quad_env import SB3QuadrotorEnv

def make_env_fn(rank, seed=0):
    """
    Utility to create multiple parallel environments.
    """
    def _init():
        env = SB3QuadrotorEnv()
        # env.seed(seed + rank)
        return env
    return _init


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--logdir", type=str, default="./AOC")
    parser.add_argument("--checkpoint_freq", type=int, default=100_000)
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "a2c", "sac"])
    parser.add_argument("--eval_freq", type=int, default=50_000)
    parser.add_argument("--eval_episodes", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Create parallel vectorized environment
    env = SubprocVecEnv([make_env_fn(i) for i in range(args.num_envs)])
    eval_env = SB3QuadrotorEnv()

    # 2. Choose algorithm (here PPO)
    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     learning_rate=args.learning_rate,
    #     n_steps=256,
    #     batch_size=2048,
    #     n_epochs=10,
    #     gamma=0.99,
    #     gae_lambda=1.0,
    #     clip_range=5.0,
    #     verbose=1,
    #     tensorboard_log=os.path.join(args.logdir, "tb"),
    #     device='cpu'
    # )

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        buffer_size=1_000_000,  # replay buffer size
        batch_size=256,  # mini-batch size for updates
        gamma=0.99,  # discount factor
        tau=0.005,  # target smoothing coefficient
        ent_coef='auto',  # automatic entropy coefficient
        verbose=1,
        tensorboard_log=os.path.join(args.logdir, "tb"),
        device='cpu'
    )

    # 3. Add callbacks for checkpointing and evaluation
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq // args.num_envs,
        save_path=os.path.join(args.logdir, "checkpoints"),
        name_prefix="quad_swarm"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.logdir, "best_model"),
        log_path=os.path.join(args.logdir, "eval"),
        eval_freq=args.eval_freq // args.num_envs,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    # 4. Train
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback]
    )

    # 5. Save
    model.save(os.path.join(args.logdir, "final_model"))
    env.close()
    eval_env.close()


if __name__ == "__main__":
    sys.exit(main())
