"""
Main script for training a swarm of quadrotors with Stable-Baselines3.
This replaces Sample Factory's training loop with SB3.
"""

import sys
import argparse
import os
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from swarm_rl.env_wrappers.subproc_vec_env_custom import SubprocVecEnvCustom
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from swarm_rl.env_wrappers.MetaQuadFactory import MetaQuadFactory
from gym_art.quadrotor_multi.quadrotor_instance import QuadrotorEnvInstance

from sample_factory.model.actor_critic import ActorCriticSharedWeights
# from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.env_wrappers.sb3_quad_env import SB3QuadrotorEnv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=48)
    parser.add_argument("--total_timesteps", type=int, default=150_000_000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--logdir", type=str, default="./PPO_4_controller")
    parser.add_argument("--checkpoint_freq", type=int, default=100_000)
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "a2c", "sac"])
    parser.add_argument("--eval_freq", type=int, default=100_000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    return parser.parse_args()


import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from swarm_rl.models.quad_multi_model import QuadMultiHeadAttentionEncoder, QuadMultiEncoder
from torch import Tensor, nn
from typing import Dict, Optional
from swarm_rl.models.ActorCriticPolicyCustom import ActorCriticPolicyCustomSeparateWeights


class QuadEncoderExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int, cfg):
        super().__init__(observation_space, features_dim)
        self.encoder = QuadMultiHeadAttentionEncoder(cfg, observation_space)
        # Store the encoder’s true output size (SB3 will use it internally)
        # self._features_dim = self.encoder.get_out_size()
        # self.encoder_output_size = 2 * cfg.rnn_size
        self._features_dim = 2 * cfg.rnn_size

    def forward(self, observations):
        # SB3 will pass observations as tensors
        # If your encoder expects a dict, you can wrap here
        if isinstance(observations, torch.Tensor):
            obs_dict = {'obs': observations}
        else:
            obs_dict = observations
        ret = self.encoder(obs_dict)
        return ret


from dataclasses import dataclass, field
from typing import List
from swarm_rl.models.ActorCriticPolicyCustom import QuadrotorEnvConfig

def main():
    args = parse_args()

    cfg = QuadrotorEnvConfig()
    num_of_agents = cfg.quads_num_agents

    def make_env_fn(rank, seed=0):
        def _init():
            env = SB3QuadrotorEnv(num_agents=num_of_agents)
            return env
        return _init

    # 1. Create parallel vectorized environment
    # meta_quad_factory = MetaQuadFactory()
    # meta_quad_factory.initialize()

    # env = DummyVecEnv([make_env_fn(i) for i in range(args.num_envs*num_of_agents)])
    # eval_env = DummyVecEnv([make_env_fn(i) for i in range(1*num_of_agents)])

    # env = SubprocVecEnv([make_env_fn(i) for i in range(args.num_envs*num_of_agents)])
    # eval_env = SubprocVecEnv([make_env_fn(i) for i in range(1*num_of_agents)])

    env = SubprocVecEnvCustom([make_env_fn(i) for i in range(args.num_envs)], agents_per_env=num_of_agents)
    eval_env = SubprocVecEnvCustom([make_env_fn(i) for i in range(1)], agents_per_env=num_of_agents)

    policy_kwargs = dict(
        features_extractor_class=QuadEncoderExtractor,
        features_extractor_kwargs=dict(
            cfg=cfg,  # your config object
            features_dim=256,
        ),
    )

    # 2. Choose algorithm (here PPO)
    model = PPO(
        policy = ActorCriticPolicyCustomSeparateWeights,
        env=env,
        # policy_kwargs=policy_kwargs,
        learning_rate=args.learning_rate,
        n_steps=512,  # rollout
        batch_size=1024,
        n_epochs=10,
        gamma=0.99,
        # gae_lambda=1.0,
        # clip_range=5.0,
        verbose=1,
        tensorboard_log=os.path.join(args.logdir, "tb"),
        device='cuda'
        # device='cpu'
    )

    # model = SAC(
    #     "MlpPolicy",
    #     env,
    #     learning_rate=args.learning_rate,
    #     buffer_size=1_000_000,  # replay buffer size
    #     batch_size=256,  # mini-batch size for updates
    #     gamma=0.99,  # discount factor
    #     tau=0.005,  # target smoothing coefficient
    #     ent_coef='auto',  # automatic entropy coefficient
    #     verbose=1,
    #     tensorboard_log=os.path.join(args.logdir, "tb"),
    #     device='cpu'
    # )

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
