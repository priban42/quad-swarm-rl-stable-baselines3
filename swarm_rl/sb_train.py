"""
Main script for training a swarm of quadrotors with Stable-Baselines3.
This replaces Sample Factory's training loop with SB3.
"""

import sys
import argparse
import os
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from swarm_rl.env_wrappers.subproc_vec_env_custom import SubprocVecEnvCustom
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from swarm_rl.custom_callbacks import CheckpointCallback, EvalCallback, CurriculumCallback, TensorboardHParamCallback


from swarm_rl.env_wrappers.sb3_quad_env import SB3QuadrotorEnv
from swarm_rl.env_wrappers.custom_dummy_vec_env import DummyVecEnv
import multiprocessing as mp

from swarm_rl.models.ActorCriticPolicyCustom import ActorCriticPolicyCustomSeparateWeights
from swarm_rl.global_cfg import QuadrotorEnvConfig

import torch
# torch.set_num_threads(1)

def parse_args_from_cfg(cfg):
    parser = argparse.ArgumentParser()
    for key, value in vars(cfg).items():
        parser.add_argument(f"--{key}", type=type(value), default=None)
    return parser.parse_args()

def update_cfg_from_args(cfg, args):
    for key, value in vars(args).items():
        if value is not None:
            setattr(cfg, key, value)

def train(cfg:QuadrotorEnvConfig):
    num_of_agents = cfg.num_agents
    # used for curriculum parameters across parallel processes
    note=f"{cfg.to_string()}"
    print(note)
    def make_env_fn(rank, seed=0):
        def _init():
            env = SB3QuadrotorEnv(cfg)
            return env
        return _init

    # env = DummyVecEnv([make_env_fn(i) for i in range(args.num_envs * num_of_agents)])
    env = SubprocVecEnvCustom([make_env_fn(i) for i in range(cfg.num_envs)], agents_per_env=num_of_agents)
    eval_env = SubprocVecEnvCustom([make_env_fn(i) for i in range(1)], agents_per_env=num_of_agents)

    # 2. Choose algorithm (here PPO)
    model = PPO(
        policy = ActorCriticPolicyCustomSeparateWeights,
        policy_kwargs={"cfg":cfg},
        env=env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,  # rollout
        # n_steps=2048,  # rollout
        batch_size=cfg.batch_size,
        # batch_size=2048,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        verbose=1,
        tensorboard_log=os.path.join(cfg.logdir, "tb"),
        device=cfg.device
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.checkpoint_freq // cfg.num_envs,
        save_path=os.path.join(cfg.logdir, "checkpoints"),
        name_prefix="quad_swarm"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(cfg.logdir, "best_model"),
        log_path=os.path.join(cfg.logdir, "eval"),
        eval_freq=cfg.eval_freq // cfg.num_envs,
        n_eval_episodes=cfg.eval_episodes,
        deterministic=True,
        render=False,
    )

    curriculum_callback = CurriculumCallback(
        cfg=cfg,
        initial_capture_radius=cfg.initial_capture_radius,
        eval_env=eval_env,
        save_path=cfg.logdir,
        eval_freq=1000,
        n_eval_episodes=10,
    )

    note_callback = TensorboardHParamCallback(hparams_dict=cfg.to_dict(), note=note)

    # 4. Train
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=[checkpoint_callback, eval_callback, curriculum_callback, note_callback],
        tb_log_name=f"{cfg.algo}_{cfg.rnn_size}_{cfg.neighbor_hidden_size}_{cfg.rnn_type}_{cfg.rnn_num_layers}",
        # callback=[checkpoint_callback, eval_callback]
    )

    # 5. Save
    model.save(os.path.join(cfg.logdir, "final_model"))
    env.close()
    eval_env.close()

def parameter_sweep():
    cfg = QuadrotorEnvConfig()
    cfg.logdir = "./quad_experiment2"
    cfg.num_envs = 12
    cfg.rnn_size = 128
    cfg.neighbor_hidden_size = 128
    # cfg.use_rnn = True  # use rnn for core. False: core=identity
    cfg.rnn_type = "full"
    cfg.neighbor_encoder_type = "mlp"
    cfg.rnn_num_layers = 3
    cfg.total_timesteps = 30_000_000
    cfg.neighbor_obs_type = "dist_angle_heading"

    args = parse_args_from_cfg(cfg)
    update_cfg_from_args(cfg, args)
    for rnn_num_layers in [3, 6]:
        for rnn_size in [128]:
            for neighbor_encoder_type in ["attention"]:
                for neighbor_obs_type in ["dist_angle", "dist_angle_heading"]:
                    cfg.neighbor_obs_type = neighbor_obs_type
                    cfg.rnn_size = rnn_size
                    cfg.rnn_num_layers = rnn_num_layers
                    cfg.neighbor_encoder_type = neighbor_encoder_type
                    train(cfg)
    # train()

def main():
    parameter_sweep()

if __name__ == "__main__":
    sys.exit(main())
