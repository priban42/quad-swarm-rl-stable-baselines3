import copy
import torch
import gymnasium as gym
from gym_art.quadrotor_multi.quadrotor_multi import QuadrotorEnvMulti
from gym_art.quadrotor_multi.quad_experience_replay import ExperienceReplayWrapper
from swarm_rl.env_wrappers.reward_shaping import DEFAULT_QUAD_REWARD_SHAPING, QuadsRewardShapingWrapper
from swarm_rl.env_wrappers.v_value_map import V_ValueMapWrapper
from swarm_rl.env_wrappers.compatibility import QuadEnvCompatibility
from swarm_rl.env_wrappers.MetaQuadFactory import MetaQuadFactory
from swarm_rl.global_cfg import QuadrotorEnvConfig

class AnnealSchedule:
    def __init__(self, coeff_name, final_value, anneal_env_steps):
        self.coeff_name = coeff_name
        self.final_value = final_value
        self.anneal_env_steps = anneal_env_steps

class SB3QuadrotorEnv(gym.Env):
    """
    SB3-compatible quadrotor swarm environment builder.
    Converts SampleFactory's cfg-driven setup into a plain Gym env.
    """

    def __init__(self, cfg:QuadrotorEnvConfig, shared_param=None):
        self.cfg = cfg
        self.shared_param = shared_param
        # self.seed = seed
        self.env = self._make_env(self.cfg, self.shared_param)

        # self.device = device
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def _make_env(self,cfg:QuadrotorEnvConfig, shared_param):

        # from gym_art.quadrotor_multi.quadrotor_multi_controller import QuadrotorEnvMulti
        from gym_art.quadrotor_multi.quadrotor_multi_rewards import QuadrotorEnvMulti

        rew_coeff = DEFAULT_QUAD_REWARD_SHAPING["quad_rewards"]

        env = QuadrotorEnvMulti(cfg=cfg, shared_param=shared_param)

        # --- 2. Optional wrappers (same as before) ---
        if cfg.use_replay_buffer:
            env = ExperienceReplayWrapper(env, 0.5, cfg.obst_density, cfg.obst_size, False, False, False, 0, 1, 0.1, 1.0)

        return env

    # --- SB3-required API ---
    def reset(self, seed=None, options=None):
        # obs, info = self.env.reset(seed=seed, options=options)
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        obs, reward, terminated, info = self.env.step(action)
        # return obs, reward, terminated, info  # custom vec env
        return obs, reward, terminated, terminated, info  # vec env

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
