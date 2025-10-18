import copy
import time
from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np

from gym_art.quadrotor_multi.aerodynamics.downwash import perform_downwash
from gym_art.quadrotor_multi.collisions.obstacles import perform_collision_with_obstacle
from gym_art.quadrotor_multi.collisions.quadrotors import calculate_collision_matrix, \
    calculate_drone_proximity_penalties, perform_collision_between_drones
from gym_art.quadrotor_multi.collisions.room import perform_collision_with_wall, perform_collision_with_ceiling
from gym_art.quadrotor_multi.obstacles.utils import get_cell_centers
from gym_art.quadrotor_multi.quad_utils import QUADS_OBS_REPR, QUADS_NEIGHBOR_OBS_TYPE

from gym_art.quadrotor_multi.obstacles.obstacles import MultiObstacles
from gym_art.quadrotor_multi.quadrotor_multi_visualization import Quadrotor3DSceneMulti
from gym_art.quadrotor_multi.quadrotor_single import QuadrotorSingle
from gym_art.quadrotor_multi.scenarios.mix import create_scenario
from sample_factory.utils.utils import experiment_dir

from swarm_rl.env_wrappers.MetaQuadFactory import MetaQuadFactory
from swarm_rl.env_wrappers.MetaEnv import MetaEnv

import pickle

class QuadrotorEnvInstance(gym.Env):

    def __init__(self):
        super().__init__()
        self.mqf = MetaQuadFactory()
        self.meta_env, self.meta_index = self.mqf.assign_meta_env(self)
        self.rng = self.mqf.rng
        self.num_agents = self.mqf.num_agents
        self.device = self.mqf.device

        self.env = QuadrotorSingle(
                # Quad Parameters
                dynamics_params=self.mqf.dynamics_params, dynamics_change=self.mqf.dynamics_change,
                dynamics_randomize_every=self.mqf.dynamics_randomize_every, dyn_sampler_1=self.mqf.dyn_sampler_1,
                raw_control=self.mqf.raw_control, raw_control_zero_middle=self.mqf.raw_control_zero_middle, sense_noise=self.mqf.sense_noise,
                init_random_state=self.mqf.init_random_state, obs_repr=self.mqf.obs_repr, ep_time=self.mqf.ep_time, room_dims=self.mqf.room_dims,
                use_numba=self.mqf.use_numba,
                # Neighbor
                num_agents=self.mqf.num_agents,
                neighbor_obs_type=self.mqf.neighbor_obs_type, num_use_neighbor_obs=self.mqf.num_use_neighbor_obs,
                # Obstacle
                use_obstacles=self.mqf.use_obstacles,
                rng=self.rng,
            )
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.quad_arm = self.env.dynamics.arm
        self.control_freq = self.env.control_freq
        self.control_dt = 1.0 / self.control_freq

        if self.meta_index == (self.mqf.num_agents - 1):
            self.meta_env.init_scenario()

    def reset(self, seed=None, options=None, obst_density=None, obst_size=None):
        obs, rewards, dones, infos = [], [], [], []

        self.meta_env.reset()
        self.env.goal = self.meta_env.scenario.goals[self.meta_index]
        if self.meta_env.scenario.spawn_points is None:
            self.env.spawn_point = self.meta_env.scenario.goals[self.meta_index]
        else:
            self.env.spawn_point = self.meta_env.scenario.spawn_points[self.meta_index]
        self.env.rew_coeff = self.meta_env.rew_coeff
        observation = self.env.reset()
        obs = observation
        self.meta_env.pos[self.meta_index, :]= self.env.dynamics.pos

        if self.mqf.num_use_neighbor_obs > 0:
            obs = self.meta_env.add_neighborhood_obs(obs, self.meta_index)

        return obs, {}

    def step(self, actions):
        self.meta_env.step()
        observation, reward, done, info = self.env.step(actions)
        reward += self.meta_env.rewards[self.meta_index]
        info = self.meta_env.merge_dicts_recursive(info, self.meta_env.infos[self.meta_index])
        self.meta_env.pos[self.meta_index, :] = self.env.dynamics.pos
        if self.mqf.num_use_neighbor_obs > 0:
            observation = np.concatenate((observation, self.meta_env.neighbour_observations[self.meta_index]), axis=0)
        return observation, reward, done, done, info
