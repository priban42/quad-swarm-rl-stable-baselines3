import copy
import torch
import gymnasium as gym
from gym_art.quadrotor_multi.quadrotor_multi import QuadrotorEnvMulti
from gym_art.quadrotor_multi.quad_experience_replay import ExperienceReplayWrapper
from swarm_rl.env_wrappers.reward_shaping import DEFAULT_QUAD_REWARD_SHAPING, QuadsRewardShapingWrapper
from swarm_rl.env_wrappers.v_value_map import V_ValueMapWrapper
from swarm_rl.env_wrappers.compatibility import QuadEnvCompatibility
from swarm_rl.env_wrappers.MetaQuadFactory import MetaQuadFactory

class DummyCfg:
    """Minimal config mock to satisfy QuadrotorEnvMulti requirements."""
    def __init__(self, seed=0):
        self.seed = seed
        # Add other expected fields with reasonable defaults
        self.num_envs = 4
        self.device = "cpu"
        self.train_dir = "./sb_train_dir"

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

    def __init__(
        self,
        seed=None,
        num_agents=1,
        episode_duration=15.0,
        obs_repr="xyz_vxyz_R_omega",
        neighbor_visible_num=-1,
        neighbor_obs_type="pos",
        collision_hitbox_radius=2.0,
        collision_falloff_radius=4.0,
        use_obstacles=False,
        obst_density=0.2,
        obst_size=1.0,
        use_replay_buffer=False,
        quads_mode="mix",
        use_downwash=False,
        use_numba=True,
        render_mode="rgb_array",
        anneal_collision_steps=300000000,
        quads_collision_reward=5.0,
        quads_collision_smooth_max_penalty=10.0,
        quads_obst_collision_reward=0.0,
        visualize_v_value=False,
        device="cpu",
        checkpoint_path=None,
        quads_render=False,
        **kwargs
    ):
        self.seed = seed
        self.env = self._make_env(
            num_agents, episode_duration, obs_repr, neighbor_visible_num, neighbor_obs_type,
            collision_hitbox_radius, collision_falloff_radius,
            use_obstacles, obst_density, obst_size,
            use_replay_buffer, quads_mode, use_downwash, use_numba, render_mode,quads_render,
            anneal_collision_steps,
            quads_collision_reward, quads_collision_smooth_max_penalty, quads_obst_collision_reward,
            visualize_v_value, device, checkpoint_path
        )

        self.device = device
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def _make_env(self, num_agents, episode_duration, obs_repr, neighbor_visible_num, neighbor_obs_type,
                  collision_hitbox_radius, collision_falloff_radius,
                  use_obstacles, obst_density, obst_size,
                  use_replay_buffer, quads_mode, use_downwash, use_numba, render_mode,quads_render,
                  anneal_collision_steps, quads_collision_reward, quads_collision_smooth_max_penalty,
                  quads_obst_collision_reward, visualize_v_value, device, checkpoint_path):

        from gym_art.quadrotor_multi.quadrotor_multi_controller import QuadrotorEnvMulti

        # --- 1. Base env ---
        quad = "Crazyflie"
        dyn_randomize_every = None
        dyn_randomization_ratio = None
        raw_control = True
        raw_control_zero_middle = True
        sense_noise = "default"
        room_dims = [10, 10, 3]
        obst_spawn_area = [(-4, 4), (-4, 4), (0, 3)]
        quads_view_mode = ["global"]
        # quads_render = False

        sampler_1 = None
        if dyn_randomization_ratio is not None:
            sampler_1 = dict(type="RelativeSampler", noise_ratio=dyn_randomization_ratio, sampler="normal")

        dynamics_change = dict(
            noise=dict(thrust_noise_ratio=0.05),
            damp=dict(vel=0, omega_quadratic=0)
        )

        rew_coeff = DEFAULT_QUAD_REWARD_SHAPING["quad_rewards"]

        env = QuadrotorEnvMulti(
            cfg=DummyCfg(seed=self.seed),  # unused in this context, but required by constructor
            num_agents=num_agents,
            ep_time=episode_duration,
            rew_coeff=rew_coeff,
            obs_repr=obs_repr,
            neighbor_visible_num=neighbor_visible_num,
            neighbor_obs_type=neighbor_obs_type,
            collision_hitbox_radius=collision_hitbox_radius,
            collision_falloff_radius=collision_falloff_radius,
            use_obstacles=use_obstacles,
            obst_density=obst_density,
            obst_size=obst_size,
            obst_spawn_area=obst_spawn_area,
            room_dims=room_dims,
            use_replay_buffer=use_replay_buffer,
            quads_view_mode=quads_view_mode,
            quads_render=quads_render,
            dynamics_params=quad,
            raw_control=raw_control,
            raw_control_zero_middle=raw_control_zero_middle,
            dynamics_randomize_every=dyn_randomize_every,
            dynamics_change=dynamics_change,
            dyn_sampler_1=sampler_1,
            sense_noise=sense_noise,
            init_random_state=False,
            use_downwash=use_downwash,
            use_numba=use_numba,
            quads_mode=quads_mode,
            render_mode=render_mode,
        )

        # --- 2. Optional wrappers (same as before) ---
        if use_replay_buffer:
            env = ExperienceReplayWrapper(env, 0.5, obst_density, obst_size, False, False, False, 0, 1, 0.1, 1.0)

        reward_shaping = copy.deepcopy(DEFAULT_QUAD_REWARD_SHAPING)

        reward_shaping['quad_rewards']['quadcol_bin'] = quads_collision_reward
        reward_shaping['quad_rewards']['quadcol_bin_smooth_max'] = quads_collision_smooth_max_penalty
        reward_shaping['quad_rewards']['quadcol_bin_obst'] = quads_obst_collision_reward

        # this is annealed by the reward shaping wrapper
        if anneal_collision_steps > 0:
            reward_shaping['quad_rewards']['quadcol_bin'] = 0.0
            reward_shaping['quad_rewards']['quadcol_bin_smooth_max'] = 0.0
            reward_shaping['quad_rewards']['quadcol_bin_obst'] = 0.0
            annealing = [
                AnnealSchedule('quadcol_bin', quads_collision_reward, anneal_collision_steps),
                AnnealSchedule('quadcol_bin_smooth_max', quads_collision_smooth_max_penalty,
                               anneal_collision_steps),
                AnnealSchedule('quadcol_bin_obst', quads_obst_collision_reward, anneal_collision_steps),
            ]
        else:
            annealing = None

        env = QuadsRewardShapingWrapper(env, reward_shaping_scheme=reward_shaping, annealing=annealing,
                                        with_pbt=False)

        # reward_shaping = copy.deepcopy(DEFAULT_QUAD_REWARD_SHAPING)
        # reward_shaping["quad_rewards"].update({
        #     "quadcol_bin": quads_collision_reward,
        #     "quadcol_bin_smooth_max": quads_collision_smooth_max_penalty,
        #     "quadcol_bin_obst": quads_obst_collision_reward,
        # })
        # env = QuadsRewardShapingWrapper(env, reward_shaping_scheme=reward_shaping, annealing=None)

        # env = QuadEnvCompatibility(env)

        # if visualize_v_value and checkpoint_path:
        #     from sample_factory.model.actor_critic import create_actor_critic
        #     from sample_factory.algo.learning.learner import Learner
        #
        #     actor_critic = create_actor_critic(None, env.observation_space, env.action_space)
        #     actor_critic.eval()
        #
        #     device = torch.device(device)
        #     actor_critic.model_to_device(device)
        #     checkpoint_dict = Learner.load_checkpoint(checkpoint_path, device)
        #     actor_critic.load_state_dict(checkpoint_dict["model"])
        #     env = V_ValueMapWrapper(env, actor_critic)

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
