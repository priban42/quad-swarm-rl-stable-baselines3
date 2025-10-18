import numpy as np
from swarm_rl.env_wrappers.reward_shaping import DEFAULT_QUAD_REWARD_SHAPING, QuadsRewardShapingWrapper
from swarm_rl.env_wrappers.MetaEnv import MetaEnv

class MetaQuadFactory:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            print("MetaQuadFactory instance created")
        return cls._instance

    def initialize(self, seed=0, num_agents=1,
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
        quads_render=False,):

        self.quad = "Crazyflie"
        self.dynamics_params=self.quad
        self.dynamics_randomize_every = None
        self.dyn_randomization_ratio = None
        self.raw_control = True
        self.raw_control_zero_middle = True
        self.sense_noise = "default"
        self.room_dims = [10, 10, 3]
        self.obst_spawn_area = [(-4, 4), (-4, 4), (0, 3)]
        self.quads_view_mode = ["global"]

        self.__quads_assigned = 0

        self.num_agents = num_agents
        self.meta_envs = []
        self.rng = np.random.default_rng(seed=seed)

        self.seed = seed
        self.num_agents = num_agents
        self.ep_time = episode_duration
        self.obs_repr = obs_repr
        self.neighbor_visible_num = neighbor_visible_num
        self.neighbor_obs_type = neighbor_obs_type
        self.collision_hitbox_radius = collision_hitbox_radius
        self.collision_falloff_radius = collision_falloff_radius
        self.use_obstacles = use_obstacles
        self.obst_density = obst_density
        self.obst_size = obst_size
        self.use_replay_buffer = use_replay_buffer
        self.quads_mode = quads_mode
        self.use_downwash = use_downwash
        self.use_numba = use_numba
        self.render_mode = render_mode
        self.quads_render = quads_render
        self.anneal_collision_steps = anneal_collision_steps
        self.quads_collision_reward = quads_collision_reward
        self.quads_collision_smooth_max_penalty = quads_collision_smooth_max_penalty
        self.quads_obst_collision_reward = quads_obst_collision_reward
        self.visualize_v_value = visualize_v_value
        self.device = device
        self.checkpoint_path = checkpoint_path

        self.dynamics_change = dict(
            noise=dict(thrust_noise_ratio=0.05),
            damp=dict(vel=0, omega_quadratic=0)
        )

        self.dyn_sampler_1 = None
        if self.dyn_randomization_ratio is not None:
            self.dyn_sampler_1 = dict(type="RelativeSampler", noise_ratio=self.dyn_randomization_ratio, sampler="normal")

        self.rew_coeff = DEFAULT_QUAD_REWARD_SHAPING["quad_rewards"]

        self.init_random_state = False

        self.apply_collision_force = True

        if neighbor_visible_num == -1:
            self.num_use_neighbor_obs = self.num_agents - 1
        else:
            self.num_use_neighbor_obs = neighbor_visible_num

        print("MetaQuadFactory instance initialized")

    def __create_meta_env(self):
        meta_env = MetaEnv(self)
        return meta_env

    def assign_meta_env(self, child):
        if self.__quads_assigned % self.num_agents == 0:
            env = self.__create_meta_env()
            self.meta_envs.append(env)
        else:
            env = self.meta_envs[self.__quads_assigned//self.num_agents]
        index = self.__quads_assigned % self.num_agents
        env.envs.append(child)
        self.__quads_assigned += 1
        return env, index

def main():
    a = MetaQuadFactory()
    # a.initialize(4, 0)
    pass

if __name__ == "__main__":
    main()