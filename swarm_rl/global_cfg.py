from dataclasses import dataclass, field
from typing import List
from dataclasses import dataclass, fields
from typing import Any, Optional


@dataclass
class QuadrotorEnvConfig:
    # Quadrotor features

    _changes = {}
    #ppo
    # PPO
    n_steps: int = 512
    batch_size: int = 1024
    n_epochs: int = 10
    gamma: float = 0.99

    # Training
    num_envs: int = 9
    total_timesteps: int = 150_000_000
    learning_rate: float = 1e-4
    logdir: str = "./PPO_4_repulsive"
    checkpoint_freq: int = 100_000
    algo: str = "ppo"
    eval_freq: int = 100_000
    eval_episodes: int = 3

    # Curriculum
    initial_capture_radius: float = 3.0
    capture_radius_decay: float = 0.95
    capture_radius_sr: float = 0.95

    dim_mode: str = "2D_horizontal"
    normalize_input: bool = False

    # NN Architecture
    decoder_mlp_layers: list = field(default_factory=list)
    adaptive_stddev: bool = False
    initial_stddev: float = 1.0
    continuous_tanh_scale: float = 1.0
    policy_init_gain: float = 1.0
    nonlinearity: str = 'tanh'
    encoder_type: str = 'mlp'
    encoder_subtype: str = 'mlp_quads'
    rnn_size: int = 256
    encoder_extra_fc_layers: int = 0
    env_frameskip: int = 1

    # Core:
    use_rnn: bool = False  # use rnn for core. False: core=identity
    rnn_type: str = None  # ["gru", "lstm"]
    rnn_num_layers: int = 2

    # Observations
    num_agents: int = 4
    obs_repr: str = 'cdist_cdistdot_dist_distdot_angle_angledot'
    episode_duration: float = 30.0

    # Neighbor
    neighbor_visible_num: int = -1
    neighbor_obs_type: str = 'dist_angle'
    neighbor_hidden_size: int = 256
    neighbor_encoder_type: str = 'attention'

    # Neighbor Collision Reward
    collision_reward: float = 5.0
    collision_hitbox_radius: float = 2.0
    collision_falloff_radius: float = 4.0
    collision_smooth_max_penalty: float = 10.0

    # Obstacle
    use_obstacles: bool = False
    obstacle_obs_type: str = 'none'
    obst_density: float = 0.2
    obst_size: float = 1.0
    obst_spawn_area: list = field(default_factory=lambda: [(-4, 4), (-4, 4), (0, 3)])
    domain_random: bool = False
    obst_density_random: bool = False
    obst_density_min: float = 0.05
    obst_density_max: float = 0.2
    obst_size_random: bool = False
    obst_size_min: float = 0.3
    obst_size_max: float = 0.6

    # Obstacle Encoder
    obst_hidden_size: int = 256
    obst_encoder_type: str = 'mlp'


    # Obstacle Collision Reward
    obst_collision_reward: float = 0.0

    # Aerodynamics
    use_downwash: bool = False

    # Numba
    use_numba: bool = True

    # Scenarios
    quads_mode: str = 'dynamic_repulsive'

    # Room
    room_dims: list = field(default_factory=lambda: [15, 15, 3])

    # Replay Buffer
    replay_buffer_sample_prob: float = 0.75
    use_replay_buffer: bool = False

    # Annealing
    anneal_collision_steps: float = 300_000_000

    # Rendering
    quads_view_mode: list = field(default_factory=lambda: ["topdown"])
    quads_render: bool = False
    visualize_v_value: bool = False
    render_mode: str = "rgb_array"

    # Sim2Real
    quads_sim2real: bool = False

    # Misc
    seed: Optional[int] = None
    thrust_noise_ratio: float = 0.05
    device: str = "cuda"
    checkpoint_path: Optional[str] = None
    train_dir: str = "./sb_train_dir"
    # quad: str = "Crazyflie"
    sense_noise: str = "default"

    # Control
    raw_control: bool = True
    raw_control_zero_middle: bool = True
    tf_control: bool = False

    # Dynamics
    dynamics_params: str = "Crazyflie"
    dynamics_change: Optional[Any] = None
    dynamics_randomize_every: Optional[Any] = None
    dyn_sampler_1: Optional[Any] = None
    dyn_sampler_2: Optional[Any] = None
    dynamics_simplification: bool = False

    # Simulation
    sim_freq: float = 200.0
    sim_steps: int = 2
    init_random_state: bool = False
    verbose: bool = False
    gravity: float = 9.81
    t2w_std: float = 0.005
    t2t_std: float = 0.0005
    excite: bool = False


    def to_dict(self) -> dict[str, Any]:
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if not isinstance(value, (int, float, str)):
                value = str(value)
            result[f.name] = value
        return result

    def __setattr__(self, name, value):
        if hasattr(self, name):
            old_value = getattr(self, name)
            if old_value != value:
                if name != "__changes":
                    self._changes[name] = old_value
        super().__setattr__(name, value)

    def to_string(self):
        result = ""
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name != "__changes":
                b = ""
                if type(value) == str:
                    b = "\""
                if f.name in self._changes:
                    result += f"{f.name} = {b}{value}{b}  # {b}{str(self._changes[f.name])}{b}\n"
                else:
                    result += f"{f.name} = {b}{value}{b}\n"
        return result