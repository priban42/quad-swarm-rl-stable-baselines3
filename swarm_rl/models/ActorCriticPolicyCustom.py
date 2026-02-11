import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import make_proba_distribution, SquashedDiagGaussianDistribution
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from swarm_rl.models.quad_multi_model import QuadMultiEncoder
from sample_factory.model.core import ModelCoreIdentity
from sample_factory.model.decoder import MlpDecoder
from sample_factory.utils.typing import Config

from dataclasses import dataclass, field
from typing import List

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)

from sample_factory.model.action_parameterization import (
    ActionParameterizationContinuousNonAdaptiveStddev,
    ActionParameterizationDefault,
)
from sample_factory.algo.utils.action_distributions import is_continuous_action_space, sample_actions_log_probs
from sample_factory.utils.normalize import ObservationNormalizer

@dataclass
class QuadrotorEnvConfig:
    # Quadrotor features

    dim_mode = "2D_horizontal"

    learning_rate = 0.0001
    normalize_input = False

    decoder_mlp_layers=[]
    adaptive_stddev = False
    initial_stddev = 1.0
    continuous_tanh_scale = 1.0

    policy_init_gain = 1.0
    nonlinearity = 'tanh'
    encoder_type = 'mlp'
    encoder_subtype = 'mlp_quads'
    rnn_size = 256
    encoder_extra_fc_layers = 0
    env_frameskip = 1

    quads_num_agents: int = 4
    # quads_obs_repr: str = 'xyz_vxyz_R_omega'
    quads_obs_repr: str = 'aw_awdot_dist_distdot_angle_angledot'
    quads_obs_repr: str = 'cdist_cdistdot_dist_distdot_angle_angledot'
    quads_episode_duration: float = 15.0
    quads_encoder_type: str = 'corl'

    # Neighbor
    quads_neighbor_visible_num: int = -1
    quads_neighbor_obs_type: str = 'dist_angle'
    quads_neighbor_hidden_size: int = 256
    quads_neighbor_encoder_type: str = 'attention'

    # Neighbor Collision Reward
    quads_collision_reward: float = 0.0
    quads_collision_hitbox_radius: float = 2.0
    quads_collision_falloff_radius: float = 4.0
    quads_collision_smooth_max_penalty: float = 10.0

    # Obstacle
    quads_use_obstacles: bool = False
    quads_obstacle_obs_type: str = 'none'
    quads_obst_density: float = 0.2
    quads_obst_size: float = 1.0
    quads_obst_spawn_area: List[float] = field(default_factory=lambda: [6.0, 6.0])
    quads_domain_random: bool = False
    quads_obst_density_random: bool = False
    quads_obst_density_min: float = 0.05
    quads_obst_density_max: float = 0.2
    quads_obst_size_random: bool = False
    quads_obst_size_min: float = 0.3
    quads_obst_size_max: float = 0.6

    # Obstacle Encoder
    quads_obst_hidden_size: int = 256
    quads_obst_encoder_type: str = 'mlp'

    # Obstacle Collision Reward
    quads_obst_collision_reward: float = 0.0

    # Aerodynamics
    quads_use_downwash: bool = False

    # Numba Speed Up
    quads_use_numba: bool = True

    # Scenarios
    quads_mode: str = 'mix'

    # Room
    quads_room_dims: List[float] = field(default_factory=lambda: [10.0, 10.0, 10.0])

    # Replay Buffer
    replay_buffer_sample_prob: float = 0.75

    # Annealing
    anneal_collision_steps: float = 300000000

    # Rendering
    quads_view_mode: List[str] = field(default_factory=lambda: ['topdown', 'chase', 'global'])
    quads_render: bool = False
    visualize_v_value: bool = False

    # Sim2Real
    quads_sim2real: bool = False

class ActorCriticPolicyCustomSharedWeights(ActorCriticPolicy):
    """
    Custom Actor-Critic policy for SB3 PPO using the QuadMultiEncoder + SampleFactory core/decoder.
    """

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, net_arch=[], **kwargs)
        self.cfg = QuadrotorEnvConfig()
        # --- Custom modules ---
        self.encoder = QuadMultiEncoder(self.cfg, observation_space)
        encoder_output_size = self.encoder.get_out_size()

        self.core = ModelCoreIdentity(self.cfg, encoder_output_size)
        self.decoder = MlpDecoder(self.cfg, encoder_output_size)

        decoder_out_size: int = self.decoder.get_out_size()
        self.action_parameterization = self.get_action_parameterization(decoder_out_size)

        # Critic head
        self.value_net = nn.Linear(self.decoder.get_out_size(), 1)

        # Action distribution (for stochastic policy)
        self.dist = make_proba_distribution(action_space)

        self.all_params = []
        self.all_params += list(self.encoder.self_encoder.parameters())  # or whatever internal module they wrap
        self.all_params += list(self.encoder.feed_forward.parameters())
        if self.encoder.neighbor_encoder is not None:
            self.all_params += list(self.encoder.neighbor_encoder.parameters(recurse=True))  # or whatever internal module they wrap
        if self.encoder.use_obstacles:
            self.all_params += list(self.encoder.obstacle_encoder.parameters(recurse=True))
        self.all_params += list(self.encoder.parameters())  # or whatever internal module they wrap
        self.all_params += list(self.decoder.mlp.parameters())
        self.all_params += self.value_net.parameters()

        self.initialize_weights(self.encoder.self_encoder)  # or whatever internal module they wrap
        self.initialize_weights(self.encoder.feed_forward)
        if self.encoder.neighbor_encoder is not None:
            self.initialize_weights(self.encoder.neighbor_encoder)  # or whatever internal module they wrap
        if self.encoder.use_obstacles:
            self.initialize_weights(self.encoder.obstacle_encoder)
        self.initialize_weights(self.encoder)  # or whatever internal module they wrap
        self.initialize_weights(self.decoder.mlp)
        self.initialize_weights(self.value_net)

        # Register all submodules
        # self._build(lr_schedule)

        # Store config and dimensions

        self._features_dim = encoder_output_size
        self.optimizer = torch.optim.Adam(self.all_params, lr=lr_schedule(1.0))

    def initialize_weights(self, layer):
        # gain = nn.init.calculate_gain(self.cfg.nonlinearity)
        gain = self.cfg.policy_init_gain
        if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
            nn.init.xavier_uniform_(layer.weight.data, gain=gain)


    def get_action_parameterization(self, decoder_output_size: int):
        # action_parameterization = ActionParameterizationDefault(self.cfg, decoder_output_size, self.action_space)
        action_parameterization = ActionParameterizationContinuousNonAdaptiveStddev(
            self.cfg,
            decoder_output_size,
            self.action_space,
        )
        return action_parameterization

    def _build(self, lr_schedule):
        pass
        """Initialize optimizer etc. (standard SB3 build logic)."""
        # self.optimizer = torch.optim.Adam(self.all_params, lr=lr_schedule(1.0))

    # --- SB3 API methods ---
    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward encoder."""
        if isinstance(obs, torch.Tensor):
            obs_dict = {'obs': obs}
        else:
            obs_dict = obs
        features = self.encoder(obs_dict)
        return features

    def _predict(self, obs: torch.Tensor, deterministic=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward encoder, core, and decoder."""
        casted_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        features = self.extract_features(casted_obs)
        features, _ = self.core(features, None)
        decoder_output = self.decoder(features)
        values = self.value_net(decoder_output)
        action_distribution_params, self.last_action_distribution = self.action_parameterization(decoder_output)
        if deterministic:
            actions = self.last_action_distribution.mean
            return actions
        actions, log_prob = sample_actions_log_probs(self.last_action_distribution)
        return actions

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Used by PPO during rollout collection."""
        casted_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        features = self.extract_features(casted_obs)
        features, _ = self.core(features, None)
        decoder_output = self.decoder(features)
        values = self.value_net(decoder_output)
        action_distribution_params, self.last_action_distribution = self.action_parameterization(decoder_output)
        actions, log_prob = sample_actions_log_probs(self.last_action_distribution)
        return actions, values, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Used by PPO during training (to get log_probs and entropy)."""
        # latent_pi, values = self._predict(obs)
        # dist = self.dist.proba_distribution(logits=latent_pi)
        # log_prob = dist.log_prob(actions)
        # action_distribution_params, self.last_action_distribution = self.action_parameterization(latent_pi)
        # actions, log_prob = sample_actions_log_probs(self.last_action_distribution)
        # entropy = dist.entropy()
        casted_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        features = self.extract_features(casted_obs)
        features, _ = self.core(features, None)
        decoder_output = self.decoder(features)
        values = self.value_net(decoder_output)
        action_distribution_params, self.last_action_distribution = self.action_parameterization(decoder_output)
        log_prob = self.last_action_distribution.log_prob(actions)
        entropy = self.last_action_distribution.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Critic-only forward."""
        casted_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        features = self.extract_features(casted_obs)
        features, _ = self.core(features, None)
        decoder_output = self.decoder(features)
        values = self.value_net(decoder_output)
        return values

class ActorCriticPolicyCustomSeparateWeights(ActorCriticPolicy):
    """
    Custom Actor-Critic policy for SB3 PPO using the QuadMultiEncoder + SampleFactory core/decoder.
    """

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, net_arch=[], **kwargs)
        self.cfg = QuadrotorEnvConfig()
        # --- Custom modules ---
        self.actor_encoder = QuadMultiEncoder(self.cfg, observation_space)
        self.actor_core = ModelCoreIdentity(self.cfg, self.actor_encoder.get_out_size())
        self.actor_decoder = MlpDecoder(self.cfg, self.actor_core.get_out_size())

        self.critic_encoder = QuadMultiEncoder(self.cfg, observation_space)
        self.critic_core = ModelCoreIdentity(self.cfg, self.critic_encoder.get_out_size())
        self.critic_decoder = MlpDecoder(self.cfg, self.critic_core.get_out_size())

        # self.obs_normalizer: ObservationNormalizer = ObservationNormalizer(observation_space, self.cfg)

        self.log_std_init = 0.0
        use_sde = False
        dist_kwargs = None
        if use_sde:
            dist_kwargs = {
                "full_std": True,
                "squash_output": False,
                "use_expln": False,
                "learn_features": False,
            }
        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs
        # self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)
        self.action_dist = SquashedDiagGaussianDistribution(int(np.prod(action_space.shape)), **{})
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(latent_dim=self.actor_decoder.get_out_size(), log_std_init=self.log_std_init)
        # self.action_parameterization = self.get_action_parameterization(self.critic_decoder.get_out_size())

        # Critic head
        self.value_net = nn.Linear(self.critic_decoder.get_out_size(), 1)

        # Action distribution (for stochastic policy)
        # self.dist = make_proba_distribution(action_space)

        self.all_params = self._get_all_params()
        self._initialize_all_weights()

        self._features_dim = self.actor_encoder.get_out_size()
        self.optimizer = torch.optim.Adam(self.all_params, lr=lr_schedule(1.0))

        self.ortho_init = False
        self.optimizer_class = type(self.optimizer)
        self.share_features_extractor = False
        self.features_extractor = None

        pass

    def _get_all_params(self):
        all_params = []
        all_params += list(self.actor_encoder.self_encoder.parameters())  # or whatever internal module they wrap
        all_params += list(self.actor_encoder.feed_forward.parameters())
        if self.actor_encoder.neighbor_encoder is not None:
            all_params += list(self.actor_encoder.neighbor_encoder.embedding_mlp.parameters())
            all_params += list(self.actor_encoder.neighbor_encoder.neighbor_value_mlp.parameters())
            all_params += list(self.actor_encoder.neighbor_encoder.attention_mlp.parameters())
        # if self.actor_encoder.use_obstacles:
        #     all_params += list(self.actor_encoder.obstacle_encoder.parameters(recurse=True))
        # all_params += list(self.actor_encoder.parameters())  # or whatever internal module they wrap
        all_params += list(self.actor_decoder.mlp.parameters())
        all_params += self.value_net.parameters()

        all_params += list(self.critic_encoder.self_encoder.parameters())  # or whatever internal module they wrap
        all_params += list(self.critic_encoder.feed_forward.parameters())
        if self.critic_encoder.neighbor_encoder is not None:
            all_params += list(self.critic_encoder.neighbor_encoder.embedding_mlp.parameters())
            all_params += list(self.critic_encoder.neighbor_encoder.neighbor_value_mlp.parameters())
            all_params += list(self.critic_encoder.neighbor_encoder.attention_mlp.parameters())
        # if self.critic_encoder.use_obstacles:
        #     all_params += list(self.critic_encoder.obstacle_encoder.parameters(recurse=True))
        # all_params += list(self.critic_encoder.parameters())  # or whatever internal module they wrap
        all_params += list(self.critic_decoder.mlp.parameters())

        # all_params += self.action_parameterization.learned_stddev
        # all_params += self.action_parameterization.distribution_linear.parameters()
        all_params += self.action_net.parameters()
        all_params.append(self.log_std)
        # all_params += nn.Parameter(torch.ones(4) * 1)

        return all_params

    def _initialize_all_weights(self):
        self.initialize_weights(self.actor_encoder.self_encoder)  # or whatever internal module they wrap
        self.initialize_weights(self.actor_encoder.feed_forward)
        if self.actor_encoder.neighbor_encoder is not None:
            self.initialize_weights(self.actor_encoder.neighbor_encoder.embedding_mlp)  # or whatever internal module they wrap
            self.initialize_weights(self.actor_encoder.neighbor_encoder.neighbor_value_mlp)  # or whatever internal module they wrap
            self.initialize_weights(self.actor_encoder.neighbor_encoder.attention_mlp)  # or whatever internal module they wrap
        # if self.actor_encoder.use_obstacles:
        #     self.initialize_weights(self.actor_encoder.obstacle_encoder)
        self.initialize_weights(self.actor_decoder.mlp)

        self.initialize_weights(self.critic_encoder.self_encoder)  # or whatever internal module they wrap
        self.initialize_weights(self.critic_encoder.feed_forward)
        if self.critic_encoder.neighbor_encoder is not None:
            self.initialize_weights(self.critic_encoder.neighbor_encoder.embedding_mlp)  # or whatever internal module they wrap
            self.initialize_weights(self.critic_encoder.neighbor_encoder.neighbor_value_mlp)  # or whatever internal module they wrap
            self.initialize_weights(self.critic_encoder.neighbor_encoder.attention_mlp)  # or whatever internal module they wrap
        # if self.critic_encoder.use_obstacles:
        #     self.initialize_weights(self.critic_encoder.obstacle_encoder)
        self.initialize_weights(self.critic_decoder.mlp)
        self.initialize_weights(self.value_net)
        # self.initialize_weights(self.action_parameterization.distribution_linear)
        self.initialize_weights(self.action_net)

    def initialize_weights(self, layer):
        # gain = nn.init.calculate_gain(self.cfg.nonlinearity)
        gain = self.cfg.policy_init_gain
        if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
            nn.init.xavier_uniform_(layer.weight.data, gain=gain)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        if isinstance(self.action_dist, SquashedDiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")

    def get_action_parameterization(self, decoder_output_size: int):
        # action_parameterization = ActionParameterizationDefault(self.cfg, decoder_output_size, self.action_space)
        action_parameterization = ActionParameterizationContinuousNonAdaptiveStddev(
            self.cfg,
            decoder_output_size,
            self.action_space,
        )
        return action_parameterization

    def _build(self, lr_schedule):
        pass
        """Initialize optimizer etc. (standard SB3 build logic)."""
        # self.optimizer = torch.optim.Adam(self.all_params, lr=lr_schedule(1.0))

    # --- SB3 API methods ---
    # def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
    #     """Forward encoder."""
    #     if isinstance(obs, torch.Tensor):
    #         obs_dict = {'obs': obs}
    #     else:
    #         obs_dict = obs
    #     features = self.encoder(obs_dict)
    #     return features

    def prepare_obs(self, obs: torch.Tensor):
        casted_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if isinstance(obs, torch.Tensor):
            obs_dict = {'obs': casted_obs}
        else:
            obs_dict = casted_obs
        # normalized_obs_dict = self.obs_normalizer(obs_dict)
        return obs_dict
        # return normalized_obs_dict

    def _predict(self, obs: torch.Tensor, deterministic=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the action according to the policy for a given observation."""
        casted_obs = self.prepare_obs(obs)
        actor_features = self.actor_encoder(casted_obs)
        latent_pi, _ =self.actor_core(actor_features, None)
        actor_decoder_output = self.actor_decoder(latent_pi)

        # action_distribution_params, self.last_action_distribution = self.action_parameterization(actor_decoder_output)
        # if deterministic:
        #     actions = self.last_action_distribution.mean
        #     actions = torch.tanh(actions)
        #     return actions
        #
        # actions, log_prob = sample_actions_log_probs(self.last_action_distribution)
        distribution = self._get_action_dist_from_latent(actor_decoder_output)
        actions = distribution.get_actions(deterministic=deterministic)
        # log_prob = distribution.log_prob(actions)
        # actions = torch.tanh(actions)
        return actions

    def extract_features(self, obs, features_extractor = None):
        casted_obs = self.prepare_obs(obs)
        actor_features = self.actor_encoder(casted_obs)
        critic_features = self.critic_encoder(casted_obs)
        return actor_features, critic_features

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass in all the networks (actor and critic)"""
        casted_obs = self.prepare_obs(obs)
        actor_features = self.actor_encoder(casted_obs)
        critic_features = self.critic_encoder(casted_obs)

        latent_pi, _ =self.actor_core(actor_features, None)
        latent_vf, _ =self.critic_core(critic_features, None)

        actor_decoder_output = self.actor_decoder(latent_pi)
        critic_decoder_output = self.critic_decoder(latent_vf)

        # action_distribution_params, self.last_action_distribution = self.action_parameterization(actor_decoder_output)
        # actions, log_prob = sample_actions_log_probs(self.last_action_distribution)
        distribution = self._get_action_dist_from_latent(actor_decoder_output)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        values = self.value_net(critic_decoder_output)
        # self.action_parameterization.learned_stddev -= 0.00001
        # actions = torch.tanh(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions according to the current policy,
        given the observations."""
        casted_obs = self.prepare_obs(obs)
        actor_features = self.actor_encoder(casted_obs)
        critic_features = self.critic_encoder(casted_obs)

        latent_pi, _ =self.actor_core(actor_features, None)
        latent_vf, _ =self.critic_core(critic_features, None)

        actor_decoder_output = self.actor_decoder(latent_pi)
        critic_decoder_output = self.critic_decoder(latent_vf)

        values = self.value_net(critic_decoder_output)

        distribution = self._get_action_dist_from_latent(actor_decoder_output)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # action_distribution_params, self.last_action_distribution = self.action_parameterization(actor_decoder_output)
        # unsquashed = torch.atanh(actions.clamp(-0.999999, 0.999999))
        # log_prob = self.last_action_distribution.log_prob(actions)
        # log_prob -= torch.sum(torch.log(1 - actions.pow(2) + 1e-6), dim=-1)
        # entropy = self.last_action_distribution.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Get the estimated values according to the current policy given the observations., Critic-only forward."""
        casted_obs = self.prepare_obs(obs)
        critic_features = self.critic_encoder(casted_obs)
        latent_vf, _ =self.critic_core(critic_features, None)
        critic_decoder_output = self.critic_decoder(latent_vf)
        values = self.value_net(critic_decoder_output)
        return values


# class ActorCriticPolicyCustomSimple(ActorCriticPolicy):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs,
#                          features_extractor_class=CustomEncoder,
#                          features_extractor_kwargs=dict(features_dim=128))
#
#         # Replace the default MLP heads with your own decoder
#         self.actor = nn.Sequential(
#             nn.Linear(self.features_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, self.action_space.shape[0])
#         )
#         self.critic = nn.Sequential(
#             nn.Linear(self.features_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )
#
#         # Reinitialize parameters
#         self._initialize_weights(self.actor)
#         self._initialize_weights(self.critic)
