import numpy as np
import copy
from gym_art.quadrotor_multi.scenarios.mix import create_scenario
from gym_art.quadrotor_multi.quad_utils import QUADS_OBS_REPR, QUADS_NEIGHBOR_OBS_TYPE
from gym_art.quadrotor_multi.collisions.quadrotors import calculate_collision_matrix, \
    calculate_drone_proximity_penalties, perform_collision_between_drones

from gym_art.quadrotor_multi.collisions.room import perform_collision_with_wall, perform_collision_with_ceiling

from gym_art.quadrotor_multi.aerodynamics.downwash import perform_downwash
# from swarm_rl.env_wrappers.MetaQuadFactory import MetaQuadFactory

class MetaEnv:
    def __init__(self, meta_quad_factory):
        self.mqf = meta_quad_factory
        self.envs = []

        if self.mqf.neighbor_visible_num == -1:
            self.num_use_neighbor_obs = self.mqf.num_agents - 1
        else:
            self.num_use_neighbor_obs = self.mqf.neighbor_visible_num

        self.rew_coeff = dict(
            pos=1., effort=0.05, action_change=0., crash=1., orient=1., yaw=0., rot=0., attitude=0., spin=0.1, vel=0.,
            quadcol_bin=5., quadcol_bin_smooth_max=4., quadcol_bin_obst=5.
        )
        rew_coeff_orig = copy.deepcopy(self.rew_coeff)
        for key in self.rew_coeff.keys():
            self.rew_coeff[key] = float(self.rew_coeff[key])
        orig_keys = list(rew_coeff_orig.keys())
        # Checking to make sure we didn't provide some false rew_coeffs (for example by misspelling one of the params)
        assert np.all([key in orig_keys for key in self.rew_coeff.keys()])

        # # Collisions: Room
        self.collisions_room_per_episode = 0
        self.collisions_floor_per_episode = 0
        self.collisions_wall_per_episode = 0
        self.collisions_ceiling_per_episode = 0

        self.prev_crashed_walls = []
        self.prev_crashed_ceiling = []
        self.prev_crashed_room = []

        if self.mqf.quads_render:
            self.simulation_start_time = 0
            self.frames_since_last_render = self.render_skip_frames = 0
            self.render_every_nth_frame = 1
            # # Use this to control rendering speed
            self.render_every_nth_frame = 4
            self.allow_skip_frames = False
            self.all_collisions = {}

        self.pos = np.zeros([self.mqf.num_agents, 3])
        self.vel = np.zeros([self.mqf.num_agents, 3])
        self.acc = np.zeros([self.mqf.num_agents, 3])
        self.rot = np.stack([np.eye(3) for i in range(self.mqf.num_agents)])
        self.omega = np.zeros([self.mqf.num_agents, 3])
        self.rel_pos = np.zeros((self.mqf.num_agents, self.mqf.num_agents, 3))
        self.rel_vel = np.zeros((self.mqf.num_agents, self.mqf.num_agents, 3))

        self.step_call = 0
        self.reset_call = 0

        self.neighbour_observations = None
        self.rewards = None
        self.infos = None

    def init_scenario(self):
        self.scenario = create_scenario(quads_mode=self.mqf.quads_mode, envs=[e.env for e in self.envs], num_agents=self.mqf.num_agents,
                                        room_dims=self.mqf.room_dims, rng = self.mqf.rng)

        obs_self_size = QUADS_OBS_REPR[self.mqf.obs_repr]
        self.observation_space = self.envs[0].observation_space
        neighbor_obs_size = QUADS_NEIGHBOR_OBS_TYPE[self.mqf.neighbor_obs_type]
        self.clip_neighbor_space_length = self.num_use_neighbor_obs * neighbor_obs_size
        self.clip_neighbor_space_min_box = self.observation_space.low[
                                           obs_self_size:obs_self_size + self.clip_neighbor_space_length]
        self.clip_neighbor_space_max_box = self.observation_space.high[
                                           obs_self_size:obs_self_size + self.clip_neighbor_space_length]

        self.quad_arm = self.envs[0].env.dynamics.arm
        self.control_freq = self.envs[0].env.control_freq
        self.control_dt = 1.0 / self.control_freq

        # Collisions
        # # Collisions: Neighbors
        self.collisions_per_episode = 0
        # # # Ignore collisions because of spawn
        self.collisions_after_settle = 0
        self.collisions_grace_period_steps = 1.5 * self.control_freq
        self.collisions_grace_period_seconds = 1.5
        self.prev_drone_collisions = []

        self.collisions_final_grace_period_steps = 5.0 * self.control_freq
        self.collisions_final_5s = 0

        # # # Dense reward info
        self.collision_threshold = self.mqf.collision_hitbox_radius * self.quad_arm
        self.collision_falloff_threshold = self.mqf.collision_falloff_radius * self.quad_arm


    def reset(self):
        # execute this method only once per single step in every env
        self.reset_call += 1
        if (self.reset_call - 1)%self.mqf.num_agents > 0:
            return


        self.scenario.reset()

        self.collisions_per_episode = self.collisions_after_settle = self.collisions_final_5s = 0
        self.prev_drone_collisions = []

        # # Collision: Room
        self.collisions_room_per_episode = 0
        self.collisions_floor_per_episode = self.collisions_wall_per_episode = self.collisions_ceiling_per_episode = 0
        self.prev_crashed_walls = []
        self.prev_crashed_ceiling = []
        self.prev_crashed_room = []

        # Log
        # # Final Distance (1s / 3s / 5s)
        self.distance_to_goal = [[] for _ in range(len(self.envs))]
        self.agent_col_agent = np.ones(self.mqf.num_agents)
        self.agent_col_obst = np.ones(self.mqf.num_agents)
        self.reached_goal = [False for _ in range(len(self.envs))]

        if self.mqf.quads_render:
            self.reset_scene = True
            self.quads_formation_size = self.scenario.formation_size
            self.all_collisions = {val: [0.0 for _ in range(len(self.envs))] for val in ['drone', 'ground', 'obstacle']}

    def step(self):
        # execute this method only once per single step in every env
        self.step_call += 1
        if (self.step_call - 1)%self.mqf.num_agents > 0:
            return

        obs, dones, = [], []
        rewards = np.zeros(self.mqf.num_agents)
        infos = [{"rewards":{}} for _ in range(self.mqf.num_agents)]

        drone_col_matrix, curr_drone_collisions, distance_matrix = \
            calculate_collision_matrix(positions=self.pos, collision_threshold=self.collision_threshold)

        curr_drone_collisions = curr_drone_collisions.astype(int)
        curr_drone_collisions = np.delete(curr_drone_collisions, np.unique(
            np.where(curr_drone_collisions == [-1000, -1000])[0]), axis=0)

        old_quad_collision = set(map(tuple, self.prev_drone_collisions))
        new_quad_collision = np.array([x for x in curr_drone_collisions if tuple(x) not in old_quad_collision])

        self.last_step_unique_collisions = np.setdiff1d(curr_drone_collisions, self.prev_drone_collisions)
        if len(self.last_step_unique_collisions) > 0:
            print("collision detected")

        # # Filter distance_matrix; Only contains quadrotor pairs with distance <= self.collision_threshold
        near_quad_ids = np.where(distance_matrix[:, 2] <= self.collision_falloff_threshold)
        distance_matrix = distance_matrix[near_quad_ids]

        # Collision between 2 drones counts as a single collision
        # # Calculate collisions (i) All collisions (ii) collisions after grace period
        collisions_curr_tick = len(self.last_step_unique_collisions) // 2
        self.collisions_per_episode += collisions_curr_tick

        if collisions_curr_tick > 0 and self.envs[0].env.tick >= self.collisions_grace_period_steps:
            self.collisions_after_settle += collisions_curr_tick
            for agent_id in self.last_step_unique_collisions:
                self.agent_col_agent[agent_id] = 0
        if collisions_curr_tick > 0 and self.envs[0].env.time_remain <= self.collisions_final_grace_period_steps:
            self.collisions_final_5s += collisions_curr_tick

        # # Aux: Neighbor Collisions
        self.prev_drone_collisions = curr_drone_collisions

        floor_crash_list, wall_crash_list, ceiling_crash_list = self.calculate_room_collision()
        room_crash_list = np.unique(np.concatenate([floor_crash_list, wall_crash_list, ceiling_crash_list]))
        room_crash_list = np.setdiff1d(room_crash_list, self.prev_crashed_room)

        self.prev_crashed_walls = wall_crash_list
        self.prev_crashed_ceiling = ceiling_crash_list
        self.prev_crashed_room = room_crash_list

        rew_collisions_raw = np.zeros(self.mqf.num_agents)
        if self.last_step_unique_collisions.any():
            rew_collisions_raw[self.last_step_unique_collisions] = -1.0
        rew_collisions = self.rew_coeff["quadcol_bin"] * rew_collisions_raw

        # penalties for being too close to other drones
        if len(distance_matrix) > 0:
            rew_proximity = -1.0 * calculate_drone_proximity_penalties(
                distance_matrix=distance_matrix, collision_falloff_threshold=self.collision_falloff_threshold,
                dt=self.control_dt, max_penalty=self.rew_coeff["quadcol_bin_smooth_max"], num_agents=self.mqf.num_agents,
            )
        else:
            rew_proximity = np.zeros(self.mqf.num_agents)

        rew_collisions_obst_quad = np.zeros(self.mqf.num_agents)

        if self.envs[0].env.tick >= self.collisions_grace_period_steps:
            self.collisions_room_per_episode += len(room_crash_list)
            self.collisions_floor_per_episode += len(floor_crash_list)
            self.collisions_wall_per_episode += len(wall_crash_list)
            self.collisions_ceiling_per_episode += len(ceiling_crash_list)

        for i in range(self.mqf.num_agents):
            rewards[i] += rew_collisions[i]
            rewards[i] += rew_proximity[i]

            infos[i]["rewards"]["rew_quadcol"] = rew_collisions[i]
            infos[i]["rewards"]["rew_proximity"] = rew_proximity[i]
            infos[i]["rewards"]["rewraw_quadcol"] = rew_collisions_raw[i]

            # self.distance_to_goal[i].append(-infos[i]["rewards"]["rewraw_pos"])
            # if len(self.distance_to_goal[i]) >= 5 and \
            #         np.mean(self.distance_to_goal[i][-5:]) / self.envs[0].dt < self.scenario.approch_goal_metric \
            #         and not self.reached_goal[i]:
            #     self.reached_goal[i] = True

        # 3. Applying random forces: 1) aerodynamics 2) between drones 3) obstacles 4) room
        self_state_update_flag = False

        # # 1) aerodynamics
        if self.mqf.use_downwash:
            envs_dynamics = [env.env.dynamics for env in self.envs]
            applied_downwash_list = perform_downwash(drones_dyn=envs_dynamics, dt=self.control_dt)
            downwash_agents_list = np.where(applied_downwash_list == 1)[0]
            if len(downwash_agents_list) > 0:
                self_state_update_flag = True

        # # 2) Drones
        if self.mqf.apply_collision_force:
            if len(new_quad_collision) > 0:
                self_state_update_flag = True
                for val in new_quad_collision:
                    dyn1, dyn2 = self.envs[val[0]].env.dynamics, self.envs[val[1]].env.dynamics
                    dyn1.vel, dyn1.omega, dyn2.vel, dyn2.omega = perform_collision_between_drones(
                        pos1=dyn1.pos, vel1=dyn1.vel, omega1=dyn1.omega, pos2=dyn2.pos, vel2=dyn2.vel, omega2=dyn2.omega)

        # # 4) Room
        if len(wall_crash_list) > 0 or len(ceiling_crash_list) > 0:
            self_state_update_flag = True

            for val in wall_crash_list:
                perform_collision_with_wall(drone_dyn=self.envs[val].env.dynamics, room_box=self.envs[0].env.room_box)

            for val in ceiling_crash_list:
                perform_collision_with_ceiling(drone_dyn=self.envs[val].env.dynamics)

        self.scenario.step()

        for i in range(self.mqf.num_agents):
            self.pos[i, :] = self.envs[i].env.dynamics.pos
            self.vel[i, :] = self.envs[i].env.dynamics.vel
            self.acc[i, :] = self.envs[i].env.dynamics.acc
            self.rot[i, :] = self.envs[i].env.dynamics.rot

        # Rendering
        if self.mqf.quads_render:
            # Collisions with room
            ground_collisions = [1.0 if env.dynamics.on_floor else 0.0 for env in self.envs]
            obst_coll = [0.0 for _ in range(self.mqf.num_agents)]
            self.all_collisions = {'drone': drone_col_matrix, 'ground': ground_collisions,
                                   'obstacle': obst_coll}

        self.rewards = rewards
        self.infos = infos
        if self.mqf.num_use_neighbor_obs > 0:
            self.__compute_neighborhood_obs()


    def calculate_room_collision(self):
        floor_collisions = np.array([env.env.dynamics.crashed_floor for env in self.envs])
        wall_collisions = np.array([env.env.dynamics.crashed_wall for env in self.envs])
        ceiling_collisions = np.array([env.env.dynamics.crashed_ceiling for env in self.envs])

        floor_crash_list = np.where(floor_collisions >= 1)[0]

        cur_wall_crash_list = np.where(wall_collisions >= 1)[0]
        wall_crash_list = np.setdiff1d(cur_wall_crash_list, self.prev_crashed_walls)

        cur_ceiling_crash_list = np.where(ceiling_collisions >= 1)[0]
        ceiling_crash_list = np.setdiff1d(cur_ceiling_crash_list, self.prev_crashed_ceiling)

        return floor_crash_list, wall_crash_list, ceiling_crash_list

    @staticmethod
    def merge_dicts_recursive(d1, d2):
        """
        Recursively merge d2 into d1.
        Values in d2 override those in d1 unless both are dicts, in which case they are merged.
        """
        for key, value in d2.items():
            if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
                MetaEnv.merge_dicts_recursive(d1[key], value)
            else:
                d1[key] = value
        return d1

    @staticmethod
    def rotation_matrix(axis, angle):
        """
        Create a 3x3 rotation matrix from an axis and an angle using Rodrigues' formula.

        Parameters:
        axis (array-like): A 3-element array representing the axis of rotation.
        angle (float): Rotation angle in radians.


        Returns:
        numpy.ndarray: A 3x3 rotation matrix.
        """
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)  # normalize axis
        x, y, z = axis

        c = np.cos(angle)
        s = np.sin(angle)
        C = 1 - c

        R = np.array([
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C]
        ])
        return R

    def get_rel_pos_vel_item(self, env_id, indices=None):
        i = env_id

        if indices is None:
            # if not specified explicitly, consider all neighbors
            indices = [j for j in range(self.mqf.num_agents) if j != i]
        ret = np.zeros((len(indices), 0))
        if "npos" in self.mqf.neighbor_obs_type:
            cur_pos = self.pos[i]
            pos_neighbor = np.stack([self.pos[j] for j in indices])
            pos_rel = pos_neighbor - cur_pos
            for p in pos_rel:
                length = np.linalg.norm(p)
                cov = np.diag([0.002 * length, 0.002 * length, 0.02 * length])
                p_noise = np.random.multivariate_normal(np.array([0, 0, 0]), cov)
                cross = np.cross(np.array([0, 0, 1]), p[:3])
                angle = np.arccos(np.dot(np.array([0, 0, 1]), p[:3]) / (
                            np.linalg.norm(np.array([0, 0, 1])) * np.linalg.norm(p[:3])))
                R_pc = self.rotation_matrix(cross, angle)
                pos_rel + R_pc@p_noise
            ret = np.concatenate((ret, pos_rel), axis=1)
        elif "pos" in self.mqf.neighbor_obs_type:
            cur_pos = self.pos[i]
            pos_neighbor = np.stack([self.pos[j] for j in indices])
            pos_rel = pos_neighbor - cur_pos
            ret = np.concatenate((ret, pos_rel), axis=1)
        if "vel" in self.mqf.neighbor_obs_type:
            cur_vel = self.vel[i]
            vel_neighbor = np.stack([self.vel[j] for j in indices])
            vel_rel = vel_neighbor - cur_vel
            ret = np.concatenate((ret, vel_rel), axis=1)
        if "Rz" in self.mqf.neighbor_obs_type:
            cur_R = self.rot[i]
            cur_R_inv = cur_R.T
            Rz_rel = np.stack([cur_R_inv@self.rot[j][:3, 2] for j in indices])
            ret = np.concatenate((ret, Rz_rel), axis=1)
        elif "R" in self.mqf.neighbor_obs_type:
            cur_R = self.rot[i]
            cur_R_inv = cur_R.T
            R_rel = np.stack([cur_R_inv@self.rot[j] for j in indices])
            R_rel_flat = R_rel.reshape(R_rel.shape[0], -1)
            ret = np.concatenate((ret, R_rel_flat), axis=1)
        if "rng3" in self.mqf.neighbor_obs_type:
            ret = np.concatenate([ret, np.random.rand(len(indices), 3)], axis=1)
        return ret

    def neighborhood_indices(self):
        """Return a list of closest drones for each drone in the swarm."""
        # indices of all the other drones except us
        indices = [[j for j in range(self.mqf.num_agents) if i != j] for i in range(self.mqf.num_agents)]
        indices = np.array(indices)

        if self.mqf.num_use_neighbor_obs == self.mqf.num_agents - 1:
            return indices
        elif 1 <= self.mqf.num_use_neighbor_obs < self.mqf.num_agents - 1:
            close_neighbor_indices = []

            for i in range(self.mqf.num_agents):
                # rel_pos, rel_vel = self.get_rel_pos_vel_item(env_id=i, indices=indices[i])  # pribavoj
                rel_pos = self.get_rel_pos_vel_item(env_id=i, indices=indices[i])  # pribavoj
                rel_dist = np.linalg.norm(rel_pos, axis=1)
                rel_dist = np.maximum(rel_dist, 0.01)
                rel_pos_unit = rel_pos / rel_dist[:, None]

                # new relative distance is a new metric that combines relative position and relative velocity
                # the smaller the new_rel_dist, the closer the drones

                # new_rel_dist = rel_dist + np.sum(rel_pos_unit * rel_vel, axis=1)  # pribavoj
                # rel_pos_index = new_rel_dist.argsort()  # pribavoj

                rel_pos_index = rel_dist.argsort()  # pribavoj

                rel_pos_index = rel_pos_index[:self.mqf.num_use_neighbor_obs]
                close_neighbor_indices.append(indices[i][rel_pos_index])

            return close_neighbor_indices
        else:
            raise RuntimeError("Incorrect number of neigbors")

    def get_obs_neighbor_rel(self, env_id, closest_drones):
        i = env_id
        # pos_neighbors_rel, vel_neighbors_rel = self.get_rel_pos_vel_item(env_id=i, indices=closest_drones[i])
        # obs_neighbor_rel = np.concatenate((pos_neighbors_rel, vel_neighbors_rel), axis=1)
        obs_neighbor_rel = self.get_rel_pos_vel_item(env_id=i, indices=closest_drones[i])
        return obs_neighbor_rel

    def extend_obs_space(self, obs, closest_drones, index):
        obs_neighbors = []
        for i in range(len(self.envs)):
            obs_neighbor_rel = self.get_obs_neighbor_rel(env_id=i, closest_drones=closest_drones)
            obs_neighbors.append(obs_neighbor_rel.reshape(-1))
        obs_neighbors = np.stack(obs_neighbors)

        # clip observation space of neighborhoods

        obs_neighbors = np.clip(
            obs_neighbors, a_min=self.clip_neighbor_space_min_box, a_max=self.clip_neighbor_space_max_box,
        )

        obs_ext = np.concatenate((obs, obs_neighbors[index]), axis=0)
        return obs_ext



    #TODO: this is currently being used multiple times so most calculations are being repeated needlesly
    def add_neighborhood_obs(self, obs, idx):
        indices = self.neighborhood_indices()
        obs_ext = self.extend_obs_space(obs, closest_drones=indices, index=idx)
        return obs_ext

    def __compute_neighborhood_obs(self):
        closest_drones = self.neighborhood_indices()
        obs_neighbors = []
        for i in range(len(self.envs)):
            obs_neighbor_rel = self.get_obs_neighbor_rel(env_id=i, closest_drones=closest_drones)
            obs_neighbors.append(obs_neighbor_rel.reshape(-1))
        obs_neighbors = np.stack(obs_neighbors)

        # clip observation space of neighborhoods

        obs_neighbors = np.clip(
            obs_neighbors, a_min=self.clip_neighbor_space_min_box, a_max=self.clip_neighbor_space_max_box,
        )
        self.neighbour_observations = obs_neighbors
