import numpy as np

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario
from pathlib import Path

def parse_csv_to_numpy(filepath):
    with open(filepath, "r") as f:
        header = f.readline().strip().split(",")

    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    return data, header

def interpolate_nd(coords, scale):
    n, d = coords.shape
    new_n = int(np.round(n * scale))

    old_t = np.linspace(0, 1, n)
    new_t = np.linspace(0, 1, new_n)

    new_coords = np.empty((new_n, d))
    for i in range(d):
        new_coords[:, i] = np.interp(new_t, old_t, coords[:, i])

    return new_coords

class Scenario_dynamic_repulsive(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, rng=None):
        super().__init__(quads_mode, envs, num_agents, room_dims, rng)
        self.rng = rng
        self.v_max = 0.5
        self.dt = 1/200
        self.spawn_points = np.zeros((self.num_agents, 3))
        self.arena_size = 5
        self.center_height = 7.5
        if self.envs[0].cfg.dim_mode == "3D":
            self.dimension = 3
        else:
            self.dimension = 2
        self.pos = np.array([0., 0., self.center_height])


    def update_formation_size(self, new_formation_size):
        pass

    def step(self):
        agent_force = np.zeros(self.dimension)
        for env in self.envs:
            if hasattr(env.dynamics, "pos"):
                chaser_pos = env.dynamics.pos[:self.dimension]
                rel_vec = -(chaser_pos - self.pos[:self.dimension])
                d = np.linalg.norm(rel_vec)
                agent_force += rel_vec/(d*d)
        d_e = np.linalg.norm(self.pos[:self.dimension])
        arena_force = -(self.pos-np.array([0, 0, self.center_height]))[:self.dimension]/(d_e*(max(self.arena_size-d_e, 0.1)))

        v_vect = (agent_force + arena_force)
        v_scale = np.linalg.norm(v_vect)
        v = (v_vect/v_scale)*min(v_scale, self.v_max)
        self.pos[:self.dimension] = self.pos[:self.dimension] + v*self.dt
        # z = 2
        # z = max(0.25, z)
        self.formation_center = np.array([self.pos[0], self.pos[1], self.pos[2]])
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                         layer_dist=0.0)
        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

        return

    def reset(self):

        self.standard_reset()
        spawn_points = np.zeros((self.num_agents, 3), dtype=np.float64)
        spawn_points[:self.num_agents, :self.dimension] = self.rng.random((self.num_agents, self.dimension))-0.5
        spawn_points = (spawn_points.T/np.linalg.norm(spawn_points, axis=1)).T
        spawn_points *= self.rng.random(1)*0.5
        spawn_points[:, 2] += self.center_height
        # spawn_points = np.hstack([spawn_points, np.ones((self.num_agents, 1))*2])
        self.spawn_points = spawn_points
        self.pos[2] = 0
        self.pos[:self.dimension] = self.rng.random(self.dimension) - 0.5
        self.pos[:self.dimension] = (self.pos[:self.dimension]/np.linalg.norm(self.pos[:self.dimension]))*(self.rng.random(1)*3 + 2)
        self.pos[2] += self.center_height
        self.step()

    @staticmethod
    def rescale_trajectory_constant_speed(x, y, v, dt):
        """
        Resample a 2D trajectory so that speed is constant.

        Parameters
        ----------
        x, y : 1D numpy arrays
            Original trajectory coordinates (meters).
        v : float
            Desired speed (meters / second).
        dt : float
            Timestep (seconds).

        Returns
        -------
        x_new, y_new : 1D numpy arrays
            Resampled trajectory with constant speed v.
        """

        x = np.asarray(x)
        y = np.asarray(y)

        # Arc length
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt(dx ** 2 + dy ** 2)

        s = np.concatenate(([0.0], np.cumsum(ds)))
        total_length = s[-1]
        ds_target = v * dt  # Desired arc-length step
        s_uniform = np.arange(0, total_length, ds_target)  # Uniform arc-length sampling

        # Linear interpolation
        x_new = np.interp(s_uniform, s, x)
        y_new = np.interp(s_uniform, s, y)

        return x_new, y_new


