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

class Scenario_dynamic_same_goal_trajectory(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, rng=None):
        super().__init__(quads_mode, envs, num_agents, room_dims, rng)
        self.rng = rng
        duration_time = 5.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)
        self.trajectory, header = parse_csv_to_numpy(Path(__file__).resolve().parent/f"trajectories/trajectory_{45}.csv")
        self.v = 0.5
        self.dt = 1/200
        self.trajectory = self.trajectory[:, 1:3]*0.4
        self.trajectory = np.array(self.rescale_trajectory_constant_speed(self.trajectory[:, 0], self.trajectory[:, 1], self.v, self.dt)).T
        self.spawn_points = np.zeros((self.num_agents, 3))
        self.tick_offset = 0

    def update_formation_size(self, new_formation_size):
        pass

    def step(self):
        tick = self.envs[0].tick + self.tick_offset
        # if tick % self.control_step_for_sec == 0 and tick > 0:
        box_size = self.envs[0].box
        # x, y = np.random.uniform(low=-box_size, high=box_size, size=(2,))
        # z = np.random.uniform(low=-0.5 * box_size, high=0.5 * box_size) + 2.0
        x, y = self.trajectory[tick%self.trajectory.shape[0]]
        # z = self.trajectory[tick%self.trajectory.shape[0]][3]*0.1
        z = 2
        z = max(0.25, z)
        self.formation_center = np.array([x, y, z])
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                         layer_dist=0.0)
        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

        return

    def reset(self):
        # Update duration time
        # duration_time = np.random.uniform(low=4.0, high=6.0)
        duration_time = self.rng.uniform(low=4.0, high=6.0)

        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

        # Reset formation, and parameters related to the formation; formation center; goals
        self.standard_reset()
        spawn_points = self.rng.random((self.num_agents, 2))-0.5
        spawn_points = (spawn_points.T/np.linalg.norm(spawn_points, axis=1)).T
        spawn_points *= self.rng.random(1)*0.5
        spawn_points = np.hstack([spawn_points, np.ones((self.num_agents, 1))*2])
        self.spawn_points = spawn_points
        self.tick_offset = int((self.rng.random(1)*10000)[0])
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


