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

        duration_time = 5.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)
        self.trajectory, header = parse_csv_to_numpy(Path(__file__).resolve().parent/f"trajectories/trajectory_{45}.csv")
        self.trajectory = interpolate_nd(self.trajectory, 4)

    def update_formation_size(self, new_formation_size):
        pass

    def step(self):
        tick = self.envs[0].tick
        # if tick % self.control_step_for_sec == 0 and tick > 0:
        box_size = self.envs[0].box
        # x, y = np.random.uniform(low=-box_size, high=box_size, size=(2,))
        # z = np.random.uniform(low=-0.5 * box_size, high=0.5 * box_size) + 2.0
        x, y = self.trajectory[tick%self.trajectory.shape[0]][1:3]*0.1
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
