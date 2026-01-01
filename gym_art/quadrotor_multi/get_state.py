import numpy as np


# NOTE: the state_* methods are static because otherwise getattr memorizes self


def state_aw_awdot_xy_vxy_a_adot(self):
    if self.use_numba:
        pos, vel, rot, omega, acc = self.sense_noise.add_noise_numba(
            self.dynamics.pos,
            self.dynamics.vel,
            self.dynamics.rot,
            self.dynamics.omega,
            self.dynamics.accelerometer,
            self.dt
        )
    else:
        pos, vel, rot, omega, acc = self.sense_noise.add_noise(
            pos=self.dynamics.pos,
            vel=self.dynamics.vel,
            rot=self.dynamics.rot,
            omega=self.dynamics.omega,
            acc=self.dynamics.accelerometer,
            dt=self.dt
        )
    rel_pos = self.goal[:2] - pos[:2]
    angle_world = self.pre_controller.angle
    rel_pos_norm = rel_pos/np.linalg.norm(rel_pos)
    goal_angle_world = np.arctan2(rel_pos_norm[1], rel_pos_norm[0])
    rel_angle = angle_world-goal_angle_world
    rel_angle = (rel_angle + np.pi)%(2*np.pi) - np.pi
    ang_vel = self.pre_controller.angular_velocity
    return np.concatenate([[angle_world, ang_vel], rel_pos, vel[:2], [rel_angle, -np.sign(ang_vel*rel_angle)*ang_vel]])

def state_xyz_vxyz_R_omega(self):
    if self.use_numba:
        pos, vel, rot, omega, acc = self.sense_noise.add_noise_numba(
            self.dynamics.pos,
            self.dynamics.vel,
            self.dynamics.rot,
            self.dynamics.omega,
            self.dynamics.accelerometer,
            self.dt
        )
    else:
        pos, vel, rot, omega, acc = self.sense_noise.add_noise(
            pos=self.dynamics.pos,
            vel=self.dynamics.vel,
            rot=self.dynamics.rot,
            omega=self.dynamics.omega,
            acc=self.dynamics.accelerometer,
            dt=self.dt
        )
    return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega])


def state_xyz_vxyz_R_omega_floor(self):
    if self.use_numba:
        pos, vel, rot, omega, acc = self.sense_noise.add_noise_numba(
            self.dynamics.pos,
            self.dynamics.vel,
            self.dynamics.rot,
            self.dynamics.omega,
            self.dynamics.accelerometer,
            self.dt
        )
    else:
        pos, vel, rot, omega, acc = self.sense_noise.add_noise(
            pos=self.dynamics.pos,
            vel=self.dynamics.vel,
            rot=self.dynamics.rot,
            omega=self.dynamics.omega,
            acc=self.dynamics.accelerometer,
            dt=self.dt
        )
    return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, (pos[2],)])


def state_xyz_vxyz_R_omega_wall(self):
    if self.use_numba:
        pos, vel, rot, omega, acc = self.sense_noise.add_noise_numba(
            self.dynamics.pos,
            self.dynamics.vel,
            self.dynamics.rot,
            self.dynamics.omega,
            self.dynamics.accelerometer,
            self.dt
        )
    else:
        pos, vel, rot, omega, acc = self.sense_noise.add_noise(
            pos=self.dynamics.pos,
            vel=self.dynamics.vel,
            rot=self.dynamics.rot,
            omega=self.dynamics.omega,
            acc=self.dynamics.accelerometer,
            dt=self.dt
        )
    # return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, (pos[2],)])
    wall_box_0 = np.clip(pos - self.room_box[0], a_min=0.0, a_max=5.0)
    wall_box_1 = np.clip(self.room_box[1] - pos, a_min=0.0, a_max=5.0)
    return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, wall_box_0, wall_box_1])
