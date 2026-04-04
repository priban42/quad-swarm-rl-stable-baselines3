import numpy as np
from gym_art.quadrotor_multi.quad_utils import simulate_camera_measurement_vect

# NOTE: the state_* methods are static because otherwise getattr memorizes self


def state_aw_awdot_dist_distdot_angle_angledot(self):
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
    rel_dist = np.linalg.norm(rel_pos)
    dot_rel_dist = (np.linalg.norm(rel_pos + vel[:2]*self.dt) - rel_dist)/self.dt
    angle_world = self.pre_controller.angle
    rel_pos_norm = rel_pos/np.linalg.norm(rel_pos)
    target_angle_world = np.arctan2(rel_pos_norm[1], rel_pos_norm[0])
    rel_angle = target_angle_world - angle_world
    rel_angle = (rel_angle + np.pi)%(2*np.pi) - np.pi
    ang_vel = self.pre_controller.angular_velocity
    return np.array([angle_world, ang_vel, rel_dist, dot_rel_dist, rel_angle, -np.sign(ang_vel*rel_angle)*abs(ang_vel)])

def state_cdist_cdistdot_dist_distdot_angle_angledot(self):
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
    rel_dist = np.linalg.norm(rel_pos)
    dot_rel_dist = (np.linalg.norm(rel_pos + vel[:2]*self.dt) - rel_dist)/self.dt
    angle_world = self.pre_controller.angle
    rel_pos_norm = rel_pos/np.linalg.norm(rel_pos)
    target_angle_world = np.arctan2(rel_pos_norm[1], rel_pos_norm[0])
    rel_angle = target_angle_world - angle_world
    rel_angle = (rel_angle + np.pi)%(2*np.pi) - np.pi
    ang_vel = self.pre_controller.angular_velocity

    cdist = np.linalg.norm(pos[:2])
    cdistdot = (np.linalg.norm(pos[:2] + vel[:2]*self.dt) - cdist)/self.dt

    return np.array([cdist, cdistdot, rel_dist, dot_rel_dist, rel_angle, -np.sign(ang_vel*rel_angle)*abs(ang_vel)])

def state_cdist_dist_angle(self):
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
    rel_dist = np.linalg.norm(rel_pos)
    # dot_rel_dist = (np.linalg.norm(rel_pos + vel[:2]*self.dt) - rel_dist)/self.dt
    angle_world = self.pre_controller.angle
    rel_pos_norm = rel_pos/np.linalg.norm(rel_pos)
    target_angle_world = np.arctan2(rel_pos_norm[1], rel_pos_norm[0])
    rel_angle = target_angle_world - angle_world
    rel_angle = (rel_angle + np.pi)%(2*np.pi) - np.pi
    # ang_vel = self.pre_controller.angular_velocity

    cdist = np.linalg.norm(pos[:2])
    # cdistdot = (np.linalg.norm(pos[:2] + vel[:2]*self.dt) - cdist)/self.dt

    return np.array([cdist, rel_dist, rel_angle])

def state_cdist_cdistdot_dist_distdot_sangle_angledot(self):
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
    rel_dist = np.linalg.norm(rel_pos)
    dot_rel_dist = (np.linalg.norm(rel_pos + vel[:2]*self.dt) - rel_dist)/self.dt
    angle_world = self.pre_controller.angle
    rel_pos_norm = rel_pos/np.linalg.norm(rel_pos)
    target_angle_world = np.arctan2(rel_pos_norm[1], rel_pos_norm[0])
    rel_angle = target_angle_world - angle_world
    rel_angle = (rel_angle + np.pi)%(2*np.pi) - np.pi
    ang_vel = self.pre_controller.angular_velocity

    cdist = np.linalg.norm(pos[:2])
    cdistdot = (np.linalg.norm(pos[:2] + vel[:2]*self.dt) - cdist)/self.dt

    return np.array([cdist, cdistdot, rel_dist, dot_rel_dist, np.cos(rel_angle), np.sin(rel_angle), -np.sign(ang_vel*rel_angle)*abs(ang_vel)])


def state_cdist_cdistdot_ndist_distdot_nsangle_angledot(self):
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
    rel_dist = np.linalg.norm(rel_pos)
    dot_rel_dist = (np.linalg.norm(rel_pos + vel[:2]*self.dt) - rel_dist)/self.dt
    angle_world = self.pre_controller.angle
    rel_pos_norm = rel_pos/np.linalg.norm(rel_pos)
    target_angle_world = np.arctan2(rel_pos_norm[1], rel_pos_norm[0])
    rel_angle = target_angle_world - angle_world
    rel_angle = (rel_angle + np.pi)%(2*np.pi) - np.pi
    ang_vel = self.pre_controller.angular_velocity

    cdist = np.linalg.norm(pos[:2])
    cdistdot = (np.linalg.norm(pos[:2] + vel[:2]*self.dt) - cdist)/self.dt
    ndist, nangle = simulate_camera_measurement_vect(rel_pos[:, np.newaxis], self.cfg.neighbour_size_cam, self.cfg.focal_length_cam, self.cfg.pixel_noise_cam, np.ones(1)*angle_world, cameras_num=np.ones(1)*self.cfg.n_cameras)
    ndist = np.clip(ndist, 0, 10)[0]
    nangle = nangle[0]
    return np.array([cdist, cdistdot, ndist, dot_rel_dist, np.cos(nangle), np.sin(nangle), -np.sign(ang_vel*rel_angle)*abs(ang_vel)])

def state_cdist_cdistdot_ndist_distdot_nangle_angledot(self):
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
    rel_dist = np.linalg.norm(rel_pos)
    dot_rel_dist = (np.linalg.norm(rel_pos + vel[:2]*self.dt) - rel_dist)/self.dt
    angle_world = self.pre_controller.angle
    rel_pos_norm = rel_pos/np.linalg.norm(rel_pos)
    target_angle_world = np.arctan2(rel_pos_norm[1], rel_pos_norm[0])
    rel_angle = target_angle_world - angle_world
    rel_angle = (rel_angle + np.pi)%(2*np.pi) - np.pi
    ang_vel = self.pre_controller.angular_velocity

    cdist = np.linalg.norm(pos[:2])
    cdistdot = (np.linalg.norm(pos[:2] + vel[:2]*self.dt) - cdist)/self.dt
    ndist, nangle = simulate_camera_measurement_vect(rel_pos[:, np.newaxis], self.cfg.neighbour_size_cam, self.cfg.focal_length_cam, self.cfg.pixel_noise_cam, np.ones(1)*angle_world, cameras_num=np.ones(1)*self.cfg.n_cameras)
    ndist = np.clip(ndist, 0, 10)[0]
    nangle = nangle[0]
    return np.array([cdist, cdistdot, ndist, dot_rel_dist, nangle, -np.sign(ang_vel*rel_angle)*abs(ang_vel)])

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
