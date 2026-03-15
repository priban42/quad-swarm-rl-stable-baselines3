import numpy as np


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

def circle_intersection_vect(c1: np.ndarray, r1: float, c2: np.ndarray, r2) -> np.ndarray:
    """
    Args:
        c1: (2, N) centers of circle 1
        r1: radius of circle 1 (scalar)
        c2: (2, N) centers of circle 2
        r2: (N,) or scalar radius of circle 2

    Returns:
        p1, p2: each (2, N)
    """
    d = np.linalg.norm(c2 - c1, axis=0)  # (N,)
    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)  # (N,)
    h = np.sqrt(r1 ** 2 - a ** 2)  # (N,)

    radial = (c2 - c1) / d  # (2, N)
    mid = c1 + a * radial  # (2, N)

    perp = np.array([-radial[1], radial[0]])  # (2, N)

    p1 = mid + h * perp  # (2, N)
    p2 = mid - h * perp  # (2, N)
    return p1, p2

def get_camera_angle(rel_angle, n):
    """
    Args:
        rel_angle: the angle of a detected point relative to the drone orientation (-pi, pi)
        n: number of cameras

    Returns: angle of the best camera
    """
    cam_idx = np.round((rel_angle % (2 * np.pi)) / (2 * np.pi / n)).astype(int) % n
    return cam_idx * 2 * np.pi / n

def simulate_camera_measurement_vect(
        rel_pos: np.ndarray,  # (2, N)
        known_size_m: float,
        focal_length_m: float,
        camera_noise_px: float,
        global_angle: np.ndarray,
        fov_deg: float = 70,
        camera_resolution: float = 640.0,
        cameras_num: int = 1):
    # rel_pose = R(-global_angle) @ rel_pos
    c, s = np.cos(-global_angle), np.sin(-global_angle)  # (2, N)
    rel_pose = np.array([c * rel_pos[0] - s * rel_pos[1],
                         s * rel_pos[0] + c * rel_pos[1]])  # (2, N)
    angle_orig = np.arctan2(rel_pose[1], rel_pose[0])  # (N,)
    camera_angle = get_camera_angle(angle_orig, cameras_num)  # (N,)

    # rotate each point by its own camera angle
    c, s = np.cos(-camera_angle), np.sin(-camera_angle)  # (N,)
    center_2d = np.array([c * rel_pose[0] - s * rel_pose[1],
                          s * rel_pose[0] + c * rel_pose[1]])  # (2, N)

    r = known_size_m / 2
    f = focal_length_m
    w = 2 * np.tan((fov_deg / 2) * np.pi / 180) * f

    x1, x2 = circle_intersection_vect(center_2d, r,
                                      center_2d / 2,
                                      np.linalg.norm(center_2d, axis=0) / 2)  # (2, N) each

    u1_orig = x1[1] * f / x1[0]
    u2_orig = x2[1] * f / x2[0]
    u1_px = u1_orig * camera_resolution / w
    u2_px = u2_orig * camera_resolution / w
    u1_px += np.random.normal(0, camera_noise_px, size=u1_px.shape)
    u2_px += np.random.normal(0, camera_noise_px, size=u2_px.shape)
    u1 = u1_px * w / camera_resolution
    u2 = u2_px * w / camera_resolution

    alpha = np.abs(np.arctan(u1 / f) - np.arctan(u2 / f))
    l = r / np.sin(alpha / 2)

    angle_cam = (np.arctan(u1 / f) + np.arctan(u2 / f)) / 2
    angle_rel = angle_cam + camera_angle
    angle_rel = (angle_rel + np.pi) % (2 * np.pi) - np.pi

    l = np.nan_to_num(l, nan=0.0)
    angle_rel = np.nan_to_num(angle_rel, nan=0.0)

    return l, angle_rel

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
