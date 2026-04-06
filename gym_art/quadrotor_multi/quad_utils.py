import numpy as np
import numpy.random as nr
import numba as nb
from numba import njit
from numpy.linalg import norm
from numpy import cos, sin
from scipy import spatial
from copy import deepcopy

EPS = 1e-5

QUAD_COLOR = (
    (1.0, 0.0, 0.0),  # red
    (1.0, 0.5, 0.0),  # orange
    (1.0, 1.0, 0.0),  # yellow
    (0.0, 1.0, 1.0),  # cyan
    (1.0, 1.0, 0.5),  # magenta
    (0.0, 0.0, 1.0),  # blue
    (0.22, 0.2, 0.47),  # purple
    (1.0, 0.0, 1.0),  # Violet
    # (0.0, 1.0, 1.0),  # cyan
    # (1.0, 0.0, 1.0),  # magenta
    # (1.0, 0.0, 1.0),  # Violet
)

OBST_COLOR_3 = (0., 0.5, 0.)
OBST_COLOR_4 = (0., 0.5, 0., 1.)


QUADS_OBS_REPR = {
    'xyz_vxyz_R_omega': 18,
    'aw_awdot_dist_distdot_angle_angledot': 6,
    'cdist_cdistdot_dist_distdot_angle_angledot': 6,
    'cdist_dist_angle': 3,
    'cdist_angle': 2,
    'cdist_cdistdot_ndist_distdot_nangle_angledot': 6,
    'cdist_cdistdot_dist_distdot_sangle_angledot': 7,
    'cdist_cdistdot_ndist_distdot_nsangle_angledot': 7,
    'xyz_vxyz_R_omega_floor': 19,
    'xyz_vxyz_R_omega_wall': 24,
}

QUADS_NEIGHBOR_OBS_TYPE = {
    'none': 0,
    'pos_vel': 6,
    'pos_vel_R':15,
    'pos_Rz':6,
    'pos_vel_Rz':9,
    'rng3':3,
    'pos':3,
    'npos':3,
    'dist':1,
    'angle':1,
    'dist_angle':2,
    'ndist_nangle':2,
    'dist_sangle':3,
    'ndist_nsangle':3,
    'dist_angle_heading':3,
    'dist_sangle_sheading':5,
    'dist_angle_vel2d':4,

}

QUADS_OBSTACLE_OBS_TYPE = {
    'none': 0,
    'octomap': 9,
}


# dict pretty printing
def print_dic(dic, indent=""):
    for key, item in dic.items():
        if isinstance(item, dict):
            print(indent, key + ":")
            print_dic(item, indent=indent + "  ")
        else:
            print(indent, key + ":", item)


# walk dictionary
def walk_dict(node, call):
    for key, item in node.items():
        if isinstance(item, dict):
            walk_dict(item, call)
        else:
            node[key] = call(key, item)


def walk_2dict(node1, node2, call):
    for key, item in node1.items():
        if isinstance(item, dict):
            walk_2dict(item, node2[key], call)
        else:
            node1[key], node2[key] = call(key, item, node2[key])


# numpy's cross is really slow for some reason
def cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


# returns (normalized vector, original norm)
def normalize(x):
    # n = norm(x)
    n = (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** 0.5  # np.sqrt(np.cumsum(np.square(x)))[2]

    if n < 0.00001:
        return x, 0
    return x / n, n


def norm2(x):
    return np.sum(x ** 2)


# uniformly sample from the set of all 3D rotation matrices
def rand_uniform_rot3d(rng=np.random.default_rng()):
    # randunit = lambda: normalize(np.random.normal(size=(3,)))[0]
    randunit = lambda: normalize(rng.random.normal(size=(3,)))[0]
    up = randunit()
    fwd = randunit()
    while np.dot(fwd, up) > 0.95:
        fwd = randunit()
    left, _ = normalize(cross(up, fwd))
    # import pdb; pdb.set_trace()
    up = cross(fwd, left)
    rot = np.column_stack([fwd, left, up])
    return rot


# shorter way to construct a numpy array
def npa(*args):
    return np.array(args)


def clamp_norm(x, maxnorm):
    # n = np.linalg.norm(x)
    # n = np.sqrt(np.cumsum(np.square(x)))[2]
    n = (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** 0.5
    return x if n <= maxnorm else (maxnorm / n) * x


# project a vector into the x-y plane and normalize it.
def to_xyhat(vec):
    v = np.array([vec[0], vec[1], 0])
    # v = deepcopy(vec)
    v[2] = 0
    v, _ = normalize(v)
    return v


def log_error(err_str, ):
    with open("/tmp/sac/errors.txt", "a") as myfile:
        myfile.write(err_str)
        # myfile.write('###############################################')


def quat2R(qw, qx, qy, qz):
    R = \
        [[1.0 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
         [2 * qx * qy + 2 * qz * qw, 1.0 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
         [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1.0 - 2 * qx ** 2 - 2 * qy ** 2]]
    return np.array(R)


quat2R_numba = njit()(quat2R)


def qwxyz2R(quat):
    return quat2R(qw=quat[0], qx=quat[1], qy=quat[2], qz=quat[3])


def quatXquat(quat, quat_theta):
    ## quat * quat_theta
    noisy_quat = np.zeros(4)
    noisy_quat[0] = quat[0] * quat_theta[0] - quat[1] * quat_theta[1] - quat[2] * quat_theta[2] - quat[3] * quat_theta[
        3]
    noisy_quat[1] = quat[0] * quat_theta[1] + quat[1] * quat_theta[0] - quat[2] * quat_theta[3] + quat[3] * quat_theta[
        2]
    noisy_quat[2] = quat[0] * quat_theta[2] + quat[1] * quat_theta[3] + quat[2] * quat_theta[0] - quat[3] * quat_theta[
        1]
    noisy_quat[3] = quat[0] * quat_theta[3] - quat[1] * quat_theta[2] + quat[2] * quat_theta[1] + quat[3] * quat_theta[
        0]
    return noisy_quat


quatXquat_numba = njit()(quatXquat)


def R2quat(rot):
    # print('R2quat: ', rot, type(rot))
    R = rot.reshape([3, 3])
    w = np.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
    w4 = (4.0 * w)
    x = (R[2, 1] - R[1, 2]) / w4
    y = (R[0, 2] - R[2, 0]) / w4
    z = (R[1, 0] - R[0, 1]) / w4
    return np.array([w, x, y, z])


def rot2D(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])


def rotZ(theta):
    r = np.eye(4)
    r[:2, :2] = rot2D(theta)
    return r


def rpy2R(r, p, y):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(r), -np.sin(r)],
                    [0, np.sin(r), np.cos(r)]
                    ])
    R_y = np.array([[np.cos(p), 0, np.sin(p)],
                    [0, 1, 0],
                    [-np.sin(p), 0, np.cos(p)]
                    ])
    R_z = np.array([[np.cos(y), -np.sin(y), 0],
                    [np.sin(y), np.cos(y), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def randyaw():
    rotz = np.random.uniform(-np.pi, np.pi)
    return rotZ(rotz)[:3, :3]


def exUxe(e, U):
    """
    Cross product approximation
    exUxe = U - (U @ e) * e, where
    Args:
        e[3,1] - norm vector (assumes the same norm vector for all vectors in the batch U)
        U[3,batch_dim] - set of vectors to perform cross product on
    Returns:
        [3,batch_dim] - batch-wise cross product approximation
    """
    return U - (U.T @ rot_z).T * np.repeat(rot_z, U.shape[1], axis=1)


def cross_vec(v1, v2):
    return np.array([[0, -v1[2], v1[1]], [v1[2], 0, -v1[0]], [-v1[1], v1[0], 0]]) @ v2


def cross_mx4(V1, V2):
    x1 = cross(V1[0, :], V2[0, :])
    x2 = cross(V1[1, :], V2[1, :])
    x3 = cross(V1[2, :], V2[2, :])
    x4 = cross(V1[3, :], V2[3, :])
    return np.array([x1, x2, x3, x4])


def cross_vec_mx4(V1, V2):
    x1 = cross(V1, V2[0, :])
    x2 = cross(V1, V2[1, :])
    x3 = cross(V1, V2[2, :])
    x4 = cross(V1, V2[3, :])
    return np.array([x1, x2, x3, x4])


def dict_update_existing(dic, dic_upd):
    for key in dic_upd.keys():
        if isinstance(dic[key], dict):
            dict_update_existing(dic[key], dic_upd[key])
        else:
            dic[key] = dic_upd[key]

def rotation_matrix(axis, angle):
    """
    Create a 3x3 rotation matrix from an axis and an angle using Rodrigues' formula.
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

def rotation_matrix_2d(angle_rad):
    return np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])


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
        cameras_num: int = 1,
        min_distance=0.25):
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

    # l = np.nan_to_num(l, nan=0.0)
    # angle_rel = np.nan_to_num(angle_rel, nan=0.0)

    outlier_mask = abs(np.linalg.norm(rel_pos, axis=0))<min_distance
    outlier_mask = outlier_mask | np.isnan(l)

    l[outlier_mask] = min_distance
    angle_rel[outlier_mask] = angle_orig[outlier_mask]

    return l, angle_rel


class OUNoise:
    """Ornstein–Uhlenbeck process"""

    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.3, use_seed=False):
        """
        @param: mu: mean of noise
        @param: theta: stabilization coeff (i.e. noise return to mean)
        @param: sigma: noise scale coeff
        @param: use_seed: set the random number generator to some specific seed for test
        """
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()
        if use_seed:
            nr.seed(2)

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state


if __name__ == "__main__":
    # Cross product test
    import time

    rot_z = np.array([[3], [4], [5]])
    rot_z = rot_z / np.linalg.norm(rot_z)
    v_rotors = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6]])

    start_time = time.time()
    cr1 = v_rotors - (v_rotors.T @ rot_z).T * np.repeat(rot_z, 4, axis=1)
    print("cr1 time:", time.time() - start_time)

    start_time = time.time()
    cr2 = np.cross(rot_z.T, np.cross(v_rotors.T, np.repeat(rot_z, 4, axis=1).T)).T
    print("cr2 time:", time.time() - start_time)
    print("cr1 == cr2:", np.sum(cr1 - cr2) < 1e-10)
