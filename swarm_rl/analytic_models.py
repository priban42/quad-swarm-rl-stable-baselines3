import numpy as np
obs_dim = {"dist":1,
           "angle":1,
           "cdist":1}
def get_obs_parser(obs_type, repeats=1, skip=0):
    obs_types = str.split(obs_type, "_")
    indexes = np.zeros(len(obs_types)+1, dtype=np.int32)
    indexes[0] = skip
    for i in range(len(obs_types)):
        indexes[i+1] = indexes[i] + obs_dim[obs_types[i]]
    slice_step = (indexes[-1]-skip)
    def parser(obs):
        ret = {}
        for i in range(len(obs_types)):
            ret[obs_types[i]] = obs[:, indexes[i]:indexes[i+1]:slice_step]
        return ret
    return parser, indexes[-1]

class Janosov:

    def __init__(self, cfg):

        self.cfg = cfg
        self.dt = 1/25
        self.Cf = 0
        self.Cinter = 0.55
        self.v_max = 0.4
        self.r_inter = 3

    def predict(self, obs, deterministic=True):
        angle = obs[:, 2]
        dist = obs[:, 1]
        neighbor_dist = obs[:, 3::2]
        neighbor_angle = obs[:, 4::2]
        r_rel_norm = np.array([np.cos(angle), np.sin(angle)])
        r_rel_neigh_norm = np.array([np.cos(neighbor_angle), np.sin(neighbor_angle)])
        v_ch = self.v_max*r_rel_norm
        v_inter_ij = r_rel_neigh_norm*(neighbor_dist-self.r_inter)
        v_inter = np.sum(v_inter_ij, axis=2)
        v_inter = self.Cinter*self.v_max*v_inter/np.linalg.norm(v_inter, axis=0)
        v_final = v_inter + v_ch
        ang_vel = np.arctan2(v_final[1, :], v_final[0, :])*10
        ang_vel = np.clip(ang_vel, -np.pi, np.pi)/np.pi
        action = ang_vel[:, np.newaxis]
        return action, None

class Angelani:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dt = 1/25
        self.v_max = 0.4
        self.beta = -1  # (tuneable)
        self.Rn = 0
        self.re = 10.0  # (tuneable) radius of repulsive sphere
        self.gamma = 1  # (tuneable)
        self.sigma = 1  # (tuneable)
        self.r_f = 0.5  # (tuneable)
        self.p = 10

    def f(self, r):
        # r... shape: (2, num_agents, num_agents-1)
        # ret ... shape: (2, num_agents)
        r_mag = np.linalg.norm(r, axis=0)
        r_hat = r / (r_mag + 1e-10)  # small epsilon to avoid division by zero
        scalar = 1.0 / (1.0 + np.exp((r_mag - self.r_f) / self.sigma))
        f_pairs = r_hat * scalar
        return f_pairs.sum(axis=-1)

    def set(self, x):
        self.beta = x[0]
        self.re = x[1]
        self.gamma = x[2]
        self.sigma = x[3]
        self.r_f = x[4]
        self.p = x[5]

    def get(self):
        x = np.array([self.beta, self.re, self.gamma, self.sigma, self.r_f, self.p])
        return x

    def __repr__(self):
        return(f"{self.beta=}, {self.re=}, {self.gamma=}, {self.sigma=}, {self.r_f=}, {self.p=}")

    def predict(self, obs, deterministic=True):
        angle = obs[:, 2]
        dist = obs[:, 1]
        neighbor_dist = obs[:, 3::2]
        neighbor_angle = obs[:, 4::2]
        r_rel_norm = np.array([np.cos(angle), np.sin(angle)])
        r_rel_neigh_norm = np.array([np.cos(neighbor_angle), np.sin(neighbor_angle)])
        r_rel_neigh = r_rel_neigh_norm*neighbor_dist
        f_rep = self.f(r_rel_neigh)
        f_attr = r_rel_norm

        v_final = f_rep*self.beta + f_attr*self.gamma
        ang_vel = np.arctan2(v_final[1, :], v_final[0, :])*self.p
        ang_vel = np.clip(ang_vel, -np.pi, np.pi)/np.pi
        action = ang_vel[:, np.newaxis]
        return action, None
