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
        self.Cinter = 0.5
        self.v_max = 0.4
        self.r_inter = 1

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