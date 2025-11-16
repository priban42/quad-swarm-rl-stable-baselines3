import numpy as np
import torch

class VelocityController:
    def __init__(self):
        pass

    def velocity_control(self, obs):
        N = 4
        obs = torch.zeros((N, 27))
        actions = np.zeros((N, 4))