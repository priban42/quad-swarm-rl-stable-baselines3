import numpy as np
from dataclasses import dataclass

from .Pid import PIDController
from .MultirotorModel import ModelParams
from .references import Position, VelocityHdg


class PositionController:

    @dataclass
    class Params:
        kp:float = 4.1625
        kd:float = 0.5473
        ki:float = 0.0023
        # kp: float = 6.0
        # kd: float = 0.3
        # ki: float = 3.0
        max_velocity: float = 6.0   # m/s

    # --------------------------------------------------------

    def __init__(self, model_params: ModelParams = None):
        self.model_params = model_params if model_params is not None else ModelParams()
        self.params = PositionController.Params()

        # 3 independent PIDs along x, y, z
        self.pid_x = PIDController()
        self.pid_y = PIDController()
        self.pid_z = PIDController()

        self.initialize_pids()

        self.out = VelocityHdg()

    # --------------------------------------------------------

    def set_params(self, params: Params):
        self.params = params
        self.initialize_pids()

    # --------------------------------------------------------

    def initialize_pids(self):
        # Reset states
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()

        # Apply parameters (antiwindup=1.0, identical to C++)
        self.pid_x.set_params(self.params.kp, self.params.kd, self.params.ki,
                              self.params.max_velocity, 1.0)
        self.pid_y.set_params(self.params.kp, self.params.kd, self.params.ki,
                              self.params.max_velocity, 1.0)
        self.pid_z.set_params(self.params.kp, self.params.kd, self.params.ki,
                              self.params.max_velocity, 2.0)

    # --------------------------------------------------------
    # get_control_signal: Position → VelocityHdg
    # --------------------------------------------------------

    def get_control_signal(self, state, reference: Position, dt):
        """
        state.x is expected to be a 3D numpy vector
        """

        pos_error = reference.position - state.x

        vel = np.zeros(3)
        vel[0] = self.pid_x.update(pos_error[0], dt)
        vel[1] = self.pid_y.update(pos_error[1], dt)
        vel[2] = self.pid_z.update(pos_error[2], dt)

        self.out.velocity = vel
        self.out.heading = reference.heading

        return self.out
