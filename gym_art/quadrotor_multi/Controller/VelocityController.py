# velocity_controller.py

import numpy as np
from dataclasses import dataclass

from Pid import PIDController
from MultirotorModel import ModelParams

from references import *

# ------------------------------------------------------------
# VelocityController
# ------------------------------------------------------------

class VelocityController:

    @dataclass
    class Params:
        kp: float = 1.0
        kd: float = 0.5
        ki: float = 0.1
        max_acceleration: float = 4.0   # m/s^2

    # --------------------------------------------------------

    def __init__(self, model_params: ModelParams = None):
        self.model_params = model_params if model_params is not None else ModelParams()
        self.params = VelocityController.Params()

        # three independent PIDs for x, y, z velocity
        self.pid_x = PIDController()
        self.pid_y = PIDController()
        self.pid_z = PIDController()

        self.initialize_pids()

    # --------------------------------------------------------

    def set_params(self, params: Params):
        self.params = params
        self.initialize_pids()

    # --------------------------------------------------------

    def initialize_pids(self):
        # Reset internal states
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()

        # Apply parameter set
        # antiwindup = 1.0 (matches C++)
        self.pid_x.set_params(self.params.kp, self.params.kd, self.params.ki,
                              self.params.max_acceleration, 1.0)
        self.pid_y.set_params(self.params.kp, self.params.kd, self.params.ki,
                              self.params.max_acceleration, 1.0)
        self.pid_z.set_params(self.params.kp, self.params.kd, self.params.ki,
                              self.params.max_acceleration, 1.0)

    # --------------------------------------------------------
    # getControlSignal for Velocity + Heading (no heading-rate)
    # --------------------------------------------------------

    def get_control_signal(self, state, reference: VelocityHdg, dt):
        """
        state.v is expected to be a 3D numpy vector
        """

        vel_error = reference.velocity - state.v

        acc = np.zeros(3)
        acc[0] = self.pid_x.update(vel_error[0], dt)
        acc[1] = self.pid_y.update(vel_error[1], dt)
        acc[2] = self.pid_z.update(vel_error[2], dt)

        return AccelerationHdg(acceleration=acc,
                               heading=reference.heading)

    # --------------------------------------------------------
    # getControlSignal for Velocity + Heading Rate
    # --------------------------------------------------------

    def get_control_signal_rate(self, state, reference: VelocityHdgRate, dt):
        """
        Matches the second C++ getControlSignal() overload:
        VelocityHdgRate → AccelerationHdgRate
        """

        vel_error = reference.velocity - state.v

        acc = np.zeros(3)
        acc[0] = self.pid_x.update(vel_error[0], dt)
        acc[1] = self.pid_y.update(vel_error[1], dt)
        acc[2] = self.pid_z.update(vel_error[2], dt)

        return AccelerationHdgRate(acceleration=acc,
                                   heading_rate=reference.heading_rate)
