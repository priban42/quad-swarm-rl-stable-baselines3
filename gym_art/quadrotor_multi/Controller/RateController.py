import numpy as np
from dataclasses import dataclass

from Pid import PIDController
from MultirotorModel import ModelParams
from references import ControlGroup, AttitudeRate


class RateController:

    @dataclass
    class Params:
        kp: float = 4.0
        kd: float = 0.04
        ki: float = 0.0

    # ----------------------------------------------------------------------

    def __init__(self, model_params: ModelParams = None):
        self.model_params = model_params if model_params is not None else ModelParams()
        self.params = RateController.Params()

        # PIDs for body rates
        self.pid_x = PIDController()
        self.pid_y = PIDController()
        self.pid_z = PIDController()

        self.initialize_pids()

    # ----------------------------------------------------------------------

    def set_params(self, params: Params):
        self.params = params
        self.initialize_pids()

    # ----------------------------------------------------------------------

    def initialize_pids(self):
        """Initialize and scale PIDs using inertia matrix J."""
        J = self.model_params.J  # 3x3 inertia matrix

        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()

        # -1 acceleration limit → no saturation (matches C++)
        # antiwindup = 1.0
        self.pid_x.set_params(self.params.kp * J[0, 0],
                              self.params.kd * J[0, 0],
                              self.params.ki * J[0, 0],
                              -1, 1.0)

        self.pid_y.set_params(self.params.kp * J[1, 1],
                              self.params.kd * J[1, 1],
                              self.params.ki * J[1, 1],
                              -1, 1.0)

        self.pid_z.set_params(self.params.kp * J[2, 2],
                              self.params.kd * J[2, 2],
                              self.params.ki * J[2, 2],
                              -1, 1.0)

    # ----------------------------------------------------------------------
    # AttitudeRate → ControlGroup
    # ----------------------------------------------------------------------

    def get_control_signal(self, state, reference: AttitudeRate, dt) -> ControlGroup:
        """
        Matches the C++ implementation:
        - PIDs operate on angular-rate error
        - throttle is passed directly
        """

        ang_ref = np.array([reference.rate_x,
                            reference.rate_y,
                            reference.rate_z])

        ang_error = ang_ref - state.omega

        out = ControlGroup()
        out.roll     = self.pid_x.update(ang_error[0], dt)
        out.pitch    = self.pid_y.update(ang_error[1], dt)
        out.yaw      = self.pid_z.update(ang_error[2], dt)
        out.throttle = reference.throttle

        return out
