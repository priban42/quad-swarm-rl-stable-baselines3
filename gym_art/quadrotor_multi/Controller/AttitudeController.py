import numpy as np
from .Pid import PIDController
from .references import *
from .MultirotorModel import MultirotorModel, ModelParams, State
from dataclasses import dataclass

class AttitudeController:

    @dataclass
    class Params:
        kp: float = 11.2081
        kd: float = 0.0490
        ki: float = 0.0073
        # kp: float = 6.0
        # kd: float = 0.05
        # ki: float = 0.01
        max_rate_roll_pitch: float = 10.0  # rad/s
        max_rate_yaw: float = 1.0         # rad/s

    # -------------------------------------------------------------------

    def __init__(self, model_params: ModelParams | None = None):
        self.params = AttitudeController.Params()

        # default if not supplied
        self.model_params = model_params if model_params is not None else ModelParams()

        # PID controllers
        self.pid_x = PIDController()
        self.pid_y = PIDController()
        self.pid_z = PIDController()

        self.initialize_pids()
        self.out = AttitudeRate()

    # -------------------------------------------------------------------

    def set_params(self, params: Params):
        self.params = params
        self.initialize_pids()

    # -------------------------------------------------------------------

    def initialize_pids(self):
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()

        self.pid_x.set_params(self.params.kp, self.params.kd, self.params.ki,
                              self.params.max_rate_roll_pitch, 0.1)
        self.pid_y.set_params(self.params.kp, self.params.kd, self.params.ki,
                              self.params.max_rate_roll_pitch, 0.1)
        self.pid_z.set_params(self.params.kp, self.params.kd, self.params.ki,
                              self.params.max_rate_yaw, 0.1)

    # -------------------------------------------------------------------
    #   Control signal from Attitude reference
    # -------------------------------------------------------------------

    def get_control_signal(self, state: State,
                           reference: Attitude, dt: float) -> AttitudeRate:
        """
        Equivalent of:
          reference::AttitudeRate getControlSignal(
              const State&, const reference::Attitude&, double)
        """

        R_error = 0.5*(reference.orientation.T @ state.R - state.R.T@reference.orientation)

        # vectorized version
        R_error_vec = np.array([
            (R_error[1, 2] - R_error[2, 1]) / 2.0,
            (R_error[2, 0] - R_error[0, 2]) / 2.0,
            (R_error[0, 1] - R_error[1, 0]) / 2.0
        ])

        self.out.rate_x = self.pid_x.update(R_error_vec[0], dt)
        self.out.rate_y = self.pid_y.update(R_error_vec[1], dt)
        self.out.rate_z = self.pid_z.update(R_error_vec[2], dt)
        self.out.throttle = reference.throttle

        return self.out

    # -------------------------------------------------------------------
    #   Control signal from Tilt + Heading Rate reference
    # -------------------------------------------------------------------

    def get_control_signal_tilt_hdg_rate(self, state: State,
                                         reference: TiltHdgRate, dt: float) -> AttitudeRate:
        """
        Equivalent of:
          reference::AttitudeRate getControlSignal(
              const State&, const reference::TiltHdgRate&, double)
        """

        R = state.R

        # construct desired orientation Rd
        Rd = np.zeros((3, 3))
        z = reference.tilt_vector / np.linalg.norm(reference.tilt_vector)

        Rd[:, 2] = z
        Rd[:, 1] = np.cross(z, R[:, 0])
        Rd[:, 1] /= np.linalg.norm(Rd[:, 1])
        Rd[:, 0] = np.cross(Rd[:, 1], Rd[:, 2])
        Rd[:, 0] /= np.linalg.norm(Rd[:, 0])

        # orientation error
        R_error = 0.5 * (Rd.T @ R - R.T @ Rd)

        R_error_vec = np.array([
            (R_error[1, 2] - R_error[2, 1]) / 2.0,
            (R_error[2, 0] - R_error[0, 2]) / 2.0,
            (R_error[0, 1] - R_error[1, 0]) / 2.0
        ])

        # PID base rates
        rate_x = self.pid_x.update(R_error_vec[0], dt)
        rate_y = self.pid_y.update(R_error_vec[1], dt)
        rate_z = self.pid_z.update(R_error_vec[2], dt)

        # heading-rate correction
        parasitic = self.intrinsic_body_rate_to_heading_rate(R, np.array([rate_x, rate_y, rate_z]))
        rate_z += self.get_yaw_rate_intrinsic(R, reference.heading_rate - parasitic)

        out = AttitudeRate()
        out.rate_x = rate_x
        out.rate_y = rate_y
        out.rate_z = rate_z
        out.throttle = reference.throttle

        return out

    # -------------------------------------------------------------------
    #   Helpers
    # -------------------------------------------------------------------

    @staticmethod
    def signum(val):
        return 1 if val > 0 else (-1 if val < 0 else 0)

    # -------------------------------------------------------------------

    def intrinsic_body_rate_to_heading_rate(self, R: np.ndarray, w: np.ndarray) -> float:
        """
        Mimics C++ version exactly.
        """

        # Angular velocity tensor
        W = np.array([
            [0,    -w[2],  w[1]],
            [w[2],  0,    -w[0]],
            [-w[1], w[0],  0   ]
        ])

        # R dot = R * W
        R_d = R @ W

        rx = R[0, 0]
        ry = R[1, 0]
        denom = rx * rx + ry * ry

        if abs(denom) <= 1e-5:
            atan2_dx = 0.0
            atan2_dy = 0.0
        else:
            atan2_dx = -ry / denom
            atan2_dy = rx / denom

        # derivative of atan2
        heading_rate = atan2_dx * R_d[0, 0] + atan2_dy * R_d[1, 0]

        return heading_rate

    # -------------------------------------------------------------------

    def get_yaw_rate_intrinsic(self, R: np.ndarray, heading_rate: float) -> float:

        if abs(heading_rate) < 1e-3:
            return 0.0

        # construct heading orbital velocity vector
        heading_vector = np.array([R[0, 0], R[1, 0], 0.0])
        orbital_velocity = np.cross(np.array([0.0, 0.0, heading_rate]), heading_vector)

        b_orb = np.cross(np.array([0.0, 0.0, 1.0]), heading_vector)
        b_orb_norm = np.linalg.norm(b_orb)
        if b_orb_norm < 1e-6:
            return 0.0
        b_orb /= b_orb_norm

        P = np.outer(b_orb, b_orb)
        projected = P @ R[:, 1]

        proj_norm = np.linalg.norm(projected)
        if proj_norm < 1e-5:
            return 0.0

        direction = AttitudeController.signum(orbital_velocity @ projected)
        output_yaw_rate = direction * (np.linalg.norm(orbital_velocity) / proj_norm)

        if not np.isfinite(output_yaw_rate):
            return 0.0

        return output_yaw_rate
