import numpy as np
from dataclasses import dataclass

from .references import Attitude, TiltHdgRate, AccelerationHdg, AccelerationHdgRate
from .MultirotorModel import ModelParams


class AccelerationController:
    def __init__(self, model_params: ModelParams = None):
        self.model_params = model_params if model_params is not None else ModelParams()
        self.out = Attitude()

    # ----------------------------------------------------------------------
    # AccelerationHdg → Attitude (full attitude construction)
    # ---------------------------------------------------------------------
    # -

    def get_control_signal(self, state, reference: AccelerationHdg, dt=None) -> Attitude:
        g = self.model_params.g
        mass = self.model_params.mass
        kf = self.model_params.kf
        n_motors = self.model_params.n_motors
        min_rpm = self.model_params.min_rpm
        max_rpm = self.model_params.max_rpm

        # Desired force in world frame
        fd = (reference.acceleration + np.array([0.0, 0.0, g])) * mass
        # fd_norm = fd / np.linalg.norm(fd)
        fd_norm = fd / np.sqrt(fd[0]**2 + fd[1]**2 + fd[2]**2)

        # Desired heading direction projected into XY
        bxd = np.array([np.cos(reference.heading), np.sin(reference.heading), 0.0])

        Rd = np.zeros((3, 3))
        # Body Z axis = fd direction
        Rd[:, 2] = fd_norm
        # Complement projector

        ### equivalent implementation but slower because of np.linalg.pinv(Bt_A) taking 30 % of time of the entire method
        # projector = np.eye(3) - np.outer(fd_norm, fd_norm)
        # # Basis for nullspace (A)
        # A = projector[:, :2]     # 3x2 matrix
        # # Basis for XY plane (B)
        # B = np.eye(3)[:, :2]     # 3x2
        #
        # Bt_A = B.T @ A                   # (2x2)
        # # Bt_A_pinv = np.linalg.inv(Bt_A.T @ Bt_A) @ Bt_A.T
        # Bt_A_pinv = np.linalg.pinv(Bt_A)
        # oblique_projector = A @ Bt_A_pinv @ B.T     # (3×3)
        #
        # # Body X axis
        # x_des = oblique_projector @ bxd
        # x_des /= np.linalg.norm(x_des)
        ####################Optimized version (unreadable)####################
        A2 = np.array([
            [1.0 - fd_norm[0] * fd_norm[0], -fd_norm[0] * fd_norm[1]],
            [-fd_norm[1] * fd_norm[0], 1.0 - fd_norm[1] * fd_norm[1]],
            [-fd_norm[2] * fd_norm[0], -fd_norm[2] * fd_norm[1]],
        ])

        Bt_A2 = A2[:2, :]

        det2 = Bt_A2[0, 0] * Bt_A2[1, 1] - Bt_A2[0, 1] * Bt_A2[1, 0]
        Bt_A2_inv = np.array([
            [Bt_A2[1, 1], -Bt_A2[0, 1]],
            [-Bt_A2[1, 0], Bt_A2[0, 0]],
        ]) / det2

        bxd_xy2 = bxd[:2]
        coeffs2 = Bt_A2_inv @ bxd_xy2
        x_des2 = A2 @ coeffs2
        x_des2 /= np.sqrt(x_des2[0] ** 2 + x_des2[1] ** 2 + x_des2[2] ** 2)

        x_des = x_des2

        #########################33
        Rd[:, 0] = x_des

        # Body Y = Z × X
        # y_des = np.cross(fd_norm, x_des)
        y_des = np.array([
            fd_norm[1] * x_des[2] - fd_norm[2] * x_des[1],
            fd_norm[2] * x_des[0] - fd_norm[0] * x_des[2],
            fd_norm[0] * x_des[1] - fd_norm[1] * x_des[0],
        ])
        y_des /= np.sqrt(y_des[0]**2 + y_des[1]**2 + y_des[2]**2)

        Rd[:, 1] = y_des

        # -----------------------------------------
        # Compute throttle
        # -----------------------------------------
        thrust_force = np.dot(fd, state.R[:, 2])
        thrust_force = max(thrust_force, 0)
        throttle = (np.sqrt(thrust_force / (kf * n_motors)) - min_rpm) / (max_rpm - min_rpm)
        # throttle = float(np.clip(throttle, 0.0, 1.0))
        throttle = 0.0 if throttle < 0.0 else (1.0 if throttle > 1.0 else float(throttle))
        # -----------------------------------------
        # Output
        # -----------------------------------------
        # out = Attitude()
        self.out.orientation = Rd
        self.out.throttle = throttle

        if np.isnan(throttle):
            pass

        return self.out

    # ----------------------------------------------------------------------
    # AccelerationHdgRate → TiltHdgRate (no full attitude construction)
    # ----------------------------------------------------------------------
    def get_control_signal_rate(self, state, reference: AccelerationHdgRate, dt=None) -> TiltHdgRate:
        g = self.model_params.g
        mass = self.model_params.mass
        kf = self.model_params.kf
        n_motors = self.model_params.n_motors
        min_rpm = self.model_params.min_rpm
        max_rpm = self.model_params.max_rpm

        # Desired force
        fd = (reference.acceleration + np.array([0.0, 0.0, g])) * mass
        fd_norm = fd / np.linalg.norm(fd)

        # Compute throttle
        thrust_force = np.dot(fd, state.R[:, 2])
        throttle = (np.sqrt(thrust_force / (kf * n_motors)) - min_rpm) / (max_rpm - min_rpm)
        throttle = float(np.clip(throttle, 0.0, 1.0))

        out = TiltHdgRate()
        out.tilt_vector = fd_norm
        out.heading_rate = reference.heading_rate
        out.throttle = throttle

        return out
