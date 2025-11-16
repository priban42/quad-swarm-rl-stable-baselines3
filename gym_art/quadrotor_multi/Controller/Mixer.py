# mixer.py
import numpy as np
from dataclasses import dataclass
from references import ControlGroup, Actuators
from MultirotorModel import ModelParams   # import the ModelParams from previous translation


class Mixer:

    @dataclass
    class Params:
        desaturation: bool = True

    def __init__(self, model_params: ModelParams = None):
        self.params = Mixer.Params()
        self.model_params = model_params if model_params is not None else ModelParams()
        self.allocation_matrix_inv = None

        self.calculate_allocation()

    # --------------------------------------------------------------

    def set_params(self, params: Params):
        self.params = params
        self.calculate_allocation()

    # --------------------------------------------------------------

    def calculate_allocation(self):
        """
        Python equivalent of C++ Mixer::calculateAllocation()
        """

        # full allocation matrix from model parameters
        alloc = np.array(self.model_params.allocation_matrix, dtype=float)

        # compute pseudo-inverse: Aᵀ (A Aᵀ)⁻¹
        self.allocation_matrix_inv = alloc.T @ np.linalg.inv(alloc @ alloc.T)

        # ------------------------------------------------------
        # Normalize allocation matrix to PX4-style mixing
        # ------------------------------------------------------

        # first two columns (roll, pitch)
        for i in range(self.model_params.n_motors):
            col = self.allocation_matrix_inv[i, 0:2]
            norm = np.linalg.norm(col)
            if norm > 0:
                self.allocation_matrix_inv[i, 0:2] = col / norm

        # third column (yaw)
        for i in range(self.model_params.n_motors):
            v = self.allocation_matrix_inv[i, 2]
            if v > 1e-2:
                self.allocation_matrix_inv[i, 2] = 1.0
            elif v < -1e-2:
                self.allocation_matrix_inv[i, 2] = -1.0
            else:
                self.allocation_matrix_inv[i, 2] = 0.0

        # fourth column (throttle)
        self.allocation_matrix_inv[:, 3] = 1.0

    # --------------------------------------------------------------

    def get_control_signal(self, reference: ControlGroup) -> Actuators:
        """
        Equivalent to C++ Mixer::getControlSignal()
        """

        ctrl_group = np.array([
            reference.roll,
            reference.pitch,
            reference.yaw,
            reference.throttle
        ], dtype=float)

        # Base motor command
        motors = self.allocation_matrix_inv @ ctrl_group

        # ------------------
        # Desaturation logic
        # ------------------
        if self.params.desaturation:

            # shift if negative
            mn = motors.min()
            if mn < 0.0:
                motors = motors + abs(mn)

            # scale if above max
            mx = motors.max()
            if mx > 1.0:

                # If throttle > small value → preserve throttle direction
                if reference.throttle > 1e-2:
                    scale = motors.mean() / reference.throttle
                    # reduce roll/pitch/yaw
                    ctrl_group[0:3] /= scale
                    motors = self.allocation_matrix_inv @ ctrl_group

                else:
                    # throttle very small → simply scale down
                    motors = motors / mx

        return Actuators(motors=motors)

    # --------------------------------------------------------------

    def get_allocation_matrix(self):
        return np.array(self.allocation_matrix_inv, copy=True)
