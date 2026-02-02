# pid.py

import math
from numba import njit

@njit
def _pid_update_numba(error, last_error, integral, dt, kp, kd, ki, saturation, antiwindup):
    """Numba-compiled PID update logic"""
    difference = (error - last_error) / dt
    last_error = error

    output = kp * error + kd * difference + ki * integral

    # Saturation (if enabled)
    if saturation > 0:
        if output >= saturation:
            output = saturation
        elif output <= -saturation:
            output = -saturation

    # Anti-windup (if enabled)
    if antiwindup > 0:
        if -antiwindup < output < antiwindup:
            integral += error * dt

    return output, last_error, integral

class PIDController:
    def __init__(self):
        # gains
        self.kp = 0.0
        self.kd = 0.0
        self.ki = 0.0

        # internal state
        self.last_error = 0.0
        self.integral = 0.0

        # saturation parameters
        self.saturation = -1.0   # disabled when negative
        self.antiwindup = -1.0   # disabled when negative

        self.reset()

    # ----------------------------------------------------------

    def set_params(self, kp, kd, ki, saturation, antiwindup):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.saturation = saturation
        self.antiwindup = antiwindup

    # ----------------------------------------------------------

    def set_saturation(self, saturation=-1.0):
        self.saturation = saturation

    # ----------------------------------------------------------

    def reset(self):
        self.last_error = 0.0
        self.integral = 0.0

    def update(self, error, dt):
        """PID update using numba-compiled backend"""
        output, self.last_error, self.integral = _pid_update_numba(
            error, self.last_error, self.integral, dt,
            self.kp, self.kd, self.ki, self.saturation, self.antiwindup
        )
        return output

    # ----------------------------------------------------------

    # def update(self, error, dt):
    #     """
    #     Equivalent to C++ PIDController::update()
    #     """
    #
    #     # derivative term
    #     difference = (error - self.last_error) / dt
    #     self.last_error = error
    #
    #     # components
    #     p = self.kp * error
    #     d = self.kd * difference
    #     i = self.ki * self.integral
    #
    #     output = p + d + i
    #
    #     # ------------------------------
    #     # Saturation (if enabled)
    #     # ------------------------------
    #     if self.saturation > 0:
    #         if output >= self.saturation:
    #             output = self.saturation
    #         elif output <= -self.saturation:
    #             output = -self.saturation
    #
    #     # ------------------------------
    #     # Anti-windup (if enabled)
    #     # ------------------------------
    #     if self.antiwindup > 0:
    #         if abs(output) < self.antiwindup:
    #             self.integral += error * dt
    #
    #     return output
