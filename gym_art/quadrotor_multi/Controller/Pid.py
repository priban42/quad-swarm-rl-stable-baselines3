# pid.py

import math

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

    # ----------------------------------------------------------

    def update(self, error, dt):
        """
        Equivalent to C++ PIDController::update()
        """

        # derivative term
        difference = (error - self.last_error) / dt
        self.last_error = error

        # components
        p = self.kp * error
        d = self.kd * difference
        i = self.ki * self.integral

        output = p + d + i

        # ------------------------------
        # Saturation (if enabled)
        # ------------------------------
        if self.saturation > 0:
            if output >= self.saturation:
                output = self.saturation
            elif output <= -self.saturation:
                output = -self.saturation

        # ------------------------------
        # Anti-windup (if enabled)
        # ------------------------------
        if self.antiwindup > 0:
            if abs(output) < self.antiwindup:
                self.integral += error * dt

        return output
