# multirotor_model.py
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import math

N_INTERNAL_STATES = 18

@dataclass
class ModelParams:
    # default parameters of the x500 quadrotor (same as C++)
    n_motors: int = 4
    g: float = 9.81
    mass: float = 0.028  # 2.0
    kf: float = 0.000000001  # Thrust coefficient
    km: float = 0.0025  # Torque coefficient
    prop_radius: float = 0.00015  # 0.15
    arm_length: float = 0.04596  # 0.25
    body_height: float = 0.003
    motor_time_constant: float = 0.03
    max_rpm: float = 13000  # 7800.0
    min_rpm: float = 1170.0
    air_resistance_coeff: float = 0.30

    J: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    allocation_matrix: np.ndarray = field(default_factory=lambda: np.zeros((4, 4)))

    ground_enabled: bool = False
    ground_z: float = 0.0

    takeoff_patch_enabled: bool = True

    def __post_init__(self):
        # inertia matrix
        a = self.arm_length
        bh = self.body_height
        m = self.mass
        self.J = np.zeros((3, 3))
        self.J[0, 0] = m * (3.0 * a * a + bh * bh) / 12.0
        self.J[1, 1] = m * (3.0 * a * a + bh * bh) / 12.0
        self.J[2, 2] = (m * a * a) / 2.0

        # allocation matrix as in C++
        alloc = np.array([
            [-0.707,  0.707,  0.707, -0.707],
            [-0.707,  0.707, -0.707,  0.707],
            [-1.0,    -1.0,    1.0,   1.0],
            [1.0,     1.0,     1.0,   1.0]
        ], dtype=float)

        alloc[0, :] *= self.arm_length * self.kf
        alloc[1, :] *= self.arm_length * self.kf
        alloc[2, :] *= self.km * (3.0 * self.prop_radius) * self.kf
        alloc[3, :] *= self.kf

        self.allocation_matrix = alloc


@dataclass
class State:
    x: np.ndarray = field(default_factory=lambda: np.zeros(3))          # position
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))          # linear velocity
    v_prev: np.ndarray = field(default_factory=lambda: np.zeros(3))     # previous velocity (for IMU)
    R: np.ndarray = field(default_factory=lambda: np.eye(3))            # rotation matrix (body->world)
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))      # angular velocity in body frame
    motor_rpm: np.ndarray = field(default_factory=lambda: np.zeros(4))  # per-motor RPM


class MultirotorModel:
    def __init__(self, params: Optional[ModelParams] = None, spawn_pos: Optional[np.ndarray] = None, spawn_heading: float = 0.0):
        self.params = params if params is not None else ModelParams()
        self.state = State()
        # ensure motor_rpm length matches params
        self.state.motor_rpm = np.zeros(self.params.n_motors)
        self.input = np.zeros(self.params.n_motors)  # target motor rpm from set_input (converted)
        self.external_force = np.zeros(3)
        self.external_moment = np.zeros(3)
        self.imu_acceleration = np.zeros(3)
        self._initial_pos = np.zeros(3)
        self.internal_state = np.zeros(N_INTERNAL_STATES)

        # spawn if provided
        if spawn_pos is not None:
            self._initial_pos = np.array(spawn_pos, dtype=float)
            self.state.x = np.array(spawn_pos, dtype=float)
            # heading: rotation about z. C++ used AngleAxis(-heading, z)
            ch = float(spawn_heading)
            c, s = math.cos(-ch), math.sin(-ch)
            self.state.R = np.array([[c, -s, 0],
                                     [s,  c, 0],
                                     [0,  0, 1.0]], dtype=float)

        self.initialize_state()
        self.update_internal_state()

    def initialize_state(self):
        self.state.x = np.zeros(3)
        self.state.v = np.zeros(3)
        self.state.v_prev = np.zeros(3)
        self.state.R = np.eye(3)
        self.state.omega = np.zeros(3)
        self.imu_acceleration = np.zeros(3)
        self.state.motor_rpm = np.zeros(self.params.n_motors)
        self.input = np.zeros(self.params.n_motors)
        self.external_force = np.zeros(3)
        self.external_moment = np.zeros(3)

    def update_internal_state(self):
        # layout:
        # [0..2]   x
        # [3..5]   v
        # [6..8]   R[:,0]
        # [9..11]  R[:,1]
        # [12..14] R[:,2]
        # [15..17] omega
        s = self.state
        self.internal_state[:] = 0.0
        self.internal_state[0:3] = s.x
        self.internal_state[3:6] = s.v
        self.internal_state[6:9] = s.R[:, 0]
        self.internal_state[9:12] = s.R[:, 1]
        self.internal_state[12:15] = s.R[:, 2]
        self.internal_state[15:18] = s.omega

    def set_params(self, params: ModelParams):
        self.params = params
        # ensure sizes consistent
        self.state.motor_rpm = np.zeros(self.params.n_motors)
        self.input = np.zeros(self.params.n_motors)

    def get_params(self) -> ModelParams:
        return self.params

    def set_input(self, actuators: np.ndarray):
        """
        actuators: array-like of length n_motors with values in [0,1]
        behaves like C++ setInput: clamps and maps to RPM between min_rpm and max_rpm
        """
        actuators = np.asarray(actuators, dtype=float)
        for i in range(self.params.n_motors):
            val = actuators[i]
            if not np.isfinite(val):
                val = 0.0
            if val < 0.0:
                val = 0.0
            elif val > 1.0:
                val = 1.0
            self.input[i] = self.params.min_rpm + (self.params.max_rpm - self.params.min_rpm) * val

    def apply_force(self, force):
        self.external_force = np.array(force, dtype=float)

    def set_external_force(self, force):
        self.external_force = np.array(force, dtype=float)

    def get_external_force(self):
        return self.external_force

    def set_external_moment(self, moment):
        self.external_moment = np.array(moment, dtype=float)

    def get_external_moment(self):
        return self.external_moment

    def set_state(self, state: State):
        self.state.x = np.array(state.x, dtype=float)
        self.state.v = np.array(state.v, dtype=float)
        self.state.R = np.array(state.R, dtype=float)
        self.state.omega = np.array(state.omega, dtype=float)
        self.state.motor_rpm = np.array(state.motor_rpm, dtype=float)
        self.update_internal_state()

    def set_state_pos(self, pos, heading: float):
        pos = np.asarray(pos, dtype=float)
        self._initial_pos = pos
        self.state.x = pos
        ch = float(heading)
        c, s = math.cos(-ch), math.sin(-ch)
        self.state.R = np.array([[c, -s, 0],
                                 [s,  c, 0],
                                 [0,  0, 1.0]], dtype=float)
        self.update_internal_state()

    def get_state(self) -> State:
        return self.state

    def get_imu_acceleration(self) -> np.ndarray:
        return self.imu_acceleration.copy()

    # ---------- core dynamics (returns derivative vector for internal_state) ----------
    def dynamics(self, x_internal: np.ndarray, t: float = 0.0) -> np.ndarray:
        dxdt = np.zeros_like(x_internal)

        cur_state = State()
        # unpack
        cur_state.x = x_internal[0:3].copy()
        cur_state.v = x_internal[3:6].copy()
        Rcols = np.column_stack((x_internal[6:9], x_internal[9:12], x_internal[12:15]))
        cur_state.R = Rcols.copy()
        cur_state.omega = x_internal[15:18].copy()

        # Re-orthonormalize candidate R the same way as C++: LLT of R^T R, P = L, R = R * P^{-1}
        RtR = cur_state.R.T @ cur_state.R
        # ensure symmetric positive definite, attempt cholesky; if fails, add small epsilon
        eps = 1e-12
        try:
            P = np.linalg.cholesky(RtR)
        except np.linalg.LinAlgError:
            P = np.linalg.cholesky(RtR + eps * np.eye(3))
        R = cur_state.R @ np.linalg.inv(P)

        # skew-symmetric omega tensor (such that R_dot = R * omega_tensor)
        ot = np.zeros((3, 3))
        # mapping consistent with C++:
        # omega_tensor(2,1) = omega(0)
        # omega_tensor(1,2) = -omega(0)
        # omega_tensor(0,2) = omega(1)
        # omega_tensor(2,0) = -omega(1)
        # omega_tensor(1,0) = omega(2)
        # omega_tensor(0,1) = -omega(2)
        w = cur_state.omega
        ot[2, 1] = w[0]
        ot[1, 2] = -w[0]
        ot[0, 2] = w[1]
        ot[2, 0] = -w[1]
        ot[1, 0] = w[2]
        ot[0, 1] = -w[2]

        # use current filtered motor rpm (self.state.motor_rpm) as in C++
        motor_rpm_sq = (self.state.motor_rpm ** 2)
        torque_thrust = self.params.allocation_matrix @ motor_rpm_sq
        thrust = float(torque_thrust[3])

        # air resistance: coeff * pi * arm_length^2 * |v|^2
        vnorm_dir = cur_state.v.copy()
        vnorm = np.linalg.norm(vnorm_dir)
        if vnorm != 0.0:
            vnorm_dir = vnorm_dir / vnorm
        resistance = self.params.air_resistance_coeff * math.pi * (self.params.arm_length ** 2) * (vnorm ** 2)

        # derivatives
        x_dot = cur_state.v
        v_dot = -np.array([0.0, 0.0, self.params.g]) + (thrust * R[:, 2]) / self.params.mass + self.external_force / self.params.mass
        if resistance != 0.0:
            v_dot = v_dot - (resistance * vnorm_dir) / self.params.mass

        R_dot = R @ ot
        # compute omega_dot = J^{-1} (torque - omega x (J * omega) + external_moment)
        J = self.params.J
        # invert J (J is diagonal in the given defaults)
        J_inv = np.linalg.inv(J)
        omega_dot = J_inv @ (torque_thrust[0:3] - np.cross(cur_state.omega, J @ cur_state.omega) + self.external_moment)

        # pack into dxdt
        dxdt[0:3] = x_dot
        dxdt[3:6] = v_dot
        dxdt[6:9] = R_dot[:, 0]
        dxdt[9:12] = R_dot[:, 1]
        dxdt[12:15] = R_dot[:, 2]
        dxdt[15:18] = omega_dot

        # sanitize NaNs (C++ sets to 0)
        nan_mask = ~np.isfinite(dxdt)
        if np.any(nan_mask):
            dxdt[nan_mask] = 0.0

        return dxdt

    # ---------- RK4 step over internal_state ----------
    def step(self, dt: float):
        x0 = self.internal_state.copy()
        save = x0.copy()

        # RK4
        k1 = self.dynamics(x0, 0.0)
        k2 = self.dynamics(x0 + 0.5 * dt * k1, 0.5 * dt)
        k3 = self.dynamics(x0 + 0.5 * dt * k2, 0.5 * dt)
        k4 = self.dynamics(x0 + dt * k3, dt)

        x_new = x0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # NaN check: if any NaN, revert
        if not np.all(np.isfinite(x_new)):
            self.internal_state = save
        else:
            self.internal_state = x_new

        # unpack back into state
        for i in range(3):
            self.state.x[i] = self.internal_state[0 + i]
            self.state.v[i] = self.internal_state[3 + i]
            self.state.R[i, 0] = self.internal_state[6 + i]
            self.state.R[i, 1] = self.internal_state[9 + i]
            self.state.R[i, 2] = self.internal_state[12 + i]
            self.state.omega[i] = self.internal_state[15 + i]

        # motor rpm first-order filter (same formula)
        filter_const = math.exp((-dt) / (self.params.motor_time_constant))
        self.state.motor_rpm = filter_const * self.state.motor_rpm + (1.0 - filter_const) * self.input

        # Re-orthonormalize R (polar decomposition like C++)
        RtR = self.state.R.T @ self.state.R
        try:
            P = np.linalg.cholesky(RtR)
        except np.linalg.LinAlgError:
            P = np.linalg.cholesky(RtR + 1e-12 * np.eye(3))
        Rmat = self.state.R @ np.linalg.inv(P)
        self.state.R = Rmat

        # simulate ground
        if self.params.ground_enabled:
            if self.state.x[2] < self.params.ground_z and self.state.v[2] < 0.0:
                self.state.x[2] = self.params.ground_z
                self.state.v[:] = 0.0
                self.state.omega[:] = 0.0

        # takeoff patch
        hover_rpm = math.sqrt((self.params.mass * self.params.g) / (self.params.n_motors * self.params.kf))
        if self.input.mean() <= 0.90 * hover_rpm:
            if self.state.x[2] < self._initial_pos[2] and self.state.v[2] < 0.0:
                self.state.x[2] = self._initial_pos[2]
                self.state.v[:] = 0.0
                self.state.omega[:] = 0.0
        else:
            self.params.takeoff_patch_enabled = False

        # fabricate IMU accelerometer reading
        # imu_acceleration = R^T * ((v - v_prev)/dt + [0,0,g])
        self.imu_acceleration = self.state.R.T @ (((self.state.v - self.state.v_prev) / dt) + np.array([0.0, 0.0, self.params.g]))
        self.state.v_prev = self.state.v.copy()

        # update internal_state to current
        self.update_internal_state()

    # allow calling this object like C++ operator()
    def __call__(self, x_internal, dxdt_out, t=0.0):
        """
        Fills dxdt_out (array-like) with dynamics at x_internal (array-like).
        This mirrors the C++ operator() signature.
        """
        dx = self.dynamics(np.asarray(x_internal, dtype=float), t)
        dxdt_out[:] = dx

def main():
    model = MultirotorModel()
    model.set_input([0.5, 0.5, 0.5, 0.5])
    model.step(0.01)
    state = model.get_state()
    pass

if __name__ == "__main__":
    main()
# Example usage:

