# references.py
from dataclasses import dataclass, field
import numpy as np
from typing import Optional

# Note: this module mirrors the C++ `mrs_multirotor_simulator::reference` types
# but exposes them as top-level classes so you can do:
# from .references import Attitude, TiltHdgRate, AttitudeRate, Actuators, ...

@dataclass
class Actuators:
    """
    Vector of motor throttles scaled as [0, 1].
    `motors` is a numpy array of length n_motors (can be empty / resized by caller).
    """
    motors: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    def __str__(self):
        return f"Actuators = {self.motors.flatten()}"


@dataclass
class ControlGroup:
    """
    Applied torques/throttle normalized to [-1, 1].
    """
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    throttle: float = 0.0

    def __str__(self):
        return (
            f"Control group: roll = {self.roll}, pitch = {self.pitch}, "
            f"yaw = {self.yaw}, throttle = {self.throttle}"
        )


@dataclass
class AttitudeRate:
    """
    Angular rates around body axes (rad/s) and collective throttle.
    """
    rate_x: float = 0.0
    rate_y: float = 0.0
    rate_z: float = 0.0
    throttle: float = 0.0

    def __str__(self):
        return (
            f"Attitude rate: roll = {self.rate_x}, pitch = {self.rate_y}, "
            f"yaw = {self.rate_z}, throttle = {self.throttle}"
        )


@dataclass
class Attitude:
    """
    Desired orientation (rotation matrix) + throttle.
    orientation is a 3x3 numpy array (identity by default).
    """
    orientation: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=float))
    throttle: float = 0.0

    def __post_init__(self):
        # ensure orientation is a 3x3 numpy array
        self.orientation = np.asarray(self.orientation, dtype=float).reshape((3, 3))

    def __str__(self):
        return f"Attitude: throttle {self.throttle}, R =\n{self.orientation}"


@dataclass
class TiltHdgRate:
    """
    Tilt vector (3D) and heading rate (rad/s) + throttle.
    By default tilt_vector matches Eigen::Vector3d::Identity() from C++ (i.e. ones).
    """
    tilt_vector: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=float))
    heading_rate: float = 0.0
    throttle: float = 0.0

    def __post_init__(self):
        self.tilt_vector = np.asarray(self.tilt_vector, dtype=float).reshape(3)

    def __str__(self):
        return (
            f"TiltHdgRate: throttle {self.throttle}, tilt = {self.tilt_vector.T}, "
            f"heading rate = {self.heading_rate}"
        )


@dataclass
class AccelerationHdgRate:
    """
    Acceleration vector + heading rate.
    """
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    heading_rate: float = 0.0

    def __post_init__(self):
        self.acceleration = np.asarray(self.acceleration, dtype=float).reshape(3)

    def __str__(self):
        return f"Acceleration: acc = {self.acceleration.T}, heading rate = {self.heading_rate}"


@dataclass
class AccelerationHdg:
    """
    Acceleration vector + heading (atan2 of body-x projected on ground plane).
    """
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    heading: float = 0.0

    def __post_init__(self):
        self.acceleration = np.asarray(self.acceleration, dtype=float).reshape(3)

    def __str__(self):
        return f"Acceleration: acc = {self.acceleration.T}, heading = {self.heading}"


@dataclass
class VelocityHdgRate:
    """
    Velocity vector + heading rate.
    """
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    heading_rate: float = 0.0

    def __post_init__(self):
        self.velocity = np.asarray(self.velocity, dtype=float).reshape(3)

    def __str__(self):
        return f"Velocity: vel = {self.velocity.T}, heading rate = {self.heading_rate}"


@dataclass
class VelocityHdg:
    """
    Velocity vector + heading (atan2 of body-x projected on ground plane).
    """
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    heading: float = 0.0

    def __post_init__(self):
        self.velocity = np.asarray(self.velocity, dtype=float).reshape(3)

    def __str__(self):
        return f"Velocity: vel = {self.velocity.T}, heading = {self.heading}"


@dataclass
class Position:
    """
    Position vector + heading.
    """
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    heading: float = 0.0

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float).reshape(3)

    def __str__(self):
        return f"Position: pos = {self.position.T}, heading = {self.heading}"
