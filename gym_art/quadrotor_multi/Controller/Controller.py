from .MultirotorModel import MultirotorModel
from .VelocityController import VelocityController, VelocityHdg
from .AttitudeController import AttitudeController
from .AccelerationController import AccelerationController
from .PositionController import PositionController, Position
from .RateController import RateController
from .Mixer import Mixer, ControlGroup

import numpy as np

class Controller:
    def __init__(self):

        self.model = MultirotorModel()
        self.params = self.model.get_params()

        self.position_controller = PositionController(self.params)
        self.velocity_controller = VelocityController(self.params)
        self.acceleration_controller = AccelerationController(self.params)
        self.attitude_controller = AttitudeController(self.params)
        self.rate_controller = RateController(self.params)
        self.mixer = Mixer(self.params)

    def update_vel(self, state, command, dt):
        velocity_hdg_cmd = VelocityHdg(
            velocity=np.array([0.0, 0.0, 0.0]),  # fly forward at 1 m/s
            heading=0.0
        )
        velocity_hdg_cmd.velocity += command[:3]*6
        velocity_hdg_cmd.heading += command[3]

        # position_cmd = Position(position=command, heading=0.0)
        # velocity_hdg_cmd = self.position_controller.get_control_signal(state, position_cmd, dt)

        acceleration_hdg_cmd = self.velocity_controller.get_control_signal(state, velocity_hdg_cmd, dt)
        attitude_cmd = self.acceleration_controller.get_control_signal(state, acceleration_hdg_cmd, dt)
        attitude_rate_cmd = self.attitude_controller.get_control_signal(state, attitude_cmd, dt)
        control_group_cmd = self.rate_controller.get_control_signal(state, attitude_rate_cmd, dt)
        actuators_cmd = self.mixer.get_control_signal(control_group_cmd)
        # if np.isnan(actuators_cmd.motors[0]):
        #     pass
        return actuators_cmd.motors

    def update_pos(self, state, command, dt):
        position_cmd = Position(position=command, heading=0.0)
        velocity_hdg_cmd = self.position_controller.get_control_signal(state, position_cmd, dt)

        acceleration_hdg_cmd = self.velocity_controller.get_control_signal(state, velocity_hdg_cmd, dt)
        attitude_cmd = self.acceleration_controller.get_control_signal(state, acceleration_hdg_cmd, dt)
        attitude_rate_cmd = self.attitude_controller.get_control_signal(state, attitude_cmd, dt)
        control_group_cmd = self.rate_controller.get_control_signal(state, attitude_rate_cmd, dt)
        actuators_cmd = self.mixer.get_control_signal(control_group_cmd)
        # if np.isnan(actuators_cmd.motors[0]):
        #     pass
        return actuators_cmd.motors