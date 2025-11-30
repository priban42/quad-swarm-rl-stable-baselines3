from .MultirotorModel import MultirotorModel
from .VelocityController import VelocityController, VelocityHdg
from .AttitudeController import AttitudeController
from .AccelerationController import AccelerationController
from .PositionController import PositionController, Position
from .RateController import RateController
from .Mixer import Mixer, ControlGroup


import numpy as np
import pickle

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
        self.num_steps = 0
        self.log_dict = {"state":[], "ref":[], "ctrl":[]}

    def update_vel(self, state, command, dt):
        velocity_hdg_cmd = VelocityHdg(
            velocity=np.array([0.0, 0.0, 0.0]),
            heading=0.0
        )
        velocity_hdg_cmd.velocity += command[:3]*0.3
        velocity_hdg_cmd.velocity[2] += 2
        velocity_hdg_cmd.heading += command[3]*0

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

    def update_vel_height(self, state, command, height, dt):
        position_cmd = Position(position=np.array([0, 0, height]), heading=0.0)
        velocity_hdg_cmd = self.position_controller.get_control_signal(state, position_cmd, dt)
        velocity_hdg_cmd.velocity[:2] = command[:2]*0.6
        # velocity_hdg_cmd.heading = 0.0

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

    def test_step_response(self, state, dt, position_cmd=None, velocity_hdg_cmd=None, acceleration_hdg_cmd=None, attitude_cmd=None, attitude_rate_cmd=None, file_name="pos_response.p"):

        self.log_dict["state"].append(state)
        self.log_dict["ref"].append([position_cmd, velocity_hdg_cmd, acceleration_hdg_cmd, attitude_cmd, attitude_rate_cmd])
        if position_cmd is None:
            position_cmd = Position(position=np.array([0, 0, 0]), heading=0.0)
        if velocity_hdg_cmd is None:
            velocity_hdg_cmd = self.position_controller.get_control_signal(state, position_cmd, dt)
        if acceleration_hdg_cmd is None:
            acceleration_hdg_cmd = self.velocity_controller.get_control_signal(state, velocity_hdg_cmd, dt)
        if attitude_cmd is None:
            attitude_cmd = self.acceleration_controller.get_control_signal(state, acceleration_hdg_cmd, dt)
        if attitude_rate_cmd is None:
            attitude_rate_cmd = self.attitude_controller.get_control_signal(state, attitude_cmd, dt)
        control_group_cmd = self.rate_controller.get_control_signal(state, attitude_rate_cmd, dt)
        actuators_cmd = self.mixer.get_control_signal(control_group_cmd)
        self.log_dict["ctrl"].append([position_cmd, velocity_hdg_cmd, acceleration_hdg_cmd, attitude_cmd, attitude_rate_cmd])
        self.num_steps += 1
        if self.num_steps == 1000:
            with open(file_name, "wb") as f:
                pickle.dump(self.log_dict, f)

        return actuators_cmd.motors