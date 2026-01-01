from .MultirotorModel import MultirotorModel
from .VelocityController import VelocityController, VelocityHdg
from .AttitudeController import AttitudeController
from .AccelerationController import AccelerationController
from .PositionController import PositionController, Position
from .RateController import RateController
from .Mixer import Mixer, ControlGroup
from .references import *


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
        self.angle = 0
        self.angular_velocity = 0
        self.MAX_ANGULAR_RATE = (np.pi*80/180)  # π/10 per timestep viz  https://arxiv.org/pdf/2010.08193 V. SIMULATION EXPERIMENTS

    def reset_all_pids(self):
        self.position_controller.initialize_pids()
        self.position_controller.initialize_pids()
        self.velocity_controller.initialize_pids()
        self.attitude_controller.initialize_pids()
        self.rate_controller.initialize_pids()

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
        velocity_hdg_cmd.velocity[:2] = command[:2]*2
        # velocity_hdg_cmd.heading = 0.0

        acceleration_hdg_cmd = self.velocity_controller.get_control_signal(state, velocity_hdg_cmd, dt)
        attitude_cmd = self.acceleration_controller.get_control_signal(state, acceleration_hdg_cmd, dt)
        attitude_rate_cmd = self.attitude_controller.get_control_signal(state, attitude_cmd, dt)
        control_group_cmd = self.rate_controller.get_control_signal(state, attitude_rate_cmd, dt)
        actuators_cmd = self.mixer.get_control_signal(control_group_cmd)
        # if np.isnan(actuators_cmd.motors[0]):
        #     pass
        return actuators_cmd.motors

    def update_vel_height_dir(self, state, command, height, dt):
        '''
        inspired by https://arxiv.org/pdf/2304.03443
        command = [angular_velocity, linear_velocity]
        '''
        self.angular_velocity = command[0]
        # self.angle = (self.angle + command[0]*dt*self.MAX_ANGULAR_RATE)
        self.angle = command[0]*np.pi
        # self.angle = (self.angle + 1*dt*self.MAX_ANGULAR_RATE)
        self.angle = (self.angle + np.pi)%(2*np.pi) - np.pi
        dir_vec = np.array([np.cos(self.angle), np.sin(self.angle)])
        velocity = (command[1] + 1)*0.3
        position_cmd = Position(position=np.array([0, 0, height]), heading=0.0)
        velocity_hdg_cmd = self.position_controller.get_control_signal(state, position_cmd, dt)
        velocity_hdg_cmd.velocity[:2] = dir_vec*velocity
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

    def test_step_response(self, state, dt, ref=None, file_name="pos_response.p"):
        position_cmd, velocity_hdg_cmd, acceleration_hdg_cmd, attitude_cmd, attitude_rate_cmd = None, None, None, None, None
        if isinstance(ref, Position):
            position_cmd = ref
        if isinstance(ref, VelocityHdg):
            velocity_hdg_cmd = ref
        if isinstance(ref, AccelerationHdg):
            acceleration_hdg_cmd = ref
        if isinstance(ref, Attitude):
            attitude_cmd = ref
        if isinstance(ref, AttitudeRate):
            attitude_rate_cmd = ref

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
        # if self.num_steps == 1000:
        #     with open(file_name, "wb") as f:
        #         pickle.dump(self.log_dict, f)

        return actuators_cmd.motors

