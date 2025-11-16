import numpy as np
from MultirotorModel import MultirotorModel
from VelocityController import VelocityController, VelocityHdg
from AttitudeController import AttitudeController
from AccelerationController import AccelerationController
from RateController import RateController
from Mixer import Mixer, ControlGroup
from DroneVizualizer import DroneVisualizer
import matplotlib.pyplot as plt
from references import TiltHdgRate

# -------------------- Setup --------------------
viz = DroneVisualizer(drone_arm_length=0.25)

model = MultirotorModel()
params = model.get_params()

velocity_controller = VelocityController(params)
acceleration_controller = AccelerationController(params)
attitude_controller = AttitudeController(params)
rate_controller = RateController(params)
mixer = Mixer(params)

# Target velocity (m/s) in world frame
velocity_hdg_cmd = VelocityHdg(
    velocity=np.array([1.0, 1.0, 1.0]),  # fly forward at 1 m/s
    heading=0.0
)

dt = 0.01
simulation_time = 5.0
steps = int(simulation_time / dt)

trajectory = []

# -------------------- Simulation Loop --------------------
for i in range(steps):

    # 1) Read UAV state
    state = model.get_state()  # .v, .x, .R

    viz.update(state.x, state.R)
    plt.pause(0.001)

    # ---------------- Outer loop: velocity control ----------------
    # Compute desired acceleration in world frame
    acceleration_hdg_cmd = velocity_controller.get_control_signal(state, velocity_hdg_cmd, dt)

    attitude_cmd = acceleration_controller.get_control_signal(state, acceleration_hdg_cmd, dt)
    attitude_rate_cmd = attitude_controller.get_control_signal(state, attitude_cmd, dt)
    control_group_cmd = rate_controller.get_control_signal(state, attitude_rate_cmd, dt)
    actuators_cmd = mixer.get_control_signal(control_group_cmd)

    model.set_input(actuators_cmd.motors)
    model.step(dt)

    # Record trajectory
    trajectory.append(state.x.copy())
