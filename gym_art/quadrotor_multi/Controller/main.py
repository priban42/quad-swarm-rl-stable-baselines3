import numpy as np
from .MultirotorModel import MultirotorModel
from .VelocityController import VelocityController, VelocityHdg
from .AttitudeController import AttitudeController
from .AccelerationController import AccelerationController
from .RateController import RateController
from .Controller import Controller
from .Mixer import Mixer, ControlGroup
from .DroneVizualizer import DroneVisualizer
import matplotlib.pyplot as plt
from .references import TiltHdgRate

# -------------------- Setup --------------------
viz = DroneVisualizer(drone_arm_length=0.25)

model = MultirotorModel()
params = model.get_params()

velocity_controller = VelocityController(params)
acceleration_controller = AccelerationController(params)
attitude_controller = AttitudeController(params)
rate_controller = RateController(params)
mixer = Mixer(params)

controller = Controller()

# Target velocity (m/s) in world frame
# velocity_hdg_cmd = VelocityHdg(
#     velocity=np.array([1.0, 1.0, 1.0]),  # fly forward at 1 m/s
#     heading=0.0
# )

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

    actuators_cmd = controller.update(state, np.array([0, 0, 0.0, 0], ), dt)
    print(actuators_cmd)
    # actuators_cmd = np.zeros(4)
    model.set_input(actuators_cmd)
    model.step(dt)

    # Record trajectory
    trajectory.append(state.x.copy())
