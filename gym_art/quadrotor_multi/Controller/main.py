import numpy as np
from MultirotorModel import MultirotorModel
from VelocityController import VelocityController, VelocityHdg
from AttitudeController import AttitudeController
from Mixer import Mixer, ControlGroup
from DroneVizualizer import DroneVisualizer
import matplotlib.pyplot as plt
from references import TiltHdgRate

# -------------------- Setup --------------------
viz = DroneVisualizer(drone_arm_length=0.25)

model = MultirotorModel()
params = model.get_params()

velocity_controller = VelocityController(params)
attitude_controller = AttitudeController(params)
mixer = Mixer(params)

# Target velocity (m/s) in world frame
reference = VelocityHdg(
    velocity=np.array([1.0, 0.0, 0.0]),  # fly forward at 1 m/s
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
    acc_cmd = velocity_controller.get_control_signal(state, reference, dt)

    # Add gravity compensation
    accel_desired = acc_cmd.acceleration + np.array([0.0, 0.0, params.g])

    # Desired tilt vector (body Z axis points opposite gravity)
    tilt_vector = accel_desired / np.linalg.norm(accel_desired)

    # Throttle based on total desired thrust
    throttle_cmd = np.clip(np.linalg.norm(accel_desired) / (params.n_motors * params.kf), 0.0, 1.0)

    # ---------------- Prepare TiltHdgRate reference ----------------
    tilt_ref = TiltHdgRate()
    tilt_ref.tilt_vector = tilt_vector
    tilt_ref.heading_rate = 0.0   # maintain heading
    tilt_ref.throttle = throttle_cmd

    # ---------------- Inner loop: attitude control ----------------
    att_rate_cmd = attitude_controller.get_control_signal(state, tilt_ref, dt)

    # ---------------- Mixer: attitude rates → motor commands ----------------
    actuators = mixer.get_control_signal(att_rate_cmd)

    # ---------------- Apply to UAV model ----------------
    model.set_input(actuators.motors)
    model.step(dt)

    # Record trajectory
    trajectory.append(state.x.copy())
