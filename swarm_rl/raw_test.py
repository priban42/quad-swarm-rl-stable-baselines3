import gymnasium as gym
import torch
import cv2
import numpy as np
from stable_baselines3 import PPO
from swarm_rl.env_wrappers.sb3_quad_env import SB3QuadrotorEnv  # your env wrapper
from gym_art.quadrotor_multi.quad_utils import dict_update_existing
from gym_art.quadrotor_multi.Controller.references import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class RawTester:
    def __init__(self, RENDER=False, quads_mode="static_diff_goal"):
        # MODEL_PATH = "PPO/best_model/best_model.zip"  # path to your trained model
        self.MODEL_PATH = "PPO_1_controller/best_model/best_model.zip"  # path to your trained model
        # MODEL_PATH = "PPO_4_controller/checkpoints/quad_swarm_5199168_steps.zip"  # path to your trained model
        self.NUM_EPISODES = 3
        self.MAX_FRAMES = 120  # maximum frames per episode
        self.episode_duration = 15.0
        self.VIDEO_PATH = "quad_raw_test.mp4"
        self.FPS = 30
        self.RENDER = RENDER
        self.num_of_agents = 1
        # env = SB3QuadrotorEnv(quads_render=True, num_agents=num_of_agents, quads_mode="static_diff_goal")
        self.env = SB3QuadrotorEnv(seed=3, quads_render=self.RENDER, episode_duration=self.episode_duration, num_agents=self.num_of_agents,
                              quads_mode=quads_mode, thrust_noise_ratio=0.0)

        obs, info = self.env.reset()
        if self.RENDER:
            frame = self.env.render()
            frame_height, frame_width, _ = frame.shape
            self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.VIDEO_PATH, self.fourcc, self.FPS, (frame_width, frame_height))

    def reset(self):
        for single_env in self.env.env.env.envs:
            obs, info = self.env.reset()
            single_env.dynamics.set_state(single_env.goal, np.zeros(3), np.eye(3), np.zeros(3))
            single_env.dynamics.set_state(np.array([0, 0, 2]), np.zeros(3), np.eye(3), np.zeros(3))
            single_env.init_state[0] = single_env.goal

    def run_episodes(self, ref=None):
        responses = []
        for episode in range(self.NUM_EPISODES):
            self.reset()
            self.env.env.env.envs[0].pre_controller.reset_all_pids()
            done = False
            terminated = False
            truncated = False
            episode_reward = 0.0
            frame_count = 0
            self.env.env.env.envs[0].ref = ref

            while frame_count < self.MAX_FRAMES:
                action = np.zeros((self.num_of_agents, 2))
                obs, reward, terminated, truncated, info = self.env.step(action)
                responses.append(self.env.env.env.envs[0].response)
                episode_reward += np.array(reward).sum()
                if self.RENDER:
                    frame = self.env.render()  # ensure we get an image array
                    if frame is not None:
                        self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # convert RGB to BGR for OpenCV
                frame_count += 1
            # print(f"Episode {episode + 1}: Reward = {episode_reward}, Frames = {frame_count}")

        if self.RENDER:
            self.video_writer.release()
            self.env.close()
            print(f"Video saved to {self.VIDEO_PATH}")
        return responses

def rotation_x(theta):
    """Return 3×3 rotation matrix for rotation around the X-axis by angle theta (radians)."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])

def rotation_x_from_matrix(R):
    """
    Extract rotation around X-axis (roll) from a 3x3 rotation matrix R.
    Returns angle in radians.
    """
    theta_x = np.arctan2(R[2, 1], R[2, 2])
    return theta_x

def plot_responses(ax, x, y, y_label, ref_height):
    ax.set_xlabel("t (s)")
    # ax.set_title("Step response")
    ax.grid(True)
    ax.set_ylabel(y_label)
    ax.axhline(y=ref_height, color='red')
    ax.plot(x, y)
    pass

def extract_responses(responses, ref, dt=0.005):
    x = []
    y = []
    y_label = None
    ref_height = None
    if isinstance(ref, Position):
        y_label = "pos X (m)"
        for i in range(len(responses)):
            y.append(responses[i].x[0])
            x.append(i*dt)
        ref_height = ref.position[0]
    if isinstance(ref, VelocityHdg):
        y_label = "vel X (m/s)"
        for i in range(len(responses)):
            y.append(responses[i].v[0])
            x.append(i*dt)
        ref_height=ref.velocity[0]
    if isinstance(ref, AccelerationHdg):
        y_label = "acc X (m/s^2)"
        for i in range(len(responses)):
            y.append((responses[i].v[0] - responses[i].v_prev[0])/dt)
            x.append(i*dt)
        ref_height=ref.acceleration[0]
    if isinstance(ref, Attitude):
        y_label = "attitude X (deg)"
        for i in range(len(responses)):
            y.append(rotation_x_from_matrix(responses[i].R)*180/np.pi)
            x.append(i*dt)
        ref_height=rotation_x_from_matrix(ref.orientation)*180/np.pi
    if isinstance(ref, AttitudeRate):
        y_label = "attitude rate X (deg/s)"
        for i in range(len(responses)):
            y.append(responses[i].omega[0]*180/np.pi)
            x.append(i*dt)
        ref_height=ref.rate_x*180/np.pi
    return x, y, y_label, ref_height

def stepinfo(t, y, ref_y, settling_threshold=0.02):
    """
    MATLAB-like stepinfo for time series data.

    Parameters:
        t : array_like
            Time vector
        y : array_like
            Response signal
        settling_threshold : float
            Fraction for settling band (default 2%)

    Returns:
        dict with step response metrics
    """

    t = np.asarray(t)
    y = np.asarray(y)

    y_final = y[-1]
    y_peak = np.max(y)
    peak_time = t[np.argmax(y)]

    # Overshoot (%)
    overshoot = 0.0
    if y_final != 0:
        overshoot = max(0.0, (y_peak - ref_y) / abs(ref_y)) * 100

    # Rise time (10% to 90%)
    y_10 = 0.1 * y_final
    y_90 = 0.9 * y_final

    try:
        t_10 = t[np.where(y >= y_10)[0][0]]
        t_90 = t[np.where(y >= y_90)[0][0]]
        rise_time = t_90 - t_10
    except IndexError:
        rise_time = np.nan

    # Settling time
    band = settling_threshold * abs(ref_y)
    settling_indices = np.where(np.abs(y - ref_y) > band)[0]

    if len(settling_indices) == 0:
        settling_time = t[0]
    else:
        last_outside = settling_indices[-1]
        settling_time = t[last_outside + 1] if last_outside + 1 < len(t) else np.nan

    return {
        "SteadyStateValue": y_final,
        "Peak": y_peak,
        "PeakTime": peak_time,
        "OvershootPercent": overshoot,
        "RiseTime": rise_time,
        "SettlingTime": settling_time,
    }

def plot_stepinfo(ax, info, settling_threshold=0.02):
    y_final = info["SteadyStateValue"]
    peak = info["Peak"]
    peak_time = info["PeakTime"]
    settling_time = info["SettlingTime"]


    # Settling band
    band = settling_threshold * abs(y_final)
    ax.axhline(y_final + band, linestyle=":", linewidth=1)
    ax.axhline(y_final - band, linestyle=":", linewidth=1)

    # Peak marker
    ax.plot(peak_time, peak, marker="o")
    ax.annotate(
        f"Peak = {peak:.3f}, Overshoot = {info['OvershootPercent']:.1f}%",
        xy=(peak_time, peak),
        xytext=(peak_time, peak),
        textcoords="offset points",
        xycoords="data",
        arrowprops=dict(arrowstyle="->"),
    )

    # Settling time marker
    if not np.isnan(settling_time):
        ax.axvline(settling_time, linestyle="--", linewidth=1)
        ax.text(
            settling_time,
            ax.get_ylim()[0],
            f"Ts = {settling_time:.2f}s",
            rotation=90,
            verticalalignment="bottom",
        )

from gym_art.quadrotor_multi.Controller.PositionController import PositionController
from gym_art.quadrotor_multi.Controller.VelocityController import VelocityController
from gym_art.quadrotor_multi.Controller.AccelerationController import AccelerationController
from gym_art.quadrotor_multi.Controller.AttitudeController import AttitudeController
from gym_art.quadrotor_multi.Controller.RateController import RateController

def tune_pid():
    raw_tester = RawTester()
    # refs = [Position(position=raw_tester.env.env.env.envs[0].goal + np.array([0.2, 0, 2]), heading=0),
    refs = [Position(position=np.array([0.2, 0, 2]), heading=0),
    VelocityHdg(velocity=np.array([0.2, 0, 0]), heading=0),
    AccelerationHdg(acceleration=np.array([0.2, 0, 0]), heading=0),
    Attitude(orientation=rotation_x(0.1), throttle=0.6),
    AttitudeRate(rate_x=1, rate_y=0, rate_z=0, throttle=0.6)]
    episode_lengths = [1500, 1500, 400, 120, 120]

    fig, axes = plt.subplots(3, 2, figsize=(12, 4))
    controler_id = 0


    all_params = [PositionController.Params(), VelocityController.Params(), None, AttitudeController.Params(), RateController.Params()]
    all_controlers = [raw_tester.env.env.env.envs[0].pre_controller.position_controller,
                      raw_tester.env.env.env.envs[0].pre_controller.velocity_controller,
                      raw_tester.env.env.env.envs[0].pre_controller.acceleration_controller,
                      raw_tester.env.env.env.envs[0].pre_controller.attitude_controller,
                      raw_tester.env.env.env.envs[0].pre_controller.rate_controller]

    raw_tester.env.env.env.envs[0].pre_controller.position_controller.set_params(PositionController.Params(kp=4.1625, kd=0.5473, ki=0.0023))
    raw_tester.env.env.env.envs[0].pre_controller.velocity_controller.set_params(VelocityController.Params(kp=2.4531, kd=0.0003, ki=0.0382))
    raw_tester.env.env.env.envs[0].pre_controller.attitude_controller.set_params(AttitudeController.Params(kp=11.2081, kd=0.0490, ki=0.0073))
    raw_tester.env.env.env.envs[0].pre_controller.rate_controller.set_params(RateController.Params(kp=3.1222, kd=0.0477, ki=0.0001))

    for controler_id in range(5):
        ax = axes[controler_id//2, controler_id%2]
        ref = refs[controler_id]



        raw_tester.MAX_FRAMES = episode_lengths[controler_id]
        responses = raw_tester.run_episodes(ref)
        x, y, y_label, ref_height = extract_responses(responses, ref)
        step_info = stepinfo(x, y, ref_height)


        plot_responses(ax, x, y, y_label, ref_height)
        plot_stepinfo(ax, step_info)
    plt.show()
    raw_tester.MAX_FRAMES = episode_lengths[controler_id]
    ref = refs[controler_id]
    def optimize():
        params = all_params[controler_id]
        controller = all_controlers[controler_id]

        p0 = np.array([params.kp, params.kd, params.ki])

        def objective(p):
            if np.min(p) < 0:
                return 1e6
            params.kp, params.kd, params.ki = p[0], p[1], p[2]
            controller.set_params(params)
            responses = raw_tester.run_episodes(ref)
            x, y, y_label, ref_height = extract_responses(responses, ref)
            step_info = stepinfo(x, y, ref_height, settling_threshold=0.05)
            cost = step_info["SettlingTime"] + step_info["OvershootPercent"]
            print(f"kp={p[0]:0.4f}, kd={p[1]:0.4f}, ki={p[2]:0.4f}, cost={cost:0.4f}")
            return cost
        result = minimize(
            objective,
            p0,
            method="Nelder-Mead",
            options={
                "maxiter": 200,
                "xatol": 1e-1,
                "fatol": 1e-1,
                "disp": True,
            },
        )
        print("Success:", result.success)
        print("Optimal parameters:", result.x)
        print("Final cost:", result.fun)

        print(result.message)
        print(result.status)
        print(result.nit)
    # optimize()

def main():
    raw_tester = RawTester(RENDER=True, quads_mode="dynamic_same_goal_trajectory")
    # raw_tester = RawTester(RENDER=True, quads_mode="static_diff_goal")
    raw_tester.MAX_FRAMES = 3000
    raw_tester.env.env.env.envs[0].pre_controller.position_controller.set_params(PositionController.Params(kp=4.1625, kd=0.5473, ki=0.0023))
    raw_tester.env.env.env.envs[0].pre_controller.velocity_controller.set_params(VelocityController.Params(kp=2.4531, kd=0.0003, ki=0.0382))
    raw_tester.env.env.env.envs[0].pre_controller.attitude_controller.set_params(AttitudeController.Params(kp=11.2081, kd=0.0490, ki=0.0073))
    raw_tester.env.env.env.envs[0].pre_controller.rate_controller.set_params(RateController.Params(kp=3.1222, kd=0.0477, ki=0.0001))
    responses = raw_tester.run_episodes()
    print("done")

if __name__ == "__main__":
    # tune_pid()
    main()