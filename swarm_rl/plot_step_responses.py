import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_entry(entry, ax, type="pos", dt=0.005):
    ax.set_xlabel("t (s)")
    # ax.set_title("Step response")
    ax.grid(True)
    x = []
    y = []
    if type == "pos":
        ax.set_ylabel("pos X (m)")
        for i in range(len(entry["state"])):
            y.append(entry["state"][i].x[0])
            x.append(i*dt)
        ax.axhline(y=entry["ref"][0][0].position[0], color='red')
    if type == "vel":
        ax.set_ylabel("vel X (m/s)")
        for i in range(len(entry["state"])):
            y.append(entry["state"][i].v[0])
            x.append(i*dt)
        ax.axhline(y=entry["ref"][0][1].velocity[0], color='red')
    if type == "acc":
        ax.set_ylabel("acc X (m/s^2)")
        for i in range(len(entry["state"])):
            y.append((entry["state"][i].v[0] - entry["state"][i].v_prev[0])/dt)
            x.append(i*dt)
        ax.axhline(y=entry["ref"][0][2].acceleration[0], color='red')
        ax.set_ylim(-0.5, 2)
    if type == "att":
        ax.set_ylabel("attitude X (deg)")
        for i in range(len(entry["state"])):
            y.append(rotation_x_from_matrix(entry["state"][i].R*180/np.pi))
            x.append(i*dt)
        ax.axhline(y=rotation_x_from_matrix(entry["ref"][0][3].orientation*180/np.pi), color='red')
        ax.set_ylim(-0.5, 2)
    if type == "att_rate":
        ax.set_ylabel("attitude rate X (deg/s)")
        for i in range(len(entry["state"])):
            y.append(entry["state"][i].omega[0]*180/np.pi)
            x.append(i*dt)
        ax.axhline(y=entry["ref"][0][4].rate_x*180/np.pi, color='red')
        ax.set_ylim(-2, 120)
    ax.plot(x, y)
    pass

def rotation_x_from_matrix(R):
    """
    Extract rotation around X-axis (roll) from a 3x3 rotation matrix R.
    Returns angle in radians.
    """
    theta_x = np.arctan2(R[2, 1], R[2, 2])
    return theta_x

def main():
    # --- Load pickle file ---
    paths = ["pos_response.p", "vel_response.p", "acc_response.p", "att_response.p", "att_rate_response.p"]
    data = []
    fig, axes = plt.subplots(3, 2, figsize=(12, 4))
    for path in paths:
        with open(path, "rb") as f:
            data.append(pickle.load(f))
    plot_entry(data[0], axes[0, 0], type="pos")
    plot_entry(data[1], axes[0, 1], type="vel")
    plot_entry(data[2], axes[1, 0], type="acc")
    plot_entry(data[3], axes[1, 1], type="att")
    plot_entry(data[4], axes[2, 0], type="att_rate")
    plt.show()



if __name__ == "__main__":
    main()

