import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- Camera intrinsics ---
f = 800
cx, cy = 320, 240
width, height = 640, 480
K = np.array([[f, 0, cx],
              [0, f, cy],
              [0, 0, 1]])

# --- Camera extrinsics (Tcw: world -> camera) ---

Twc = np.array([[0, 1, 0, 0],
                [0, 0, 1, -2],
                [1, 0, 0, 4],
                [0, 0, 0, 1]])

Tcw = np.linalg.inv(Twc)



def rotation_matrix(axis, angle):
    """
    Create a 3x3 rotation matrix from an axis and an angle using Rodrigues' formula.


    Parameters:
    axis (array-like): A 3-element array representing the axis of rotation.
    angle (float): Rotation angle in radians.


    Returns:
    numpy.ndarray: A 3x3 rotation matrix.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis) # normalize axis
    x, y, z = axis

    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c

    R = np.array([
    [c + x*x*C, x*y*C - z*s, x*z*C + y*s],
    [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
    [z*x*C - y*s, z*y*C + x*s, c + z*z*C ]
    ])


    return R

def set_axes_equal(ax):
    """Set 3D plot axes to equal scale."""
    limits = np.array([
    ax.get_xlim3d(),
    ax.get_ylim3d(),
    ax.get_zlim3d()
    ])
    spans = limits[:,1] - limits[:,0]
    centers = np.mean(limits, axis=1)
    radius = 0.5 * max(spans)
    ax.set_xlim3d([centers[0]-radius, centers[0]+radius])
    ax.set_ylim3d([centers[1]-radius, centers[1]+radius])
    ax.set_zlim3d([centers[2]-radius, centers[2]+radius])

# --- Visibility function ---
def is_visible(point_w, K, Tcw, width, height):
    """Check if a 3D world point is visible in the camera defined by K and Tcw."""
    Pw_h = np.hstack([point_w, 1])
    Pc_h = Tcw @ Pw_h
    x, y, z = Pc_h[:3]
    if z <= 0:
        return False
    proj = K @ np.array([x, y, z])
    u, v = proj[0]/proj[2], proj[1]/proj[2]
    return (0 <= u < width) and (0 <= v < height)

# --- Draw 3D ellipsoid ---
def draw_ellipsoid(ax, cov, centroid, n_points=20, color='r', alpha=0.2):
    """Draw a 3D ellipsoid from covariance matrix cov and centroid position."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    radii = np.sqrt(eigvals)

    u = np.linspace(0, 2*np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    ellipsoid = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    ellipsoid = ellipsoid @ eigvecs.T + centroid
    x = ellipsoid[:,0].reshape(n_points, n_points)
    y = ellipsoid[:,1].reshape(n_points, n_points)
    z = ellipsoid[:,2].reshape(n_points, n_points)

    ax.plot_surface(x, y, z, rstride=1, cstride=1, color=color, alpha=alpha, linewidth=0)

# --- Draw camera frustum ---
def draw_camera(ax, K, Tcw, width, height, scale=0.01, color='b'):
    """Draw the camera frustum in 3D given intrinsics K and extrinsics Tcw."""
    # Inverse transform (camera -> world)
    Twc = np.linalg.inv(Tcw)

    # Define image corners in pixels
    corners_px = np.array([
        [0, 0, 1],
        [width, 0, 1],
        [width, height, 1],
        [0, height, 1]
    ])

    # Backproject to rays in camera coordinates
    Kinv = np.linalg.inv(K)
    rays_c = (Kinv @ corners_px.T).T

    # Scale rays to get frustum depth (unit depth * scale)
    rays_c = rays_c * (scale / np.linalg.norm(rays_c, axis=1)[:, None])

    # Transform rays to world coordinates
    rays_w = (Twc[:3,:3] @ rays_c.T).T + Twc[:3,3]
    cam_center = Twc[:3,3]

    # Draw pyramid lines
    for r in rays_w:
        ax.plot([cam_center[0], r[0]], [cam_center[1], r[1]], [cam_center[2], r[2]], c=color)
    for i in range(4):
        j = (i+1) % 4
        ax.plot([rays_w[i,0], rays_w[j,0]], [rays_w[i,1], rays_w[j,1]], [rays_w[i,2], rays_w[j,2]], c=color)

# --- Demo ---
np.random.seed(0)
points_w = np.random.uniform(low=[-2, -2, 0], high=[2, 2, 10], size=(200, 3))

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

for p in points_w:
    if is_visible(p, K, Tcw, width, height):
        ax.scatter(*p, c='g')
        p_c = Tcw@np.hstack([p, 1])
        norm = np.cross(np.array([0, 0, 1]), p_c[:3])
        angle = np.arccos(np.dot(np.array([0, 0, 1]), p_c[:3]) / (np.linalg.norm(np.array([0, 0, 1])) * np.linalg.norm(p_c[:3])))
        R_pc = rotation_matrix(norm, angle)
        cov = np.diag([0.01*p_c[2], 0.01*p_c[2], 0.05*p_c[2]**2])

        cov =  Twc[:3, :3] @ R_pc @ cov @ R_pc.T @ Twc[:3, :3].T
        draw_ellipsoid(ax, cov, p, color='g', alpha=0.3)
    else:
        ax.scatter(*p, c='r', marker='x')

# Draw camera
draw_camera(ax, K, Tcw, width, height, scale=2.0, color='b')

ax.set_xlabel('Xw')
ax.set_ylabel('Yw')
ax.set_zlabel('Zw')
ax.set_title('3D Points with Covariance Ellipsoids')

set_axes_equal(ax)

plt.show()