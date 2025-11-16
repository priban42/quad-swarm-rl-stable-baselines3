import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class DroneVisualizer:
    """
    Simple 3D visualizer for a multirotor.
    Draws the drone as a 'cross' using the rotation matrix R and position.
    """

    def __init__(self, drone_arm_length=0.25, axis_length=1.0):
        self.arm = drone_arm_length
        self.axis_length = axis_length

        # Create plot
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection="3d")

        # Plot initialization
        self.ax.set_xlim(-axis_length, axis_length)
        self.ax.set_ylim(-axis_length, axis_length)
        self.ax.set_zlim(0, axis_length * 2)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.view_init(elev=30, azim=140)

        # Body cross lines
        self.body_lines = [
            self.ax.plot([], [], [], color='black', linewidth=2)[0],
            self.ax.plot([], [], [], color='black', linewidth=2)[0]
        ]

        # Drone body center
        self.center_point = self.ax.scatter([0], [0], [0], color='red')

    def update(self, position, R):
        """
        Update the drone drawing.

        Parameters
        ----------
        position : np.array (3,)
            World position of drone.
        R : np.array (3,3)
            Rotation matrix of drone (world_from_body).
        """

        pos = np.array(position).reshape(3)

        # Drone arms in BODY frame
        arm = self.arm
        body_points = np.array([
            [-arm, 0, 0],
            [ arm, 0, 0],
            [0, -arm, 0],
            [0,  arm, 0]
        ])

        # Rotate into world frame
        world_points = (R @ body_points.T).T + pos

        # First cross arm
        self.body_lines[0].set_data(
            [world_points[0, 0], world_points[1, 0]],
            [world_points[0, 1], world_points[1, 1]]
        )
        self.body_lines[0].set_3d_properties(
            [world_points[0, 2], world_points[1, 2]]
        )

        # Second cross arm
        self.body_lines[1].set_data(
            [world_points[2, 0], world_points[3, 0]],
            [world_points[2, 1], world_points[3, 1]]
        )
        self.body_lines[1].set_3d_properties(
            [world_points[2, 2], world_points[3, 2]]
        )

        # Center point
        self.center_point._offsets3d = (
            [pos[0]], [pos[1]], [pos[2]]
        )

        # keep axes scaling consistent
        self._autoscale_view()

        plt.draw()

    def _autoscale_view(self):
        """
        Auto-adjusts axes limits so drone remains visible in flight.
        """
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        zmin, zmax = self.ax.get_zlim()

        # Expand bounds a bit
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_zlim(zmin, zmax)

    def show(self):
        plt.show()
