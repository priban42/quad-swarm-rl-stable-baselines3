import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from typing import Optional

from numpy.ma.core import arctan


class SceneVisualizer:
    """
    2D top-down scene visualizer for camera-based object detection.

    Coordinate frame: X = right, Y = forward (camera looks along +Y).
    All units in meters unless stated otherwise.
    """

    def __init__(
        self,
        fov_deg: float = 60.0,
        max_range: float = 20.0,
        figsize: tuple = (9, 9),
        dark_theme: bool = True,
    ):
        self.fov_deg = fov_deg
        self.max_range = max_range
        self.font_size=16

        # ── theme ──────────────────────────────────────────────────────────
        if dark_theme:
            self.c_bg      = "#0d1117"
            self.c_grid    = "#21262d"
            self.c_fov     = "#1f6feb"
            self.c_camera  = "#58a6ff"
            self.c_circle  = "#3fb950"
            self.c_line    = "#f78166"
            self.c_ellipse = "#d2a8ff"
            self.c_text    = "#c9d1d9"
            plt.style.use("dark_background")
        else:
            self.c_bg      = "#ffffff"
            self.c_grid    = "#e0e0e0"
            self.c_fov     = "#1565c0"
            self.c_camera  = "#0d47a1"
            self.c_circle  = "#2e7d32"
            self.c_line    = "#c62828"
            self.c_ellipse = "#6a1b9a"
            self.c_text    = "#212121"
            plt.style.use("default")

        self.fig, self.ax = plt.subplots(figsize=figsize)
        self._setup_axes()

    # ── internal helpers ───────────────────────────────────────────────────

    def _setup_axes(self):
        ax = self.ax
        ax.set_facecolor(self.c_bg)
        self.fig.patch.set_facecolor(self.c_bg)
        ax.set_aspect("equal")
        ax.set_xlim(-self.max_range, self.max_range)
        ax.set_ylim(-2, self.max_range)
        ax.set_xlabel("X  (m)", color=self.c_text)
        ax.set_ylabel("Y  (m)", color=self.c_text)
        ax.tick_params(colors=self.c_text)
        for spine in ax.spines.values():
            spine.set_edgecolor(self.c_grid)
        ax.grid(True, color=self.c_grid, linewidth=0.5, linestyle="--")
        ax.set_title("Camera Scene  —  top-down view", color=self.c_text, fontsize=13)

    def _fov_half_angle_rad(self):
        return np.deg2rad(self.fov_deg / 2.0)

    # ── public drawing API ─────────────────────────────────────────────────

    def draw_camera(self, label: str = "Camera"):
        """Draw the camera (observer) at the origin."""
        ax = self.ax
        # body
        ax.plot(0, 0, marker="D", markersize=10, color=self.c_camera, zorder=5)
        # forward axis
        ax.annotate(
            "", xy=(0, 1.2), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=self.c_camera, lw=1.5),
        )
        ax.text(0.15, 1.3, label, color=self.c_camera, fontsize=self.font_size, va="center")

    def draw_fov(self, alpha_fill: float = 0.08, alpha_edge: float = 0.6):
        """Draw the camera field-of-view wedge."""
        half = self._fov_half_angle_rad()
        # FOV as a Wedge (angle measured from +X axis, camera looks along +Y)
        wedge_start = 90 - self.fov_deg / 2.0   # degrees from +X
        wedge = patches.Wedge(
            center=(0, 0),
            r=self.max_range,
            theta1=wedge_start,
            theta2=wedge_start + self.fov_deg,
            facecolor=self.c_fov,
            edgecolor=self.c_fov,
            alpha=alpha_fill,
            zorder=1,
        )
        self.ax.add_patch(wedge)
        # boundary rays
        for sign in (-1, 1):
            angle = sign * half
            ex = self.max_range * np.sin(angle)
            ey = self.max_range * np.cos(angle)
            self.ax.plot(
                [0, ex], [0, ey],
                color=self.c_fov, lw=1.2, alpha=alpha_edge,
                linestyle="--", zorder=2,
            )
        # range arc
        arc = patches.Arc(
            (0, 0), 2 * self.max_range, 2 * self.max_range,
            angle=0,
            theta1=wedge_start, theta2=wedge_start + self.fov_deg,
            color=self.c_fov, lw=1.0, alpha=alpha_edge, zorder=2,
        )
        self.ax.add_patch(arc)
        # FOV label
        self.ax.text(
            0, self.max_range * 0.92,
            f"FOV {self.fov_deg:.0f}°",
            color=self.c_fov, fontsize=self.font_size, ha="center",
        )

    def draw_circle(
        self,
        position: np.ndarray,
        radius: float,
        label: Optional[str] = None,
        color: Optional[str] = None,
        alpha: float = 0.25,
    ):
        """
        Draw a circular object (top-down cross-section).

        Args:
            position: (x, y) world position in meters
            radius:   object radius in meters
            label:    optional text label
            color:    override default circle color
        """
        c = color or self.c_circle
        x, y = position
        circle = patches.Circle(
            (x, y), radius,
            facecolor=c, edgecolor=c,
            alpha=alpha, lw=2, zorder=3,
        )
        self.ax.add_patch(circle)
        self.ax.plot(x, y, "+", color=c, markersize=8, zorder=4)
        if label:
            self.ax.text(
                x, y + radius + 0.3, label,
                color=c, fontsize=self.font_size, ha="center", va="bottom",
            )

    def draw_line(
        self,
        start: np.ndarray,
        end: np.ndarray,
        label: Optional[str] = None,
        color: Optional[str] = None,
        linestyle: str = "-",
        lw: float = 1.5,
        arrow: bool = False,
    ):
        """
        Draw a line (e.g. a measurement ray or association line).

        Args:
            start:     (x, y) start point
            end:       (x, y) end point
            arrow:     draw an arrowhead at `end`
            linestyle: matplotlib linestyle string
        """
        c = color or self.c_line
        xs, ys = np.array(start), np.array(end)
        if arrow:
            self.ax.annotate(
                "", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw),
            )
        else:
            self.ax.plot(
                [xs[0], ys[0]], [xs[1], ys[1]],
                color=c, lw=lw, linestyle=linestyle, zorder=3,
            )
        if label:
            mid = (xs + ys) / 2.0
            self.ax.text(
                mid[0] + 0.1, mid[1] + 0.1, label,
                color=c, fontsize=self.font_size,
            )

    def draw_measurement_ray(self, position: np.ndarray, label: Optional[str] = None):
        """Convenience: draw a ray from the origin to a detected object."""
        self.draw_line(
            np.array([0.0, 0.0]), position,
            label=label, color=self.c_line,
            linestyle="--", lw=1.2, arrow=True,
        )

    def draw_covariance_ellipse(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        n_std: float = 2.0,
        label: Optional[str] = None,
        color: Optional[str] = None,
        alpha: float = 0.35,
    ):
        """
        Draw a covariance ellipse (1-σ, 2-σ, etc.) for a 2×2 covariance matrix.

        Args:
            mean:  (x, y) center
            cov:   2×2 numpy covariance matrix
            n_std: number of standard deviations to draw
            label: optional text label
        """
        c = color or self.c_ellipse
        # Eigen-decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Angle of the major axis
        angle_rad = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        angle_deg = np.rad2deg(angle_rad)
        # Axes lengths = n_std * sqrt(eigenvalue)
        width  = 2 * n_std * np.sqrt(np.maximum(eigenvalues[0], 0))
        height = 2 * n_std * np.sqrt(np.maximum(eigenvalues[1], 0))

        ellipse = patches.Ellipse(
            mean, width, height,
            angle=angle_deg,
            facecolor=c, edgecolor=c,
            alpha=alpha, lw=1.5, linestyle="-", zorder=4,
        )
        self.ax.add_patch(ellipse)
        if label:
            self.ax.text(
                mean[0], mean[1] + height / 2 + 0.2, label,
                color=c, fontsize=self.font_size, ha="center",
            )

    def draw_range_rings(self, step: float = 5.0, alpha: float = 0.2):
        """Draw dashed range rings at fixed intervals."""
        r = step
        while r <= self.max_range:
            ring = patches.Circle(
                (0, 0), r,
                fill=False, edgecolor=self.c_grid,
                lw=0.8, linestyle=":", alpha=alpha, zorder=1,
            )
            self.ax.add_patch(ring)
            self.ax.text(
                0.1, r, f"{r:.0f}m",
                color=self.c_grid, fontsize=self.font_size, va="bottom",
            )
            r += step

    def draw_angle_uncertainty_arc(
            self,
            center: np.ndarray,
            point_on_arc: np.ndarray,
            angle_deg: float,
            color: Optional[str] = None,
            lw: float = 2,
            alpha: float = 0.6,
            label: Optional[str] = None,
            tick_length: float = 0.5,
    ):
        """
        Draw an arc representing angular uncertainty around a detection ray.

        The arc is centered at `center` (e.g. the camera origin), passes through
        `point_on_arc` (the detected object position), and spans ±half_angle_deg
        around the ray to that point.

        Perpendicular tick marks are drawn at each end of the arc to clearly
        visualize the angular spread.

        Args:
            center:         (x, y) arc center — typically the camera origin (0, 0)
            point_on_arc:   (x, y) point that sits at the middle of the arc
            half_angle_deg: half-width of the arc in degrees (total arc = 2× this)
            color:          line color (defaults to self.c_ellipse)
            lw:             line width
            alpha:          opacity
            label:          optional text label placed at the arc midpoint
            tick_length:    length of the perpendicular end ticks in meters
        """
        half_angle_deg = angle_deg/2
        c = color or self.c_ellipse

        center = np.asarray(center, dtype=float)
        point_on_arc = np.asarray(point_on_arc, dtype=float)

        # radius = distance from center to the point on the arc
        radius = np.linalg.norm(point_on_arc - center)
        if radius < 1e-9:
            return

        # angle of the central ray (from +X axis, in degrees)
        delta = point_on_arc - center
        ray_angle = np.rad2deg(np.arctan2(delta[1], delta[0]))

        theta1 = ray_angle - half_angle_deg
        theta2 = ray_angle + half_angle_deg

        # ── arc ───────────────────────────────────────────────────────────────
        arc = patches.Arc(
            center,
            2 * radius, 2 * radius,
            angle=0,
            theta1=theta1, theta2=theta2,
            color=c, lw=lw, alpha=alpha, zorder=5,
        )
        self.ax.add_patch(arc)

        # ── perpendicular end ticks ───────────────────────────────────────────
        for theta_deg in (theta1, theta2):
            theta_rad = np.deg2rad(theta_deg)

            # point on the arc at this end
            arc_pt = center + radius * np.array([np.cos(theta_rad), np.sin(theta_rad)])

            # unit radial direction at this end (outward from center)
            radial = np.array([np.cos(theta_rad), np.sin(theta_rad)])

            # perpendicular = rotate radial 90° (either direction is fine; we
            # draw the tick symmetrically so direction doesn't matter)
            # perp = np.array([-radial[1], radial[0]])
            perp = np.array([radial[0], radial[1]])

            tick_start = arc_pt - perp * (tick_length / 2)
            tick_end = arc_pt + perp * (tick_length / 2)

            self.ax.plot(
                [tick_start[0], tick_end[0]],
                [tick_start[1], tick_end[1]],
                color=c, lw=lw, alpha=alpha, zorder=5,
            )

        # ── optional label at arc midpoint ────────────────────────────────────
        label = f"σ={angle_deg:.2f}˚"
        if label:
            mid_rad = np.deg2rad(ray_angle)
            label_pt = center + (radius + tick_length * 0.8) * np.array(
                [np.cos(mid_rad), np.sin(mid_rad)]
            )
            self.ax.text(
                label_pt[0], label_pt[1], label,
                color=c, fontsize=self.font_size, ha="center", va="center",
            )

    def draw_distance_uncertainty_bar(
            self,
            center: np.ndarray,
            camera_pos: np.ndarray,
            uncertainty_m: float,
            color: Optional[str] = None,
            lw: float = 2,
            alpha: float = 0.6,
            tick_length: float = 0.5,
    ):
        """
        Draw a line segment centered on `center` pointing toward `camera_pos`,
        representing distance/range uncertainty.

        Perpendicular tick marks are drawn at each end, mirroring the style of
        draw_angle_uncertainty_arc.

        Args:
            center:        (x, y) object position (bar is centered here)
            camera_pos:    (x, y) camera/observer position — defines the direction
            uncertainty_m: total length of the bar in meters (±uncertainty_m/2)
            color:         line color (defaults to self.c_line)
            lw:            line width
            alpha:         opacity
            tick_length:   length of the perpendicular end ticks in meters
        """
        c = color or self.c_line

        center = np.asarray(center, dtype=float)
        camera_pos = np.asarray(camera_pos, dtype=float)

        # unit vector pointing from object toward camera
        delta = camera_pos - center
        dist = np.linalg.norm(delta)
        if dist < 1e-9:
            return
        radial = delta / dist  # toward camera
        perp = np.array([-radial[1], radial[0]])  # perpendicular

        half = uncertainty_m / 2.0
        bar_start = center - radial * half  # far end
        bar_end = center + radial * half  # near end (toward cam)

        # ── central bar ───────────────────────────────────────────────────────
        self.ax.plot(
            [bar_start[0], bar_end[0]],
            [bar_start[1], bar_end[1]],
            color=c, lw=lw, alpha=alpha, zorder=5,
        )

        # ── perpendicular end ticks ───────────────────────────────────────────
        for endpoint in (bar_start, bar_end):
            tick_start = endpoint - perp * (tick_length / 2)
            tick_end = endpoint + perp * (tick_length / 2)
            self.ax.plot(
                [tick_start[0], tick_end[0]],
                [tick_start[1], tick_end[1]],
                color=c, lw=lw, alpha=alpha, zorder=5,
            )

        # ── label at bar midpoint, offset perpendicularly ────────────────────
        label = f"σ={uncertainty_m:.2f}m"
        label_pt = center + perp * (tick_length * 2.5)
        self.ax.text(
            label_pt[0], label_pt[1], label,
            color=c, fontsize=self.font_size, ha="center", va="center",
        )

    def clear(self):
        """Clear the axes and redraw the base frame."""
        self.ax.cla()
        self._setup_axes()

    def show(self, tight: bool = True):
        if tight:
            self.fig.tight_layout()
        plt.show()

    def save(self, path: str, dpi: int = 150):
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight",
                         facecolor=self.c_bg)

def circle_intersection(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float) -> np.ndarray:
    """
    Compute the two intersection points of two circles.

    Args:
        c1: (x, y) center of circle 1
        r1: radius of circle 1
        c2: (x, y) center of circle 2
        r2: radius of circle 2

    Returns:
        (2, 2) array — each row is one intersection point [x, y]
    """
    d = np.linalg.norm(c2 - c1)
    # distance from c1 to the radical line
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    # half-height of the chord at the radical line
    h = np.sqrt(r1**2 - a**2)
    # point on the line c1→c2 at distance a
    radial = (c2 - c1) / d
    mid    = c1 + a * radial
    # perpendicular direction
    perp = np.array([-radial[1], radial[0]])
    p1 = mid + h * perp
    p2 = mid - h * perp
    return p1, p2

def simulate_camera_measurement(
                        center_2d: np.ndarray,
                        known_size_m: float,
                        focal_length_m: float,
                        camera_noise_px: float,
                        fov_deg:float = 70,
                        camera_resolution:float = 1080.0):

    l_orig = np.linalg.norm(center_2d)
    angle_orig = np.arctan2(center_2d[0], center_2d[1])
    r = known_size_m/2
    f = focal_length_m
    w = 2 * np.tan((fov_deg/2) * np.pi / 180) * f  # widh of the sensor
    # dir = center_2d/y
    # perp = np.array(-dir[1], dir[0])
    x1, x2 = circle_intersection(center_2d, r, center_2d/2, np.linalg.norm(center_2d)/2)
    u1_orig = x1[0] * f/x1[1]
    u2_orig = x2[0] * f/x2[1]
    u1_px = u1_orig*camera_resolution/w
    u2_px = u2_orig*camera_resolution/w
    u1_px += np.random.normal(0)*camera_noise_px
    u2_px += np.random.normal(0)*camera_noise_px
    u1 = u1_px*w/camera_resolution
    u2 = u2_px*w/camera_resolution
    alpha = abs(arctan(u1/f)-arctan(u2/f))
    l = r/(np.sin(alpha/2))
    # assert abs(l-l_orig)< 0.00001
    dist = focal_length_m*known_size_m/abs(u1-u2)
    angle = (arctan(u1/f) + arctan(u2/f))/2
    return dist, angle

def sample_camera_measurements(center):
    all_d = []
    all_a = []
    for i in range(500):
        d, a = simulate_camera_measurement(center, 0.2, 0.035, 2)
        all_d.append(d)
        all_a.append(a)
    d_sigma = np.std(np.array(all_d))
    a_sigma = np.std(np.array(all_a))
    return d_sigma, a_sigma

def main():
    # ── scene setup ───────────────────────────────────────────────────────────

    viz = SceneVisualizer(fov_deg=70, max_range=15, dark_theme=False)

    viz.draw_range_rings(step=5)
    viz.draw_fov()
    viz.draw_camera()

    # ── objects ───────────────────────────────────────────────────────────────
    targets = [
        (np.array([0.0, 3.0]), 0.5, "Target A"),
        (np.array([-5.0, 10.0]), 0.5, "Target B"),
        (np.array([2.0, 6.0]), 0.3, "Target C"),
        (np.array([7.0, 13.0]), 0.3, "Target D"),
    ]

    for pos, radius, name in targets:
        viz.draw_circle(pos, radius, label=name)
        viz.draw_measurement_ray(pos, label=f"d={np.linalg.norm(pos):.1f}m")
        d_sigma, a_sigma = sample_camera_measurements(pos)
        viz.draw_angle_uncertainty_arc(
            center=np.array([0.0, 0.0]),
            point_on_arc=pos,
            angle_deg=a_sigma,
        )

        viz.draw_distance_uncertainty_bar(
            center=pos, camera_pos=np.array([0.0, 0.0]), uncertainty_m=d_sigma
        )

    # ── covariance ellipses (e.g. from EKF) ───────────────────────────────────
    # cov_a = np.array([[0.4, 0.2],
    #                   [0.2, 0.8]])
    # viz.draw_covariance_ellipse(np.array([3.0, 8.0]), cov_a, n_std=2, label="2σ")
    #
    # cov_b = np.array([[1.2, -0.3],
    #                   [-0.3, 0.5]])
    # viz.draw_covariance_ellipse(np.array([-4.0, 10.0]), cov_b, n_std=2, label="2σ")



    # ── arbitrary lines (e.g. associations, ground truth) ─────────────────────
    # viz.draw_line(
    #     np.array([3.0, 8.0]), np.array([-4.0, 10.0]),
    #     label="association", linestyle=":", lw=1.0,
    # )

    viz.show()

if __name__ == "__main__":
    main()