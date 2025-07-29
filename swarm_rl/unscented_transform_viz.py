import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def nonlinear_transform(x):
    """A sample nonlinear transformation (e.g., twisting a 3D Gaussian)."""
    y = np.empty_like(x)
    y[0] = np.sin(x[0]) + 0.1 * x[2]
    y[1] = np.cos(x[1]) + 0.2 * x[0]
    y[2] = x[2] ** 2
    return y

def generate_sigma_points(mu, cov, alpha=1e-1, beta=2, kappa=0):
    n = mu.shape[0]
    lambda_ = alpha**2 * (n + kappa) - n
    sigma_points = [mu]
    sqrt_matrix = np.linalg.cholesky((n + lambda_) * cov)

    for i in range(n):
        sigma_points.append(mu + sqrt_matrix[:, i])
        sigma_points.append(mu - sqrt_matrix[:, i])

    sigma_points = np.array(sigma_points)

    # Weights
    Wm = np.full(2 * n + 1, 0.5 / (n + lambda_))
    Wc = np.copy(Wm)
    Wm[0] = lambda_ / (n + lambda_)
    Wc[0] = Wm[0] + (1 - alpha**2 + beta)

    return sigma_points, Wm, Wc

def unscented_transform(sigma_points, Wm, Wc, f):
    y_points = np.array([f(p) for p in sigma_points])
    mean = np.sum(Wm[:, None] * y_points, axis=0)
    cov = sum(Wc[i] * np.outer(y_points[i] - mean, y_points[i] - mean) for i in range(len(Wc)))
    return y_points, mean, cov

# --- Mean and Covariance of original distribution
mu = np.array([0.0, 0.0, 0.0])
cov = np.array([
    [0.2, 0.05, 0.0],
    [0.05, 0.2, 0.0],
    [0.0, 0.0, 0.1]
])

# --- Generate sigma points
sigma_points, Wm, Wc = generate_sigma_points(mu, cov)

# --- Apply transformation
transformed_points, transformed_mean, transformed_cov = unscented_transform(sigma_points, Wm, Wc, nonlinear_transform)

# --- Plotting
fig = plt.figure(figsize=(12, 6))

# Before transformation
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(sigma_points[:, 0], sigma_points[:, 1], sigma_points[:, 2], c='blue', label='Sigma Points')
ax1.scatter(*mu, color='black', s=80, label='Original Mean', marker='x')
ax1.set_title("Before Transformation")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.legend()

# After transformation
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2], c='red', label='Transformed Points')
ax2.scatter(*transformed_mean, color='black', s=80, label='Transformed Mean', marker='x')
ax2.set_title("After Transformation")
ax2.set_xlabel("X'")
ax2.set_ylabel("Y'")
ax2.set_zlabel("Z'")
ax2.legend()

plt.tight_layout()
plt.show()
