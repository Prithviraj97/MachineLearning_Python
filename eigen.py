import numpy as np
import matplotlib.pyplot as plt

# Define matrix A
A = np.array([[2, 1], [1, 2]])

# Define eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eig(A)

# Generate a random vector for transformation demo
v = np.array([2, 1])
Av = A @ v

# Eigenvectors scaled for visualization
eigvec1 = eigvecs[:, 0] * eigvals[0] * 2
eigvec2 = eigvecs[:, 1] * eigvals[1] * 2

# Plot original vector and its transformation
plt.figure(figsize=(8, 8))
ax = plt.gca()
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)

# Original vector
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color="blue", label="Original Vector v")

# Transformed vector
plt.quiver(0, 0, Av[0], Av[1], angles='xy', scale_units='xy', scale=1, color="red", label="Transformed Av")

# Eigenvectors
plt.quiver(0, 0, eigvec1[0], eigvec1[1], angles='xy', scale_units='xy', scale=1, color="green", label="Eigenvector λ1=%.1f" % eigvals[0])
plt.quiver(0, 0, eigvec2[0], eigvec2[1], angles='xy', scale_units='xy', scale=1, color="purple", label="Eigenvector λ2=%.1f" % eigvals[1])

plt.legend()
plt.title("Eigenvectors and Transformation Example")
plt.grid(True)
plt.savefig("eigenvectors_transformation.png")
plt.show()
