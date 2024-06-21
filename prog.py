import numpy as np
import matplotlib.pyplot as plt

def initialize_grid(nx, ny):
    """
    Initialize the grid with boundary conditions.
    """
    grid = np.zeros((ny, nx))
    # Set boundary conditions (for example, a hot boundary on all sides)
    grid[:, 0] = 100  # left boundary
    grid[:, -1] = 100  # right boundary
    grid[0, :] = 100  # top boundary
    grid[-1, :] = 100  # bottom boundary
    return grid

def solve_heat_equation(grid, dx, dy, dt, steps):
    """
    Solve the heat equation using finite difference method.
    """
    ny, nx = grid.shape
    new_grid = np.copy(grid)
    
    for _ in range(steps):
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                new_grid[i, j] = grid[i, j] + dt * (
                    (grid[i + 1, j] - 2 * grid[i, j] + grid[i - 1, j]) / dx**2 +
                    (grid[i, j + 1] - 2 * grid[i, j] + grid[i, j - 1]) / dy**2
                )
        # Update the boundaries
        new_grid[:, 0] = 100  # left boundary
        new_grid[:, -1] = 100  # right boundary
        new_grid[0, :] = 100  # top boundary
        new_grid[-1, :] = 100  # bottom boundary
        grid = np.copy(new_grid)
    
    return grid

# Parameters
nx, ny = 50, 50  # grid size
dx, dy = 0.1, 0.1  # grid spacing
dt = 0.01  # time step
steps = 100  # number of time steps

# Initialize grid
grid = initialize_grid(nx, ny)

# Solve the heat equation
final_grid = solve_heat_equation(grid, dx, dy, dt, steps)

# Plot the final temperature distribution
plt.imshow(final_grid, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Final temperature distribution')
plt.show()