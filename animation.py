import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
LINK_LENGTH = 1.0  # Length of the rotating link

# Create an array for the frames of animation with the theta calculation built in
thetas = np.arange(0, 360, 2) * np.pi/180


def scott_russell(theta, length):
    """
    Converts a radian and length to x, y coordinates based on a length for determining
    the position of a Scott Russell mechanism.
    
    Args:
        theta (float): the radian position of the rotating link arm
        length (float): the length of the rotating link arm

    Returns:
        tuple (float, float): x and y coordinates for the end of the link arm
    """
    x = length * np.cos(theta)
    y = length * np.sin(theta)
    return x, y


# Function to update the animation
def update(frame):
    x, y = scott_russell(frame, LINK_LENGTH)
    line1[0].set_xdata([0, x])
    line1[0].set_ydata([0, y])
    line2[0].set_data([x * 2, 0], [0, y * 2])
    dot1.set_offsets((x, y))
    dot2.set_offsets(([x * 2, 0], [0, y * 2]))
    return (line1, line2, dot1, dot2)


# Create a figure and subplot
fig, ax = plt.subplots()
ax.set_title('Scott Russell Mechanism Animation')
ax.set_aspect('equal')
ax.set_xlim(LINK_LENGTH * -3, LINK_LENGTH * 3)
ax.set_ylim(LINK_LENGTH * -3, LINK_LENGTH * 3)
# horizontal and vertical axes shown to demonstrate link path
ax.axvline(c='purple')
ax.axhline(c='purple')

# create starting points and build subplot objects
start_x, start_y = scott_russell(thetas[0], LINK_LENGTH)
line1 = ax.plot([0, 0], [start_x, start_y], 'b-')
line2 = ax.plot([start_x * 2, 0], [0, start_y * 2], 'r-')
dot1 = ax.scatter(start_x, start_y, c='blue')  # Blue dot for the first link connection
dot2 = ax.scatter([start_x * 2, 0], [0, start_y * 2], c='red')  # Red dots for the second link connections

# Animate the mechanism
ani = FuncAnimation(fig, update, frames=thetas, interval=1)

plt.show()

