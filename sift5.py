# import cv2
# import numpy as np
# from PIL import Image

# class SIFTProcessor:
#     def out(self, img):
#         """
#         Implement Scale-Invariant Feature Transform(SIFT) algorithm
#         :param img: float/int array, shape: (height, width, channel)
#         :return sift_results (keypoints, descriptors)
#         """
#         # Initialize SIFT
#         sift = cv2.SIFT_create()

#         # Convert the image to grayscale if it isn't already
#         if len(img.shape) == 3 and img.shape[2] == 3:
#             gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         else:
#             gray_img = img

#         # Detect keypoints and compute descriptors
#         keypoints, descriptors = sift.detectAndCompute(gray_img, None)

#         # Return the results as a tuple
#         return keypoints, descriptors


# #TEST THE CLASS WITH AN IMAGE
# sift = SIFTProcessor()
# #Open the image with PIL to ensure the image is loaded correctly
# image = Image.open('11102.jpg')
# img = np.asarray(image)
# print(sift.out(img=img))
import numpy as np
from PIL import Image
class Keypoint:
    def __init__(self, x, y, sigma):
        self.x = x
        self.y = y
        self.sigma = sigma
        self.orientation = None  # Placeholder for orientation
sigmas = [1, 2]        
def my_sift_like(image):
    """
    Simplified SIFT-like algorithm using NumPy.

    Args:
        image: Grayscale image (assumed) as a NumPy array.

    Returns:
        keypoints (list): List of keypoint objects (simplified representation).
        descriptors (list): List of descriptor vectors (simplified representation).
    """

    # Scale space (simplified)
    sigmas = [1, 2]
    scaled_images = [gaussian_filter(image, sigma) for sigma in sigmas]

    # DoG (simplified)
    dog_images = [scaled_images[i] - scaled_images[i+1] for i in range(len(scaled_images)-1)]

    # Keypoint detection (simplified)
    keypoints = find_extrema(dog_images, threshold=0.05)  # Set a threshold

    # Orientation assignment (simplified)
    for kp in keypoints:
        kp.orientation = average_gradient_direction(image, kp.x, kp.y)

    # Descriptor generation (simplified)
    descriptors = compute_simple_descriptor(image, keypoints)

    return keypoints, descriptors


def gaussian_filter(image, sigma):
  """
  Implements a basic Gaussian filter using NumPy.
  """
  size = int(2 * (np.ceil(3 * sigma)) + 1)
  x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
  g = np.exp(-(x**2 + y**2) / (2 * sigma**2))  # Define g within the function
  g /= g.sum()
  return np.convolve(image, g, mode='same')


def find_extrema(dog_images, threshold=0.05):
    """
    Finds local maxima/minima in DoG images exceeding a threshold.

    Args:
        dog_images (list): List of DoG images.
        threshold (float, optional): Threshold for extremum detection. Defaults to 0.05.

    Returns:
        list: List of keypoint objects (simplified representation with x, y coordinates).
    """
    keypoints = []
    for i, dog_img in enumerate(dog_images):
        rows, cols = dog_img.shape
        for x in range(1, rows - 1):
            for y in range(1, cols - 1):
                center_point = dog_img[x, y]
                if abs(center_point) > threshold:  # Check for both maxima and minima
                    is_extremum = True
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        neighbor = dog_img[x + dx, y + dy]
                        if abs(center_point) < abs(neighbor):
                            is_extremum = False
                            break
                    if is_extremum:
                        keypoint = Keypoint(x, y, sigma=sigmas[i])  # Simplified keypoint object
                        keypoints.append(keypoint)
    return keypoints


def average_gradient_direction(image, x, y):
    """
    Computes a simple average gradient direction within a window around a keypoint.

    Args:
        image: Grayscale image.
        x: x-coordinate of the keypoint.
        y: y-coordinate of the keypoint.

    Returns:
        float: Average gradient direction in radians.
    """
    window_size = 3
    window = image[x - window_size // 2: x + window_size // 2 + 1,
                   y - window_size // 2: y + window_size // 2 + 1]
    sobelx = np.sobel(window, axis=1, mode='constant')
    sobely = np.sobel(window, axis=0, mode='constant')
    angles = np.arctan2(sobely, sobelx)  # Calculate gradient directions
    return np.average(angles)  # Simple average


def compute_simple_descriptor(image, keypoints):
    """
    Computes a simple descriptor based on average gradient magnitude and direction within subregions around a keypoint.

    Args:
        image: Grayscale image.
        keypoints: List of keypoint objects.

    Returns:
        list: List of descriptor vectors (simplified representation).
    """
    descriptors = []
    for kp in keypoints:
        window_size = 8  # Adjust window size as needed
        half_window = window_size // 2
        subregion_size = window_size // 4  # Adjust subregion size as needed
        half_subregion = subregion_size // 2

        # Extract window around the keypoint with orientation adjustment
        patch_center = (kp.x, kp.y)
        rotation_matrix = np.array([[np.cos(kp.orientation), -np.sin(kp.orientation)],
                                    [np.sin(kp.orientation), np.cos(kp.orientation)]])
        rotated_offsets = np.dot(rotation_matrix, np.array([[-dx, -dy] for dx in range(-half_window, half_window + 1) for dy in range(-half_window, half_window + 1)]))
        patch_coordinates = patch_center + rotated_offsets.T
        patch = image[patch_coordinates[:, 0].astype(int), patch_coordinates[:, 1].astype(int)]  # Handle potential out-of-bounds

        # Compute simple descriptor (average gradient magnitude and direction in subregions)
        descriptor = []
        for x in range(0, window_size, subregion_size):
            for y in range(0, window_size, subregion_size):
                subregion = patch[x:x+subregion_size, y:y+subregion_size]
                sobelx = np.sobel(subregion, axis=1, mode='constant')
                sobely = np.sobel(subregion, axis=0, mode='constant')
                avg_mag = np.mean(np.sqrt(sobelx**2 + sobely**2))  # Average gradient magnitude
                avg_dir = np.arctan2(np.mean(sobely), np.mean(sobelx))  # Average gradient direction (adjust for orientation)
                descriptor.extend([avg_mag, avg_dir])  # Concatenate subregion descriptors

        descriptors.append(descriptor)
    return descriptors
# Load example images
# Assuming you have loaded your grayscale image as `image`
img = Image.open('11102.jpg')
image = np.asarray(img)
keypoints, descriptors = my_sift_like(image)

print("Number of keypoints:", len(keypoints))
print("Example descriptor:", descriptors[0])  # Print the first descriptor for illustration
