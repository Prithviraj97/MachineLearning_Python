# # import cv2
# # import numpy as np

# # def detect_keypoints(image):
# #     """
# #     Detect keypoints in the given image using the SIFT algorithm.

# #     Args:
# #     image (numpy array): Input image.

# #     Returns:
# #     keypoints (list): List of keypoints detected in the image.
# #     """
# #     sift = cv2.SIFT_create()
# #     keypoints, _ = sift.detectAndCompute(image, None)
# #     return keypoints


# # def compute_descriptor(image, keypoints):
# #     """
# #     Compute descriptors for the given keypoints.

# #     Args:
# #     image (numpy array): Input image.
# #     keypoints (list): List of keypoints detected in the image.

# #     Returns:
# #     descriptors (numpy array): Descriptors for the keypoints.
# #     """
# #     sift = cv2.SIFT_create()
# #     _, descriptors = sift.compute(image, keypoints)
# #     return descriptors


# # def main():
# #     image_path = '11102.jpg'
# #     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# #     keypoints = detect_keypoints(image)
# #     descriptors = compute_descriptor(image, keypoints)
# #     print("Keypoints:", keypoints)
# #     print("Descriptors:", descriptors)


# # if __name__ == "__main__":
# #     main()

# import cv2
# import numpy as np

# def detect_keypoints(image):
#     """
#     Detect keypoints in the given image using the SIFT algorithm.

#     Args:
#     image (numpy array): Input image.

#     Returns:
#     keypoints (list): List of keypoints detected in the image.
#     """
#     sift = cv2.SIFT_create()
#     keypoints = sift.detect(image)
#     return keypoints


# def compute_descriptor(image, keypoints):
#     """
#     Compute descriptors for the given keypoints using the SIFT algorithm.

#     Args:
#     image (numpy array): Input image.
#     keypoints (list): List of keypoints detected in the image.

#     Returns:
#     descriptors (numpy array): Descriptors for the keypoints.
#     """
#     sift = cv2.SIFT_create()
#     keypoints, descriptors = sift.compute(image, keypoints)
#     return descriptors


# def main():
#     image_path = 'image.jpg'
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     keypoints = detect_keypoints(image)
#     descriptors = compute_descriptor(image, keypoints)
#     print("Keypoints:", keypoints)
#     print("Descriptors:", descriptors)


# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
from scipy import ndimage
import os

def detect_keypoints(image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
    image = image.astype('float32')
    base_image = ndimage.gaussian_filter(image, sigma)
    num_octaves = int(np.log2(min(image.shape)) - 1)
    keypoints = []

    for octave in range(num_octaves):
        octave_image = base_image
        for interval in range(num_intervals + 3):
            sigma = 2 ** (octave + (interval + 1) / num_intervals)
            image = ndimage.gaussian_filter(octave_image, sigma)
            if interval >= 1 and interval <= num_intervals:
                prev_image = ndimage.gaussian_filter(octave_image, 2 ** (octave + (interval) / num_intervals))
                dog_image = image - prev_image
                keypoints.extend(find_extrema(dog_image, octave, interval, image_border_width))
            octave_image = image

        base_image = ndimage.zoom(base_image, 0.5)

    return keypoints

def find_extrema(image, octave, interval, image_border_width):
    threshold = 0.04
    keypoints = []
    for i in range(image_border_width, image.shape[0] - image_border_width):
        for j in range(image_border_width, image.shape[1] - image_border_width):
            if is_extremum(image, i, j, threshold):
                keypoints.append((i, j, octave, interval))

    return keypoints

def is_extremum(image, i, j, threshold):
    """
    Check if the given point is an extremum.
    """
    val = image[i, j]
    max_val = max(max(image[i-1:i+2, j-1:j+2].flatten()), max(image[i-1:i+2, j].flatten()), max(image[i, j-1:j+2].flatten()))
    min_val = min(min(image[i-1:i+2, j-1:j+2].flatten()), min(image[i-1:i+2, j].flatten()), min(image[i, j-1:j+2].flatten()))

    if val > 0 and val >= threshold * max_val:
        return True
    elif val < 0 and val <= threshold * min_val:
        return True
    return False

def compute_descriptor(image, keypoints, num_bins=8):
    """
    Compute descriptors for the given keypoints.
    """
    descriptors = []
    for keypoint in keypoints:
        x, y, octave, interval = keypoint
        patch = get_patch(image, x, y)
        histogram = np.zeros((num_bins, num_bins))
        for i in range(patch.shape[0]):
            for j in range(patch.shape[1]):
                magnitude = np.sqrt(patch[i, j, 0]**2 + patch[i, j, 1]**2)
                direction = np.arctan2(patch[i, j, 1], patch[i, j, 0])
                bin_x = int(np.floor((direction + np.pi) / (2 * np.pi / num_bins)))
                bin_y = int(np.floor(magnitude / (255 / num_bins)))
                # Ensure bin indices are within valid range
                bin_x = max(0, min(bin_x, num_bins - 1))
                bin_y = max(0, min(bin_y, num_bins - 1))
                histogram[bin_x, bin_y] += 1
        descriptors.append(histogram.flatten())

    return descriptors

def get_patch(image, x, y):
    """
    Get a patch of the given image.
    """
    patch_size = 16
    patch = np.zeros((patch_size, patch_size, 2))
    for i in range(patch_size):
        for j in range(patch_size):
            x_idx = x + i - patch_size // 2
            y_idx = y + j - patch_size // 2
            if x_idx < 0 or x_idx >= image.shape[0] or y_idx < 0 or y_idx >= image.shape[1]:
                patch[i, j, 0] = 0
                patch[i, j, 1] = 0
            else:
                patch[i, j, 0] = image[x_idx, y_idx]
                if x_idx + 1 < image.shape[0]:
                    patch[i, j, 1] = image[x_idx + 1, y_idx] - image[x_idx, y_idx]
                else:
                    patch[i, j, 1] = 0
    return patch

def sift_algorithm(image_path):
    """
    Run the SIFT algorithm on the given image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    keypoints = detect_keypoints(image)
    descriptors = compute_descriptor(image, keypoints)
    return keypoints, descriptors

def main():
    image_dir = 'C:\\Users\\TheEarthG\\OneDrive\\Pictures\\images'
    keypoints_dict = {}
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_dir, filename)
            keypoints, descriptors = sift_algorithm(image_path)
            keypoints_dict[filename] = keypoints
    np.save('keypoints.npy', keypoints_dict)


if __name__ == "__main__":
    main()