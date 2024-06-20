import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage import filters
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os

class SIFT:
    def __init__(self, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
        self.sigma = sigma
        self.num_intervals = num_intervals
        self.assumed_blur = assumed_blur
        self.image_border_width = image_border_width

    def gaussian_blur(self, image, sigma):
        return gaussian_filter(image, sigma)

    def compute_derivatives(self, image):
        Ix = ndimage.sobel(image, axis=0)
        Iy = ndimage.sobel(image, axis=1)
        return Ix, Iy

    def compute_gradients(self, Ix, Iy):
        mag = np.sqrt(Ix**2 + Iy**2)
        theta = np.arctan2(Iy, Ix)
        return mag, theta

    def is_extremum(self, i, j, octave, interval, dog_images):
        val = dog_images[interval][i, j]
        max_val = max(max(dog_images[interval-1][i-1:i+2, j-1:j+2].max(), dog_images[interval][i-1:i+2, j-1:j+2].max()), dog_images[interval+1][i-1:i+2, j-1:j+2].max())
        min_val = min(min(dog_images[interval-1][i-1:i+2, j-1:j+2].min(), dog_images[interval][i-1:i+2, j-1:j+2].min()), dog_images[interval+1][i-1:i+2, j-1:j+2].min())
        return True if val > max_val or val < min_val else False

    def get_keypoints(self, image):
        keypoints = []
        sigma = self.sigma
        assumed_blur = self.assumed_blur
        num_intervals = self.num_intervals
        image_border_width = self.image_border_width

        # Preprocess the image
        image = self.gaussian_blur(image, sigma)

        # Build Gaussian pyramid
        gaussian_images = [image]
        for i in range(1, num_intervals+2):
            gaussian_images.append(self.gaussian_blur(gaussian_images[-1], sigma))

        # Build Difference of Gaussian (DoG) pyramid
        dog_images = []
        for i in range(1, num_intervals+2):
            dog_images.append(gaussian_images[i] - gaussian_images[i-1])

        # Detect keypoints in the DoG pyramid
        for octave in range(len(dog_images)):
            for interval in range(1, len(dog_images[octave])-1):
                for i in range(image_border_width, len(dog_images[octave][interval])-image_border_width):
                    for j in range(image_border_width, len(dog_images[octave][interval][i])-image_border_width):
                        if self.is_extremum(i, j, octave, interval, dog_images):
                            keypoints.append([i, j])

        return keypoints

    def compute_descriptor(self, image, keypoints):
        descriptors = []
        Ix, Iy = self.compute_derivatives(image)
        mag, theta = self.compute_gradients(Ix, Iy)
        for keypoint in keypoints:
            i, j = keypoint
            patch_mag = mag[i-8:i+8, j-8:j+8]
            patch_theta = theta[i-8:i+8, j-8:j+8]
            patch_vector = np.zeros(128)
            for k in range(16):
                for m in range(16):
                    theta_idx = int(patch_theta[k, m] / (2 * np.pi) * 8)
                    patch_vector[8*k+theta_idx] += patch_mag[k, m]
            descriptors.append(patch_vector)
        return descriptors


def main():
    sift = SIFT()
    image_dir = 'C:\\Users\\TheEarthG\\OneDrive\\Pictures\\images'
    image_files = os.listdir(image_dir)
    keypoints_dict = {}
    descriptors_dict = {}

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints = sift.get_keypoints(image)
        descriptors = sift.compute_descriptor(image, keypoints)
        keypoints_dict[image_file] = keypoints
        descriptors_dict[image_file] = descriptors

    print(keypoints_dict)
    print(descriptors_dict)


if __name__ == "__main__":
    main()