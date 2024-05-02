import numpy as np
from scipy.ndimage import gaussian_filter, sobel
from PIL import Image

#Creat a SIFT algorithm class that will contain all the functions for sift features extraction.
class SIFTAlgorithm:
    def __init__(self):
        pass

    # Apply Gaussian Blur to the image. 
    def gaussian_blur(self, img, sigma):
        return gaussian_filter(img, sigma)

    #soble filter to find strong edges/corners in the image.
    def sobel_filters(self, img):
        gx = sobel(img, axis=0, mode='constant') #gradient in x-direction 
        gy = sobel(img, axis=1, mode='constant') #gradient in y-direction
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        orientation = np.arctan2(gy, gx) * (180 / np.pi)
        return magnitude, orientation

    def detect_keypoints(self, img):
        # Full SIFT implementation would involve finding scale-space extrema.
        keypoints = []
        step_size = 10  # Arbitrary step size for demonstration
        for y in range(0, img.shape[0], step_size):
            for x in range(0, img.shape[1], step_size):
                keypoints.append((x, y))
        return keypoints

    def describe_keypoints(self, img, keypoints):
        # function to describe keypoints in the image using the given filter bank.
        descriptors = []
        for kp in keypoints:
            desc = np.zeros(128)  # SIFT descriptors are 128-dimensional
            descriptors.append(desc)
        return descriptors

    def out(self, img):
        """
        Implement Scale-Invariant Feature Transform(SIFT) algorithm
        :param img: float/int array, shape: (height, width, channel)
        :return sift_results (keypoints, descriptors)
        """
        if img.ndim == 3 and img.shape[2] == 3:
            # Convert color image to grayscale
            img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

        # Step 1: Detect keypoints
        keypoints = self.detect_keypoints(img)

        # Step 2: Compute descriptors
        descriptors = self.describe_keypoints(img, keypoints)

        return keypoints, descriptors

#TEST THE CLASS WITH AN IMAGE
sift = SIFTAlgorithm()
#Open the image with PIL to ensure the image is loaded correctly
image = Image.open('11102.jpg') 
img = np.asarray(image)       #convert JpegImageFile into numpy array
print(sift.out(img=img))