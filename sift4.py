# import numpy as np
# from PIL import Image
# import scipy.ndimage

# def gaussian_kernel(size, sigma):
#     """
#     Generate a Gaussian kernel with given size and standard deviation.
#     """
#     kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-size//2)**2 + (y-size//2)**2) / (2*sigma**2)), (size, size))
#     return kernel / np.sum(kernel)

# def gaussian_blur(image, sigma):
#     """
#     Apply Gaussian blur to the image.
#     """
#     kernel_size = int(6 * sigma) // 2 * 2 + 1  # Ensure the kernel size is odd
#     kernel = gaussian_kernel(kernel_size, sigma)
#     return scipy.ndimage.convolve(image, kernel)

# def compute_gradients(image):
#     """
#     Compute gradients in x and y directions.
#     """
#     gradient_x = np.gradient(image, axis=1)
#     gradient_y = np.gradient(image, axis=0)
#     return gradient_x, gradient_y

# def detect_keypoints(image):
#     """
#     Detect keypoints using Difference of Gaussians (DoG).
#     """
#     # Blur the image at different scales
#     scales = [1.6, 1.6 * np.sqrt(2), 1.6 * 2, 1.6 * 2 * np.sqrt(2)]
#     blurred_images = [gaussian_blur(image, sigma) for sigma in scales]
    
#     # Compute the difference of Gaussians
#     dog_images = [blurred_images[i + 1] - blurred_images[i] for i in range(len(blurred_images) - 1)]
    
#     # Perform non-maximum suppression and detect keypoints
#     # (You may need to implement this step separately)
#     # Keypoints will be the local extrema in the DoG images
    
#     return dog_images

# def assign_orientations(keypoints, gradient_x, gradient_y):
#     """
#     Assign orientations to keypoints based on gradient directions.
#     """
#     # Compute gradient magnitudes and directions
#     magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
#     orientation = np.arctan2(gradient_y, gradient_x)
    
#     # Assign orientation to each keypoint based on local gradient directions
#     # (You may need to implement this step separately)
    
#     return keypoints_with_orientations

# def extract_descriptors(keypoints_with_orientations, gradient_x, gradient_y):
#     """
#     Extract descriptors for keypoints.
#     """
#     # Compute descriptors based on gradient orientations and magnitudes
#     # (You may need to implement this step separately)
    
#     return descriptors

# # Load an image (you'll need to have your own image loading code)
# image = Image.open('11102.jpg')
# image = np.asarray(image)
# # Convert the image to grayscale
# gray_image = np.dot(image[..., :3], [0.299, 0.587, 0.114])

# # Step 1: Gaussian Blurring
# blurred_image = gaussian_blur(gray_image, sigma=1.6)

# # Step 2: Compute gradients
# gradient_x, gradient_y = compute_gradients(blurred_image)

# # Step 3: Detect keypoints
# keypoints = detect_keypoints(blurred_image)

# # Step 4: Assign orientations
# keypoints_with_orientations = assign_orientations(keypoints, gradient_x, gradient_y)

# # Step 5: Extract descriptors
# descriptors = extract_descriptors(keypoints_with_orientations, gradient_x, gradient_y)

# print(keypoints, descriptors)

import cv2
import numpy as np
from PIL import Image
class SIFTProcessor:
    def __init__(self):
        # Initialize the SIFT detector
        self.sift = cv2.SIFT_create()
    
    def out(self, img):
        """
        Implement Scale-Invariant Feature Transform (SIFT) algorithm
        :param img: numpy array, shape: (height, width, channels)
        :return: tuple (keypoints, descriptors)
        """
        # Ensure the image is grayscale, as SIFT works with single-channel images
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img
        
        # Detect keypoints and compute descriptors using SIFT
        keypoints, descriptors = self.sift.detectAndCompute(gray_img, None)
        
        # Draw keypoints on the image for visualization (optional)
        img_with_keypoints = cv2.drawKeypoints(gray_img, keypoints, outImage=None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Save the image with keypoints to a file (optional, for visualization purposes)
        cv2.imwrite('image_with_keypoints.jpg', img_with_keypoints)
        
        return keypoints, descriptors

#TEST THE CLASS WITH AN IMAGE
sift = SIFTProcessor()
#Open the image with PIL to ensure the image is loaded correctly
image = Image.open('11102.jpg')
img = np.asarray(image)
print(sift.out(img=img))