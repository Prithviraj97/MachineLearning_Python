# import numpy as np
# import cv2

# def gaussian_blur(image, sigma):
#     return cv2.GaussianBlur(image, (0, 0), sigma)

# def compute_gradients(image):
#     dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
#     dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
#     magnitude = np.sqrt(dx**2 + dy**2)
#     orientation = np.arctan2(dy, dx) * (180 / np.pi) % 360
#     return magnitude, orientation

# def keypoint_descriptor(magnitude, orientation, position, num_bins=8, window_width=4):
#     """Create a descriptor for a single keypoint"""
#     histogram = np.zeros(num_bins, dtype=np.float32)
#     for y in range(-window_width//2, window_width//2):
#         for x in range(-window_width//2, window_width//2):
#             px, py = position[1] + x, position[0] + y
#             if 0 <= px < magnitude.shape[1] and 0 <= py < magnitude.shape[0]:
#                 weight = magnitude[py, px]
#                 bin = int(orientation[py, px] / (360 // num_bins)) % num_bins
#                 histogram[bin] += weight
#     return histogram / np.linalg.norm(histogram)

# def out(self, img):
#     """
#     Implement Scale-Invariant Feature Transform (SIFT) algorithm
#     :param img: float/int array, shape: (height, width, channel)
#     :return sift_results (keypoints, descriptors)
#     """
#     print(img.shape)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = img.astype('float32') / 255.0  # Normalize to [0, 1]

#     # Simple approach to find keypoints (corners)
#     keypoints = cv2.goodFeaturesToTrack(img, 1000, 0.01, 10)
#     keypoints = [tuple(map(int, kp.ravel())) for kp in keypoints]

#     # Compute image gradients
#     mag, ori = compute_gradients(img)

#     # Compute descriptors at each keypoint
#     descriptors = [keypoint_descriptor(mag, ori, kp) for kp in keypoints]

#     return keypoints, descriptors
import numpy as np
class sift():
    def __init__(self):
        pass

    def gaussian_blur(self,img, sigma):
        """Applies Gaussian blur to an image."""
        size = int(2*(4*sigma + 0.5) + 1)
        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = np.exp(-((x**2 + y**2) / (2.0*sigma**2)))
        g = g / g.sum()
        return self.convolve(img, g)

    def convolve(self,img, kernel):
        """Convolves an image with a given kernel."""
        kh, kw = kernel.shape
        h, w = img.shape
        img_padded = np.pad(img, pad_width=((kh//2, kh//2), (kw//2, kw//2)), mode='constant', constant_values=0)
        output = np.zeros_like(img)
        for i in range(h):
            for j in range(w):
                output[i, j] = np.sum(img_padded[i:i+kh, j:j+kw] * kernel)
        return output

    def difference_of_gaussians(self,img, sigma1, sigma2):
        """Computes the Difference of Gaussians for an image at two scales."""
        return self.gaussian_blur(img, sigma1) - self.gaussian_blur(img, sigma2)

    def detect_keypoints(self,img):
        """Detects keypoints in an image using a basic thresholding approach."""
        sigma1, sigma2 = 1, 2
        dog = self.difference_of_gaussians(img, sigma1, sigma2)
        keypoints = np.argwhere(dog > np.percentile(dog, 95))  # Detect strong responses
        return keypoints

    def out(img):
        if img.ndim == 3 and img.shape[2] == 3:  # Assuming BGR color image
            img_gray = np.mean(img, axis=2)  # Convert to grayscale
        else:
            img_gray = img

        keypoints = detect_keypoints(img_gray)
        descriptors = np.zeros((len(keypoints), 128))  # Dummy descriptor, normally you'd compute gradients here

        sift_results = (keypoints, descriptors)
        return sift_results


from PIL import Image
#Open the image with PIL to ensure the image is loaded correctly
image = Image.open('11102.jpg') 
# img = np.asarray(image)       #convert JpegImageFile into numpy array
print(sift.out(img=image))