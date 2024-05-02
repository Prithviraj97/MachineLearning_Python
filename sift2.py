import numpy as np
from scipy.ndimage import gaussian_filter, sobel, convolve
from scipy.signal import convolve2d
from PIL import Image

class SIFTAlgorithm:
    def __init__(self):
        pass

    def gaussian_blur(self, img, sigma):
        """
        Apply Gaussian blur to the image
        """
        size = int(2*(np.ceil(3*sigma))+1)
        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = np.exp(-(x**2 + y**2) / (2*sigma**2))
        g /= g.sum()
        img_blur = np.zeros_like(img)
        for i in range(3):  # assuming img is color (height, width, channels)
            img_blur[:,:,  i] = convolve(img[:,:, i], g, mode='constant', cval=0)
        return img_blur

    # def gaussian_blur(self, img, sigma):
    #     # """
    #     # Apply Gaussian blur to each channel of the color image.
    #     # """
    #     # size = int(2*(np.ceil(3*sigma))+1)
    #     # x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    #     # g = np.exp(-(x**2 + y**2) / (2*sigma**2))
    #     # g /= g.sum()

    #     # # Apply blur to each channel independently using vectorized operations
    #     # img_blurred = np.dstack([convolve(channel, g, mode='constant', cval=0) for channel in img.T])
    #     # return img_blurred.T
    #     return gaussian_filter(img, sigma)


    def difference_of_gaussian(self, img1, img2):
        """
        Compute difference of two Gaussian-blurred images.
        """
        return img1 - img2

    def find_keypoints(self, dog_img, threshold=0.5):
        """
        Find keypoints in the Difference-of-Gaussian image
        where pixel value is higher than the specified threshold.
        """
        keypoints = np.argwhere(dog_img > threshold)
        return keypoints  # Return array of (row, col) positions

    def generate_descriptors(self, keypoints, img):
        """
        Generate a simple descriptor of 8 surrounding pixels for each keypoint.
        """
        descriptors = []
        for kp in keypoints:
            x, y = kp
            if x <= 0 or x >= img.shape[0]-1 or y <= 0 or y >= img.shape[1]-1:
                continue
            # Simple descriptor: use surrounding pixel values
            patch = img[x-1:x+2, y-1:y+2, :].flatten()
            descriptors.append(patch)
        return np.array(descriptors)

    def out(self, img):
        print(img.shape)
        h, w, c = img.shape
        img_gray = np.mean(img, axis=2)  # Convert to grayscale

        # 1. Blur with two different sigmas
        img_blur1 = self.gaussian_blur(img_gray, sigma=1)
        img_blur2 = self.gaussian_blur(img_gray, sigma=2)

        # 2. Difference of Gaussian
        dog_img = self.difference_of_gaussian(img_blur1, img_blur2)

        # 3. Find keypoints
        keypoints = self.find_keypoints(dog_img)

        # 4. Generate descriptors
        descriptors = self.generate_descriptors(keypoints, img_gray)

        return keypoints, descriptors

#TEST THE CLASS WITH AN IMAGE
sift = SIFTAlgorithm()
#PIL.Image.open(image_path).convert('RGB')

image = Image.open('11102.jpg')
img = np.asarray(image)
print(sift.out(img=img))