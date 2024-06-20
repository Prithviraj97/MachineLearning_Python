# import numpy as np
# from scipy.ndimage import gaussian_filter, sobel, convolve
# from scipy.signal import convolve2d
# from PIL import Image

# class SIFTAlgorithm:
#     def __init__(self):
#         pass

#     def gaussian_blur(self, img, sigma):
#         """
#         Apply Gaussian blur to the image
#         """
#         size = int(2*(np.ceil(3*sigma))+1)
#         x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
#         g = np.exp(-(x**2 + y**2) / (2*sigma**2))
#         g /= g.sum()
#         img_blur = np.zeros_like(img)
#         for i in range(3):  
#             img_blur[:,:,  i] = convolve(img[:,:, i], g, mode='constant', cval=0)
#         return img_blur

#     # def gaussian_blur(self, img, sigma):
#     #     # """
#     #     # Apply Gaussian blur to each channel of the color image.
#     #     # """
#     #     # size = int(2*(np.ceil(3*sigma))+1)
#     #     # x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
#     #     # g = np.exp(-(x**2 + y**2) / (2*sigma**2))
#     #     # g /= g.sum()

#     #     # # Apply blur to each channel independently using vectorized operations
#     #     # img_blurred = np.dstack([convolve(channel, g, mode='constant', cval=0) for channel in img.T])
#     #     # return img_blurred.T
#     #     return gaussian_filter(img, sigma)


#     def difference_of_gaussian(self, img1, img2):
#         """
#         Compute difference of two Gaussian-blurred images.
#         """
#         return img1 - img2

#     def find_keypoints(self, dog_img, threshold=0.5):
#         """
#         Find keypoints in the Difference-of-Gaussian image
#         where pixel value is higher than the specified threshold.
#         """
#         keypoints = np.argwhere(dog_img > threshold)
#         return keypoints  # Return array of (row, col) positions

#     def generate_descriptors(self, keypoints, img):
#         """
#         Generate a simple descriptor of 8 surrounding pixels for each keypoint.
#         """
#         descriptors = []
#         for kp in keypoints:
#             x, y = kp
#             if x <= 0 or x >= img.shape[0]-1 or y <= 0 or y >= img.shape[1]-1:
#                 continue
#             # Simple descriptor: use surrounding pixel values
#             patch = img[x-1:x+2, y-1:y+2, :].flatten()
#             descriptors.append(patch)
#         return np.array(descriptors)

#     def out(self, img):
#         print(img.shape)
#         h, w, c = img.shape
#         img_gray = np.mean(img, axis=2)  # Convert to grayscale

#         # 1. Blur with two different sigmas
#         img_blur1 = self.gaussian_blur(img_gray, sigma=1)
#         img_blur2 = self.gaussian_blur(img_gray, sigma=2)

#         # 2. Difference of Gaussian
#         dog_img = self.difference_of_gaussian(img_blur1, img_blur2)

#         # 3. Find keypoints
#         keypoints = self.find_keypoints(dog_img)

#         # 4. Generate descriptors
#         descriptors = self.generate_descriptors(keypoints, img_gray)

#         return keypoints, descriptors

# #TEST THE CLASS WITH AN IMAGE
# sift = SIFTAlgorithm()
# #PIL.Image.open(image_path).convert('RGB')

# image = Image.open('11102.jpg')
# img = np.asarray(image)
# print(sift.out(img=img))

import numpy as np
from scipy.ndimage import convolve
from PIL import Image
import os
class SIFTAlgorithm:
    def __init__(self):
        pass

    # def gaussian_blur(self, img, sigma):
    #     size = int(2*(np.ceil(3*sigma))+1)
    #     x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    #     g = np.exp(-(x**2 + y**2) / (2*sigma**2))
    #     g /= g.sum()
    #     img_blur = convolve(img, g, mode='constant', cval=0)
    #     return img_blur

    # def gaussian_blur(self, img, sigma):
    #     """
    #     Applies a Gaussian blur to the input image.

    #     Args:
    #         img (numpy array): The input image.
    #         sigma (float): The standard deviation of the Gaussian blur.

    #     Returns:
    #         img_blur (numpy array): The blurred image.
    #     """
    #     # Check if the image is RGB or grayscale
    #     if len(img.shape) == 3:  # RGB image
    #         # Apply the Gaussian blur to each channel separately
    #         img_blur = np.zeros(img.shape)
    #         for i in range(3):
    #             img_blur[:, :, i] = self._gaussian_blur_channel(img[:, :, i], sigma)
    #     else:  # Grayscale image
    #         img_blur = self._gaussian_blur_channel(img, sigma)

    #     return img_blur

    # def _gaussian_blur_channel(self, img, sigma):
    #     """
    #     Helper method to apply a Gaussian blur to a single-channel image.

    #     Args:
    #         img (numpy array): The input single-channel image.
    #         sigma (float): The standard deviation of the Gaussian blur.

    #     Returns:
    #         img_blur (numpy array): The blurred single-channel image.
    #     """
    #     # Calculate the size of the Gaussian filter
    #     size = int(2*(np.ceil(3*sigma))+1)
    #     # Generate the Gaussian filter
    #     x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    #     g = np.exp(-(x**2 + y**2) / (2*sigma**2))
    #     g /= g.sum()
    #     # Apply the Gaussian filter to the image
    #     img_blur = convolve(img, g, mode='constant', cval=0)
    #     return img_blur

    def gaussian_blur(self, img, sigma):
        """
        Applies a Gaussian blur to the input image.

        Args:
            img (numpy array): The input image.
            sigma (float): The standard deviation of the Gaussian blur.

        Returns:
            img_blur (numpy array): The blurred image.
        """
        # Calculate the size of the Gaussian filter
        size = int(2*(np.ceil(3*sigma))+1)
        # Generate the Gaussian filter
        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = np.exp(-(x**2 + y**2) / (2*sigma**2))
        g /= g.sum()

        # Check if the image is grayscale or RGB
        if len(img.shape) == 3:  # RGB image
            # Apply the Gaussian filter to each channel of the RGB image
            img_blur = np.zeros(img.shape)
            for i in range(3):
                img_blur[:, :, i] = convolve(img[:, :, i], g, mode='constant', cval=0)
        else:  # Grayscale image
            # Apply the Gaussian filter to the grayscale image
            img_blur = convolve(img, g, mode='constant', cval=0)

        return img_blur

    def difference_of_gaussian(self, img1, img2):
        return img1 - img2

    def find_keypoints(self, dog_img, threshold=0.5):
        keypoints = np.argwhere(dog_img > threshold)
        return keypoints  
    def generate_descriptors(self, keypoints, img):
        descriptors = []
        for kp in keypoints:
            x, y = kp
            if x <= 0 or x >= img.shape[0]-1 or y <= 0 or y >= img.shape[1]-1:
                continue
            patch = img[x-1:x+2, y-1:y+2].flatten()
            descriptors.append(patch)
        return np.array(descriptors)

    def out(self, img):
        print(img.shape)
        h, w, c = img.shape
        img_gray = np.mean(img, axis=2)  

        img_blur1 = self.gaussian_blur(img_gray, sigma=1)
        img_blur2 = self.gaussian_blur(img_gray, sigma=2)
        dog_img = self.difference_of_gaussian(img_blur1, img_blur2)

        keypoints = self.find_keypoints(dog_img)

        descriptors = self.generate_descriptors(keypoints, img_gray)

        return keypoints, descriptors
    
    import os

    def process_directory(self, dir_path):
        """
        Processes all images in a given directory.

        Args:
            dir_path (str): The path to the directory containing the images.

        Returns:
            None
        """
        for filename in os.listdir(dir_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image = Image.open(os.path.join(dir_path, filename))
                img = np.asarray(image)
                keypoints, descriptors = self.out(img)
                print(f"Processed image: {filename}")
                print(f"Keypoints: {keypoints}")
                print(f"Descriptors: {descriptors}")
                print("----------------------------------------")

# Example usage
sift = SIFTAlgorithm()
sift.process_directory("C:\\Users\\TheEarthG\\OneDrive\\Pictures\\images")

# sift = SIFTAlgorithm()
# image = Image.open('cat.jpeg')
# img = np.asarray(image)
# print(sift.out(img=img))

# import numpy as np
# from scipy.ndimage import convolve
# from PIL import Image

# class SIFTAlgorithm:
#     def __init__(self):
#         pass

#     def gaussian_blur(self, img, sigma):
#         size = int(2*(np.ceil(3*sigma))+1)
#         x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
#         g = np.exp(-(x**2 + y**2) / (2*sigma**2))
#         g /= g.sum()
        
#         if len(img.shape) == 3:  # RGB image
#             img_blur = np.zeros_like(img)
#             for i in range(3):  
#                 img_blur[:,:, i] = convolve(img[:,:, i], g, mode='constant', cval=0)
#         elif len(img.shape) == 2:  # Grayscale image
#             img_blur = convolve(img, g, mode='constant', cval=0)
#         else:
#             raise ValueError("Invalid image dimensions")
            
#         return img_blur

#     def difference_of_gaussian(self, img1, img2):
#         return img1 - img2

#     def find_keypoints(self, dog_img, threshold=0.5):
#         keypoints = np.argwhere(dog_img > threshold)
#         return keypoints  # Return array of (row, col) positions

#     def generate_descriptors(self, keypoints, img):
#         descriptors = []
#         for kp in keypoints:
#             x, y = kp
#             if x <= 0 or x >= img.shape[0]-1 or y <= 0 or y >= img.shape[1]-1:
#                 continue
#             # Simple descriptor: use surrounding pixel values
#             if len(img.shape) == 3:  # RGB image
#                 patch = img[x-1:x+2, y-1:y+2, :].flatten()
#             elif len(img.shape) == 2:  # Grayscale image
#                 patch = img[x-1:x+2, y-1:y+2].flatten()
#             descriptors.append(patch)
#         return np.array(descriptors)

#     def out(self, img):
#         print(img.shape)
#         h, w, c = img.shape
#         img_gray = np.mean(img, axis=2)  

#         img_blur1 = self.gaussian_blur(img_gray, sigma=1)
#         img_blur2 = self.gaussian_blur(img_gray, sigma=2)
#         dog_img = self.difference_of_gaussian(img_blur1, img_blur2)

#         keypoints = self.find_keypoints(dog_img)

#         descriptors = self.generate_descriptors(keypoints, img_gray)

#         return keypoints, descriptors

# sift = SIFTAlgorithm()
# image = Image.open('11102.jpg')
# img = np.asarray(image)
# print(sift.out(img=img))