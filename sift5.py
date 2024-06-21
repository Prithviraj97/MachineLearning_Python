# import numpy as np
# from PIL import Image
# from scipy import signal

# class Keypoint:
#     def __init__(self, x, y, sigma):
#         self.x = x
#         self.y = y
#         self.sigma = sigma
#         self.orientation = None  

# sigmas = [1, 2]        
# def my_sift_like(image):

#     image = np.mean(image, axis=2)  # Convert to grayscale
#     sigmas = [1, 2]
#     scaled_images = [gaussian_filter(image, sigma) for sigma in sigmas]

#     dog_images = [scaled_images[i] - scaled_images[i+1] for i in range(len(scaled_images)-1)]
#     keypoints = find_extrema(dog_images, threshold=0.05)  

#     for kp in keypoints:
#         kp.orientation = average_gradient_direction(image, kp.x, kp.y)

#     descriptors = compute_simple_descriptor(image, keypoints)

#     return keypoints, descriptors


# def gaussian_filter(image, sigma):
#   size = int(2 * (np.ceil(3 * sigma)) + 1)
#   x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
#   g = np.exp(-(x**2 + y**2) / (2 * sigma**2))  
#   g /= g.sum()
#   return signal.convolve2d(image, g, mode='same')


# def find_extrema(dog_images, threshold=0.05):
    
#     keypoints = []
#     for i, dog_img in enumerate(dog_images):
#         rows, cols = dog_img.shape
#         for x in range(1, rows - 1):
#             for y in range(1, cols - 1):
#                 center_point = dog_img[x, y]
#                 if abs(center_point) > threshold:  
#                     is_extremum = True
#                     for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
#                         neighbor = dog_img[x + dx, y + dy]
#                         if abs(center_point) < abs(neighbor):
#                             is_extremum = False
#                             break
#                     if is_extremum:
#                         keypoint = Keypoint(x, y, sigma=sigmas[i])  
#                         keypoints.append(keypoint)
#     return keypoints


# def average_gradient_direction(image, x, y):
    
#     window_size = 3
#     window = image[x - window_size // 2: x + window_size // 2 + 1,
#                    y - window_size // 2: y + window_size // 2 + 1]
#     sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#     sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
#     grad_x = signal.convolve2d(window, sobelx, mode='same')
#     grad_y = signal.convolve2d(window, sobely, mode='same')
#     angles = np.arctan2(grad_y, grad_x) 
#     return np.mean(angles)  


# def compute_simple_descriptor(image, keypoints):
    
#     descriptors = []
#     for kp in keypoints:
#         window_size = 8  
#         half_window = window_size // 2
#         subregion_size = window_size // 4 
#         half_subregion = subregion_size // 2

#         patch = image[kp.x - half_window: kp.x + half_window, kp.y - half_window: kp.y + half_window]

#         descriptor = []
#         for x in range(0, window_size, subregion_size):
#             for y in range(0, window_size, subregion_size):
#                 subregion = patch[x:x+subregion_size, y:y+subregion_size]
#                 sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#                 sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
#                 grad_x = signal.convolve2d(subregion, sobelx, mode='same')
#                 grad_y = signal.convolve2d(subregion, sobely, mode='same')
#                 avg_mag = np.mean(np.sqrt(grad_x**2 + grad_y**2))  
#                 avg_dir = np.arctan2(np.mean(grad_y), np.mean(grad_x)) 
#                 descriptor.extend([avg_mag, avg_dir]) 

#         descriptors.append(descriptor)
#     return descriptors

# img = Image.open('11102.jpg')
# image = np.asarray(img)
# keypoints, descriptors = my_sift_like(image)

# print("Number of keypoints:", len(keypoints))
# print("Example descriptor:", descriptors[0])  

import numpy as np
from PIL import Image
from scipy import signal

class Keypoint:
    def __init__(self, x, y, sigma):
        self.x = x
        self.y = y
        self.sigma = sigma
        self.orientation = None  

sigmas = [1, 2]        
def my_sift_like(image):

    sigmas = [1, 2]
    scaled_images = [gaussian_filter(image, sigma) for sigma in sigmas]

    dog_images = [scaled_images[i] - scaled_images[i+1] for i in range(len(scaled_images)-1)]
    keypoints = find_extrema(dog_images, threshold=0.05)  

    for kp in keypoints:
        kp.orientation = average_gradient_direction(image, kp.x, kp.y)

    descriptors = compute_simple_descriptor(image, keypoints)

    return keypoints, descriptors


def gaussian_filter(image, sigma):
    size = int(2 * (np.ceil(3 * sigma)) + 1)
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))  
    g /= g.sum()
    return signal.convolve2d(image, g, mode='same')


def find_extrema(dog_images, threshold=0.05):
    
    keypoints = []
    for i, dog_img in enumerate(dog_images):
        rows, cols = dog_img.shape
        for x in range(1, rows - 1):
            for y in range(1, cols - 1):
                center_point = dog_img[x, y]
                if abs(center_point) > threshold:  
                    is_extremum = True
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        neighbor = dog_img[x + dx, y + dy]
                        if abs(center_point) < abs(neighbor):
                            is_extremum = False
                            break
                    if is_extremum:
                        keypoint = Keypoint(x, y, sigma=sigmas[i])  
                        keypoints.append(keypoint)
    return keypoints


def average_gradient_direction(image, x, y):
    
    window_size = 3
    window = image[x - window_size // 2: x + window_size // 2 + 1,
                   y - window_size // 2: y + window_size // 2 + 1]
    sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    grad_x = signal.convolve2d(window, sobelx, mode='same')
    grad_y = signal.convolve2d(window, sobely, mode='same')
    angles = np.arctan2(grad_y, grad_x) 
    return np.average(angles)  

def compute_simple_descriptor(image, keypoints):
    
    descriptors = []
    for kp in keypoints:
        window_size = 8  
        half_window = window_size // 2
        subregion_size = window_size // 4 
        half_subregion = subregion_size // 2

        patch = image[kp.x - half_window: kp.x + half_window, kp.y - half_window: kp.y + half_window]

        descriptor = []
        for x in range(0, window_size, subregion_size):
            for y in range(0, window_size, subregion_size):
                subregion = patch[x:x+subregion_size, y:y+subregion_size]
                sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
                grad_x = signal.convolve2d(subregion, sobelx, mode='same')
                grad_y = signal.convolve2d(subregion, sobely, mode='same')
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                grad_dir = np.arctan2(grad_y, grad_x)
                avg_mag = np.mean(grad_mag)  
                avg_dir = np.mean(grad_dir) 
                descriptor.extend([avg_mag, avg_dir]) 

        descriptors.append(descriptor)
    return descriptors
# def compute_simple_descriptor(image, keypoints):
    
#     descriptors = []
#     for kp in keypoints:
#         window_size = 8  
#         half_window = window_size // 2
#         subregion_size = window_size // 4 
#         half_subregion = subregion_size // 2

#         patch = image[kp.x - half_window: kp.x + half_window, kp.y - half_window: kp.y + half_window]

#         descriptor = []
#         for x in range(0, window_size, subregion_size):
#             for y in range(0, window_size, subregion_size):
#                 subregion = patch[x:x+subregion_size, y:y+subregion_size]
#                 sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#                 sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
#                 grad_x = signal.convolve2d(subregion, sobelx, mode='same')
#                 grad_y = signal.convolve2d(subregion, sobely, mode='same')
#                 avg_mag = np.mean(np.sqrt(grad_x**2 + grad_y**2))  
#                 avg_dir = np.arctan2(np.mean(grad_y), np.mean(grad_x)) 
#                 descriptor.extend([avg_mag, avg_dir]) 

#         descriptors.append(descriptor)
#     return descriptors


img = Image.open('11102.jpg')
image = np.asarray(img.convert('L'))  # Convert to grayscale
keypoints, descriptors = my_sift_like(image)

print("Number of keypoints:", len(keypoints))
print("Example descriptor:", descriptors[0])  