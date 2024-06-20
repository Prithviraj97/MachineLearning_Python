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
#     keypoints, _ = sift.detectAndCompute(image, None)
#     return keypoints


# def compute_descriptor(image, keypoints):
#     """
#     Compute descriptors for the given keypoints.

#     Args:
#     image (numpy array): Input image.
#     keypoints (list): List of keypoints detected in the image.

#     Returns:
#     descriptors (numpy array): Descriptors for the keypoints.
#     """
#     sift = cv2.SIFT_create()
#     _, descriptors = sift.compute(image, keypoints)
#     return descriptors


# def main():
#     image_path = '11102.jpg'
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     keypoints = detect_keypoints(image)
#     descriptors = compute_descriptor(image, keypoints)
#     print("Keypoints:", keypoints)
#     print("Descriptors:", descriptors)


# if __name__ == "__main__":
#     main()

import cv2
import numpy as np

def detect_keypoints(image):
    """
    Detect keypoints in the given image using the SIFT algorithm.

    Args:
    image (numpy array): Input image.

    Returns:
    keypoints (list): List of keypoints detected in the image.
    """
    sift = cv2.SIFT_create()
    keypoints = sift.detect(image)
    return keypoints


def compute_descriptor(image, keypoints):
    """
    Compute descriptors for the given keypoints using the SIFT algorithm.

    Args:
    image (numpy array): Input image.
    keypoints (list): List of keypoints detected in the image.

    Returns:
    descriptors (numpy array): Descriptors for the keypoints.
    """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.compute(image, keypoints)
    return descriptors


def main():
    image_path = 'image.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    keypoints = detect_keypoints(image)
    descriptors = compute_descriptor(image, keypoints)
    print("Keypoints:", keypoints)
    print("Descriptors:", descriptors)


if __name__ == "__main__":
    main()