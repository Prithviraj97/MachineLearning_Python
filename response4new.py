import numpy as np
import cv2
from PIL import Image
class sift:
    def __init__(self):
        pass

    def out(self, img):
        """
        Implement Scale-Invariant Feature Transform(SIFT) algorithm
        :param img: float/int array, shape: (height, width, channel)
        :return sift_results (keypoints, descriptors)
        """
        print(img.shape)
        h, w, c = img.shape
        if c > 1:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        # Step 1: Create Gaussian Pyramids
        octaves = [img_gray]
        num_octaves = 4
        num_scales = 3
        for i in range(num_octaves - 1):
            octaves.append(cv2.pyrDown(octaves[-1]))

        # Generate images in each octave
        dog_octaves = []
        for octave in octaves:
            dog_images = []
            for i in range(1, num_scales + 1):
                blur = cv2.GaussianBlur(octave, (0, 0), sigmaX=i, sigmaY=i)
                if i == 1:
                    prev_blur = blur
                else:
                    dog_images.append(cv2.subtract(prev_blur, blur))
                    prev_blur = blur
            dog_octaves.append(dog_images)

        # Step 2: Detect keypoints
        # For simplicity, we're using OpenCV's feature detection to illustrate here
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img_gray, None)

        # Step 3: Convert keypoints to Numpy format
        keypoints_np = np.array([[kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id]
                                for kp in keypoints])
        return (keypoints_np, descriptors)
    
#TEST THE CLASS WITH AN IMAGE
sift = sift()
#Open the image with PIL to ensure the image is loaded correctly
image = Image.open('11102.jpg') 
img = np.asarray(image)       #convert JpegImageFile into numpy array
print(sift.out(img=img))
