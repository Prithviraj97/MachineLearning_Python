# import cv2
# import torch
# import numpy as np

# # Load YOLOv5s model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# # Load video
# video_path = 'C:\\Users\\TheEarthG\\Downloads\\human.mp4'
# cap = cv2.VideoCapture(video_path)

# while cap.isOpened():
#     ret, frame = cap.read()
    
#     if not ret:
#         break
    
#     # Detect people in the frame
#     results = model(frame)
    
#     # Get the bounding boxes of the detected people
#     for i, det in enumerate(results.xyxy):
#         for *xyxy, conf, cls in det:
#             x1, y1, x2, y2 = map(int, xyxy)
            
#             # Check if the detected object is a person
#             if int(cls) == 0:  # 0 is the class index for person in YOLOv5s
#                 # Crop the person from the frame
#                 person = frame[y1:y2, x1:x2]
                
#                 # Apply GrabCut to separate the person from the background
#                 mask = np.zeros(person.shape[:2], np.uint8)
#                 bgdModel = np.zeros((1, 65), np.float64)
#                 fgdModel = np.zeros((1, 65), np.float64)
#                 cv2.grabCut(person, mask, (0, 0, x2-x1, y2-y1), bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
                
#                 # Create a binary mask to select the foreground
#                 binary_mask = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
                
#                 # Apply Gaussian blur to the person
#                 blurred_person = cv2.GaussianBlur(person, (23, 23), 30)
                
#                 # Use the binary mask to combine the blurred person with the original person
#                 person = np.where(binary_mask[:, :, np.newaxis] == 255, blurred_person, person)
                
#                 # Replace the person in the frame with the blurred person
#                 frame[y1:y2, x1:x2] = person
                
#     # Display the frame with blurred people
#     cv2.imshow('Blurred People', frame)
    
#     # Exit on press 'q' key
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

import cv2
import torch
import os

# Load YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load video
video_path = 'C:\\Users\\TheEarthG\\Downloads\\human.mp4'
cap = cv2.VideoCapture(video_path)

# Create a directory to save the frames
frames_dir = 'frames'
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Detect people in the frame
    results = model(frame)
    
    # Get the bounding boxes of the detected people
    for i, det in enumerate(results.xyxy):
        for *xyxy, conf, cls in det:
            x1, y1, x2, y2 = map(int, xyxy)
            
            # Check if the detected object is a person
            if int(cls) == 0:  # 0 is the class index for person in YOLOv5s
                # Crop the person from the frame
                person = frame[y1:y2, x1:x2]
                
                # Apply Gaussian blur to the person
                blurred_person = cv2.GaussianBlur(person, (23, 23), 30)
                
                # Replace the person in the frame with the blurred person
                frame[y1:y2, x1:x2] = blurred_person
                
    # Save the frame
    cv2.imwrite(os.path.join(frames_dir, f'frame_{frame_count:06d}.jpg'), frame)
    frame_count += 1
    
    # Display the frame with blurred people
    cv2.imshow('Blurred People', frame)
    
    # Exit on press 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()