# import cv2
# import torch
# import time
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
#     start = time.time()
#     # Get the bounding boxes of the detected people
#     for i, det in enumerate(results.xyxy):
#         for *xyxy, conf, cls in det:
#             x1, y1, x2, y2 = map(int, xyxy)
            
#             # Check if the detected object is a person
#             if int(cls) == 0:  # 0 is the class index for person in YOLOv5s
#                 # Crop the person from the frame
#                 person = frame[y1:y2, x1:x2]
                
#                 # Apply Gaussian blur to the person
#                 blurred_person = cv2.GaussianBlur(person, (23, 23), 30)
                
#                 # Replace the person in the frame with the blurred person
#                 frame[y1:y2, x1:x2] = blurred_person
#     end = time.time()
#     print("The time of execution of above program is :",
#       (end-start) * 10**3, "ms")            
#     # Display the frame with blurred people
#     cv2.imshow('Blurred People', frame)
    
#     # Exit on press 'q' key
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()


# # Import necessary modules
# import cv2
# import torch
# import numpy as np
# # Import time module
# import time
 
# # Load YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# # Open video capture
# cap = cv2.VideoCapture('C:\\Users\\TheEarthG\\Downloads\\human.mp4')

# while True:
#     # Read frame
#     ret, frame = cap.read()
    
#     if not ret:
#         break

#     # Get frame dimensions
#     (H, W) = frame.shape[:2]

#     # record start time
#     start = time.time()
#     # Perform person detection
#     results = model(frame)

#     # Get person bounding boxes
#     persons = results.pandas().xyxy[0]
#     persons = persons[persons['name'] == 'person']

#     # Create a copy of the original frame
#     blurred_frame = frame.copy()

#     # Apply blur to the entire frame
#     blurred_frame = cv2.GaussianBlur(blurred_frame, (31, 31), 11)

#     # Create a mask for the persons
#     mask = np.zeros((H, W, 3), dtype=np.uint8)
#     for index, row in persons.iterrows():
#         x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
#         cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)

#     # Apply the blurred frame to the original frame where the persons are
#     blurred_frame = np.where(mask == (255, 255, 255), blurred_frame, frame)
#     end = time.time()
#     print("The time of execution of above program is :",
#       (end-start) * 10**3, "ms")
#     # Display the frame
#     cv2.imshow('Blurred Video', blurred_frame)

#     # Check for the 'q' key to exit the loop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()

import cv2
import torch
import numpy as np

# Load YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load video
video_path = 'C:\\Users\\TheEarthG\\Downloads\\human.mp4'
cap = cv2.VideoCapture(video_path)

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
                
                # Create a mask for the person
                mask = np.zeros(frame.shape, dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255
                
                # Apply Gaussian blur to the person
                blurred_person = cv2.GaussianBlur(person, (23, 23), 30)
                
                # Replace the person in the frame with the blurred person
                frame[y1:y2, x1:x2] = blurred_person
                
                # Inpaint the background
                frame = cv2.inpaint(frame, cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), 3, cv2.INPAINT_TELEA)
                
    # Display the frame with blurred people
    cv2.imshow('Blurred People', frame)
    
    # Exit on press 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()