# import cv2
# import torch
# import numpy as np

# # Function to load YOLOv8 model
# def load_yolov8_model(weights_path):
#     model = torch.hub.load('ultralytics/yolov8', 'yolov8', pretrained=False)
#     model.load(weights_path)
#     model.eval()
#     return model

# # Function to detect person in frame
# def detect_person(model, frame):
#     results = model(frame)
#     results.save()
#     for _, det in enumerate(results.xyxy[0]):
#         x, y, x_end, y_end = det[:4].numpy().astype(int)
#         if results.names[int(det[5].item())] == 'person':
#             return x, y, x_end, y_end
#     return None

# # Function to draw trajectory on frame
# def draw_trajectory(frame, trajectory):
#     for i in range(len(trajectory) - 1):
#         cv2.line(frame, trajectory[i], trajectory[i + 1], (0, 255, 0), 2)
#     return frame

# # Main function
# def track_person(video_path, weights_path):
#     model = load_yolov8_model(weights_path)
#     cap = cv2.VideoCapture(video_path)
#     trajectory = []
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         person = detect_person(model, frame)
#         if person is not None:
#             x, y, x_end, y_end = person
#             cv2.rectangle(frame, (x, y), (x_end, y_end), (0, 255, 0), 2)
#             center_x = (x + x_end) // 2
#             center_y = (y + y_end) // 2
#             trajectory.append((center_x, center_y))
#             if len(trajectory) > 10:
#                 trajectory.pop(0)
#             frame = draw_trajectory(frame, trajectory)
#         cv2.imshow('Object Tracking', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()

# # Stub to allow user to provide trained YOLOv8 weight and video
# if __name__ == '__main__':
#     video_path = input("Enter the video path: ")
#     weights_path = input("Enter the YOLOv8 weights path: ")
#     track_person(video_path, weights_path)

# import cv2
# import torch
# import numpy as np
# from ultralytics import YOLO

# # # Function to load YOLOv8 model
# # def load_yolov8_model(weights_path):
# #     model = YOLO(weights_path)
# #     return model

# # Function to detect person in frame
# def detect_person(model, frame):
#     results = model(frame)
#     for _, det in enumerate(results.xyxy[0]):
#         x, y, x_end, y_end = det[:4].numpy().astype(int)
#         if results.names[int(det[5].item())] == 'person':
#             return x, y, x_end, y_end
#     return None

# # Function to draw trajectory on frame
# def draw_trajectory(frame, trajectory):
#     for i in range(len(trajectory) - 1):
#         cv2.line(frame, trajectory[i], trajectory[i + 1], (0, 255, 0), 2)
#     return frame

# # Main function
# def track_person(video_path, weights_path):
#     model = load_yolov8_model(weights_path)
#     cap = cv2.VideoCapture(video_path)
#     trajectory = []
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         person = detect_person(model, frame)
#         if person is not None:
#             x, y, x_end, y_end = person
#             cv2.rectangle(frame, (x, y), (x_end, y_end), (0, 255, 0), 2)
#             center_x = (x + x_end) // 2
#             center_y = (y + y_end) // 2
#             trajectory.append((center_x, center_y))
#             if len(trajectory) > 10:
#                 trajectory.pop(0)
#             frame = draw_trajectory(frame, trajectory)
#         cv2.imshow('Object Tracking', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()

# # Stub to allow user to provide trained YOLOv8 weight and video
# if __name__ == '__main__':
#     video_path = input("Enter the video path: ")
#     weights_path = input("Enter the YOLOv8 weights path: ")
#     track_person(video_path, weights_path)

import cv2
import numpy as np
from ultralytics import YOLO

# Function to load YOLOv8 model
def load_yolov8_model(weights_path):
    model = YOLO(weights_path)
    return model

# Function to detect person in frame
def detect_person(model, frame):
    results = model(frame)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.cls[0] == 'person':
                return int(box.xmin), int(box.ymin), int(box.xmax), int(box.ymax)
    return None

# Function to draw trajectory on frame
def draw_trajectory(frame, trajectory):
    for i in range(len(trajectory) - 1):
        cv2.line(frame, trajectory[i], trajectory[i + 1], (0, 255, 0), 2)
    return frame

# Main function
def track_person(video_path, weights_path):
    model = load_yolov8_model(weights_path)
    cap = cv2.VideoCapture(video_path)
    trajectory = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite('temp.jpg', frame)
        person = detect_person(model, 'temp.jpg')
        if person is not None:
            x, y, x_end, y_end = person
            cv2.rectangle(frame, (x, y), (x_end, y_end), (0, 255, 0), 2)
            center_x = (x + x_end) // 2
            center_y = (y + y_end) // 2
            trajectory.append((center_x, center_y))
            if len(trajectory) > 10:
                trajectory.pop(0)
            frame = draw_trajectory(frame, trajectory)
        cv2.imshow('Object Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Stub to allow user to provide trained YOLOv8 weight and video
if __name__ == '__main__':
    video_path = input("Enter the video path: ")
    weights_path = input("Enter the YOLOv8 weights path: ")
    track_person(video_path, weights_path)