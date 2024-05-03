import cv2
import numpy as np

def detect_objects_with_mobilenet(video_path, config_path, weights_path, class_names_path):
    # Load the neural network
    net = cv2.dnn_DetectionModel(weights_path, config_path)
    net.setInputSize(320, 320)  # Standard input size for MobileNet
    net.setInputScale(1.0 / 127.5)  # [0,255] -> [-1,1]
    net.setInputMean((127.5, 127.5, 127.5))  # Mobilenet requires mean subtraction
    net.setInputSwapRB(True)

    # Load class labels
    try:
        with open(class_names_path, "r") as f:
            classes = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("Class names file not found.")
        return

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    total_detections = 0
    detections_per_frame = []

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        class_ids, confidences, boxes = net.detect(frame, confThreshold=0.5, nmsThreshold=0.4)

        # Update total and per-frame detections
        total_detections += len(boxes)
        detections_per_frame.append(len(boxes))

        # Draw bounding boxes
        for class_id, confidence, box in zip(class_ids, confidences, boxes):
            color = (23, 230, 210)  # A distinct color for detected boxes
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{classes[class_id]}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show the frame
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

    # Calculate and print statistics
    mean_detections = np.mean(detections_per_frame)
    print("Mean number of detections per frame:", mean_detections)
    print("Total detections in the video:", total_detections)

# Input paths
video_path = "C:\\Users\\TheEarthG\\Downloads\\human.mp4" 
weights_path = "C:\\Users\\TheEarthG\\Downloads\\yolov4.weights" 
config_path = "C:\\Users\\TheEarthG\\Downloads\\yolov4.cfg" 
class_names_path = "C:\\Users\\TheEarthG\\Downloads\\coco.names"

# Run the detection
detect_objects_with_mobilenet(video_path, config_path, weights_path, class_names_path)
