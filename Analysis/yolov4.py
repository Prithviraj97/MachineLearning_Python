import cv2 
import numpy as np

def detect_objects(video_path, weights_path, config_path, class_names_path): 
    # Initialize the network 
    net = cv2.dnn_DetectionModel(weights_path, config_path) 
    net.setInputSize(416, 416) 
    net.setInputScale(1.0 / 255) 
    net.setInputSwapRB(True)
    # Load class labels
    try:
        with open(class_names_path, "r") as f:
            classes = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("Class names file not found.")
        return

    # Initialize counters and lists for detections
    total_human_detections = 0
    total_non_human_detections = 0
    human_detections_per_frame = []
    non_human_detections_per_frame = []

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform detection
        class_ids, confidences, boxes = net.detect(frame, confThreshold=0.5, nmsThreshold=0.4)
        
        # Initialize counters for detections in the current frame
        frame_human_detections = 0
        frame_non_human_detections = 0
        
        # Count detections and draw bounding boxes
        for (class_id, confidence, box) in zip(class_ids, confidences, boxes):
            class_name = classes[int(class_id)]
            if class_name == 'person':
                # If the detected class is 'person', draw a green bounding box
                color = (0, 255, 0)  # Green color for humans
                frame_human_detections += 1
            else:
                # For non-human objects, use a blue bounding box
                color = (255, 0, 0)  # Blue color for non-human objects
                frame_non_human_detections += 1
            
            # Draw bounding boxes around detected objects
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Update total and per-frame detections
        total_human_detections += frame_human_detections
        total_non_human_detections += frame_non_human_detections
        human_detections_per_frame.append(frame_human_detections)
        non_human_detections_per_frame.append(frame_non_human_detections)
        
        # Optionally, print the number of detections in the current frame
        print(f"Humans detected in this frame: {frame_human_detections}, Non-human objects detected in this frame: {frame_non_human_detections}")
        
        # Show the frame
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

    # Calculate and print the mean of human and non-human detections
    mean_human_detections = np.mean(human_detections_per_frame)
    mean_non_human_detections = np.mean(non_human_detections_per_frame)
    print("Mean number of humans detected per frame:", mean_human_detections)
    print("Mean number of non-human objects detected per frame:", mean_non_human_detections)

    # Print the detection summary for the entire video
    print("Total humans detected in the video:", total_human_detections)
    print("Total non-human objects detected in the video:", total_non_human_detections)
    # Display the mean as the "correct" number of detections for demonstration
    print("Correct number of humans detected (based on mean):", round(mean_human_detections))
    print("Correct number of non-human objects detected (based on mean):", round(mean_non_human_detections))

video_path = "C:\\Users\\TheEarthG\\Downloads\\human.mp4" 
weights_path = "C:\\Users\\TheEarthG\\Downloads\\yolov4.weights" 
config_path = "C:\\Users\\TheEarthG\\Downloads\\yolov4.cfg" 
class_names_path = "C:\\Users\\TheEarthG\\Downloads\\coco.names"

detect_objects(video_path, weights_path, config_path, class_names_path)
