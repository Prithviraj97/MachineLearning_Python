import cv2
import numpy as np

def detect_people_with_hog(video_path):
    # Initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

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

        # Detect people in the frame
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32), scale=1.05)

        # Draw bounding boxes around detected people
        frame_detections = 0
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            frame_detections += 1

        # Update the detections count
        total_detections += frame_detections
        detections_per_frame.append(frame_detections)

        # Optionally, show the number of detections in the current frame
        print(f"Detected {frame_detections} people in this frame.")

        # Show the frame
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

    # Calculate and print statistics
    mean_detections = np.mean(detections_per_frame)
    print("Mean number of people detected per frame:", mean_detections)
    print("Total people detected in the video:", total_detections)
    print("Correct number of people detected (based on mean):", round(mean_detections))

# Input path
video_path = "C:/Users/joych/Downloads/nn.mp4"

# Run the detection
detect_people_with_hog(video_path)
