import cv2
import numpy as np

def is_rectangle(cnt, accuracy=0.05):
    """
    Determine if a contour is a rectangle.
    """
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, accuracy * peri, True)
    return len(approx) == 4

def process_frame(frame):
    """
    Process a single frame to detect rectangular boxes.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)  # Blur to reduce noise
    edged = cv2.Canny(blurred, 50, 150)  # Edge detection
    
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if is_rectangle(cnt):
            # Assuming a minimum area to avoid detecting small noise as boxes
            if cv2.contourArea(cnt) > 1000:  
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                return True  # Rectangular box detected
    return False

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if process_frame(frame):
            print("Box with lid properly closed detected.")
        else:
            print("No properly closed box detected.")
        
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = 'C:\\Users\\TheEarthG\\Downloads\\boxes.mp4'
process_video(video_path)
