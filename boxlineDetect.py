import cv2  
import numpy as np  
  
class DigitalSensor:
  def __init__(self, video_path):
    self.video_path = video_path
    self.cap = cv2.VideoCapture(video_path)
    self.initial_color = None
    self.detection_count = 0
    self.color_diff_threshold = 30
    self.color_radius = 100

  def extract_center_color(self, frame):
    height, width, _ = frame.shape
    centerX, centerY = width // 2, height // 2
    area = frame[
            centerY - self.color_radius: centerY + self.color_radius,
            centerX - self.color_radius: centerX + self.color_radius,
            ]
    return np.mean(area.reshape(-1, 3), axis=0)
    
  def sense(self, initial_color, center_color, prev_detection=False):
    color_diff = np.linalg.norm(center_color - initial_color)
    cur_detection = color_diff > self.color_diff_threshold
    
    if not prev_detection and cur_detection:
      self.detection_count += 1
          
    return cur_detection  
  
  def display_indication(self, frame, detected):
    height, width, _ = frame.shape
    centerX, centerY = width // 2, height // 2
    
    cv2.putText(frame, f"Detections: {self.detection_count}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 255, 255), 1)

    dot_color = (0, 255, 0) if not detected else (0, 0, 255)
    cv2.circle(frame, (centerX, centerY), 10, dot_color, -1)

    if detected:
      cv2.putText(frame, "Box Detected", (centerX - 100, centerY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)  

  def detect_boxes(self, frame):  
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
  
    # Canny edge detection
    edged = cv2.Canny(gray, 9, 99)  
  
    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(edged, 255, 1, 1, 3, 2)
    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    # Dilation and erosion
    thresh = cv2.dilate(thresh, None, iterations=1)
    thresh = cv2.erode(thresh, None, iterations=1)  
  
    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
  
    # Draw bounding rectangles for large contours
    for cnt in contours:
      area = cv2.contourArea(cnt)
      if area > 33000:  # Adjust this threshold as needed
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)


  def run(self):
    detected = False
    initial_color = None
    while self.cap.isOpened():
      ret, frame = self.cap.read()
      if not ret:
        break
                
      center_color = self.extract_center_color(frame)  
      if initial_color is None:
        initial_color = center_color  
    
      # Check if there is any boxes in the frame
      detected = self.sense(initial_color, center_color, detected)  
    
      if detected:
        self.detect_boxes(frame)  
    
      # Display indication after finishing detection to avoid text box
      self.display_indication(frame, detected)  
    
      cv2.imshow("Frame", frame)  
      if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  
    self.cap.release()  
    cv2.destroyAllWindows()  
  
if __name__ == "__main__":  
    video_path = "C:\\Users\\TheEarthG\\Downloads\\boxes.mp4"
    sensor = DigitalSensor(video_path)  
    sensor.run()