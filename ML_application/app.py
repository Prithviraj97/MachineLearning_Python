from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np

# app = Flask(__name__)

# # Define your object detection function using OpenCV
# def detect_objects(image_data):
#     # Convert base64 image data to OpenCV format
#     nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     # Your object detection code using OpenCV
#     # Example: Detect faces using OpenCV's Haar cascade classifier
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # Format results
#     results = []
#     for (x, y, w, h) in faces:
#         results.append({'x': x, 'y': y, 'width': w, 'height': h})
    
#     return results

# @app.route('/detect', methods=['POST'])
# def detect():
#     image_data = request.form['imageData']
#     results = detect_objects(image_data)
#     return jsonify(results)

# if __name__ == '__main__':
#     app.run(debug=True)

app = Flask(__name__)

# Define your object detection function using OpenCV
def detect_objects(image_data):
    # Implement your object detection logic here
    # Convert base64 image data to OpenCV format
    nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Your object detection code using OpenCV
    # Example: Detect faces using OpenCV's Haar cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Format results
    results = []
    for (x, y, w, h) in faces:
        results.append({'x': x, 'y': y, 'width': w, 'height': h})
    
    return results
    pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    image_data = request.form['imageData']
    results = detect_objects(image_data)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)

