# Working

from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load the pre-trained emotion detection model
model = load_model('emotion_model.h5')
emotions = ["positive", "negative", "neutral"]

# Load SSD model files for face detection
ssd_prototxt = r'D:\Personal\HackHeritage 2024\JavaScript\Emotion\deploy.prototxt.txt'
ssd_weights = r'D:\Personal\HackHeritage 2024\JavaScript\Emotion\res10_300x300_ssd_iter_140000.caffemodel'

# Load the pre-trained SSD face detector
face_net = cv2.dnn.readNetFromCaffe(ssd_prototxt, ssd_weights)

# Initialize the webcam
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Prepare frame for SSD face detection
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()
        
        # Loop through all the detected faces
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Extract the face ROI
                face = frame[startY:endY, startX:endX]
                roi_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                
                # Resize the face ROI to match the input shape of the model
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray.astype('float') / 255.0
                roi_gray = img_to_array(roi_gray)
                roi_gray = np.expand_dims(roi_gray, axis=0)
                
                # Make a prediction
                prediction = model.predict(roi_gray)
                max_index = np.argmax(prediction[0])
                emotion = emotions[max_index]
                
                # Draw rectangle around face and label with predicted emotion
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index_e.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
