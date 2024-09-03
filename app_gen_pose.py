from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
from ultralytics import YOLO
from keras.models import load_model
import numpy as np
import base64
from keras.layers import DepthwiseConv2D

app = Flask(__name__)
socketio = SocketIO(app)

# Load YOLO model for pose detection
yolo_model = YOLO("yolov8n-pose.pt")

# Custom DepthwiseConv2D layer to handle unrecognized arguments in the gender detection model
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super(CustomDepthwiseConv2D, self).__init__(*args, **kwargs)

# Load the gender detection model
gender_model = load_model("keras_Model.h5", custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}, compile=False)
class_names = open("labels.txt", "r").readlines()

# Load SSD model files for face detection
ssd_prototxt = cv2.data.haarcascades + 'deploy.prototxt.txt'
ssd_weights = cv2.data.haarcascades + 'res10_300x300_ssd_iter_140000.caffemodel'
face_net = cv2.dnn.readNetFromCaffe(ssd_prototxt, ssd_weights)

def generate_frames():
    cap = cv2.VideoCapture(0)  # Open webcam

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Perform inference on the current frame for pose detection
        pose_results = yolo_model(frame)

        # Plot the pose detection results on the frame
        frame_with_pose = pose_results[0].plot()

        # Prepare the frame for SSD face detection
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Pass the blob through the network to detect and predict faces
        face_net.setInput(blob)
        detections = face_net.forward()

        # Loop over the detections for gender detection
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = frame[startY:endY, startX:endX]

                if face.size == 0:
                    continue

                face_resized = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
                face_array = np.asarray(face_resized, dtype=np.float32).reshape(1, 224, 224, 3)
                face_array = (face_array / 127.5) - 1

                # Predict the gender of the face
                gender_prediction = gender_model.predict(face_array)
                gender_index = np.argmax(gender_prediction)
                gender_class_name = class_names[gender_index].strip()
                confidence_score = gender_prediction[0][gender_index]

                # Draw rectangle around the face and display label and confidence score
                cv2.rectangle(frame_with_pose, (startX, startY), (endX, endY), (255, 0, 0), 2)
                label = f"{gender_class_name}: {confidence_score*100:.2f}%"
                cv2.putText(frame_with_pose, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Encode the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame_with_pose)
        frame = buffer.tobytes()

        # Convert to base64 for streaming
        frame_b64 = base64.b64encode(frame).decode('utf-8')
        socketio.emit('video_frame', {'frame': frame_b64})

    cap.release()

@app.route('/')
def index():
    return render_template('index4.html')

@socketio.on('connect')
def handle_connect():
    socketio.start_background_task(generate_frames)

if __name__ == '__main__':
    socketio.run(app, debug=True)
