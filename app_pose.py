from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
from ultralytics import YOLO
import base64
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app)

# Load YOLO model
model = YOLO("yolov8n-pose.pt")

def generate_frames():
    cap = cv2.VideoCapture(0)  # Open webcam

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Perform inference on the current frame
        results = model(frame)

        # Plot the results on the frame
        plotted_frame = results[0].plot()

        # Encode the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', plotted_frame)
        frame = buffer.tobytes()

        # Convert to base64 for streaming
        frame_b64 = base64.b64encode(frame).decode('utf-8')
        socketio.emit('video_frame', {'frame': frame_b64})

    cap.release()

@app.route('/')
def index():
    return render_template('index3.html')

@socketio.on('connect')
def handle_connect():
    socketio.start_background_task(generate_frames)

if __name__ == '__main__':
    socketio.run(app, debug=True)
