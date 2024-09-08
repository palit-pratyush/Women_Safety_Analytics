from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO
import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from keras.layers import DepthwiseConv2D
from ultralytics import YOLO
from twilio.rest import Client
import base64
import threading
# Import show.py methods
from show import run_show

# Twilio credentials
# account_sid = give account_id
# auth_token = give auth token
client = Client(account_sid, auth_token)

# Flask and SocketIO setup
app = Flask(__name__)
socketio = SocketIO(app)

# Twilio SOS functions
def send_sos_alert(authority_number, message):
    client.messages.create(
        body=message,
        # Twilio number from = give twilio number
        to=authority_number
    )
    print(f"SMS sent to {authority_number}")

def make_sos_call(authority_number, twiml_url):
    call = client.calls.create(
        to=authority_number,
        # Twilio number from = give twilio number
        url=twiml_url  # TwiML URL containing instructions for the call
    )
    print(f"Call initiated to {authority_number}, Call SID: {call.sid}")

def start_sos_sequence(authority_number, message, twiml_url):
    alert_count = 0
    max_alerts = 15
    call_after_alerts = 2
    while alert_count < max_alerts:
        send_sos_alert(authority_number, message)
        alert_count += 1
        if alert_count == call_after_alerts:
            make_sos_call(authority_number, twiml_url)
        if alert_count < max_alerts:
            time.sleep(30)

# Load model with custom objects
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super(CustomDepthwiseConv2D, self).__init__(*args, **kwargs)

# Load models for gender, emotion, and violence detection
gender_model = load_model('gender_model_best.h5')
emotion_model = load_model('emotion_model.h5')
violence_model = load_model("violence.h5", custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}, compile=False)
pose_model = YOLO("yolov8n-pose.pt")

# Define labels and confidence threshold for gender detection
gender_labels = ['Male', 'Female']
confidence_threshold = 0.6

# Load SSD model files for face detection
ssd_prototxt = 'deploy.prototxt.txt'
ssd_weights = 'res10_300x300_ssd_iter_140000.caffemodel'
face_net = cv2.dnn.readNetFromCaffe(ssd_prototxt, ssd_weights)

# Emotion labels
emotions = ["positive", "negative", "neutral"]

# Violence detection labels
violence_labels = open("labels_violence.txt", "r").readlines()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Global counters for male, female, frames, and violence detection
male_count = 0
female_count = 0
frame_count = 0
violence_count = 0
ratio = 0.0
start_time = time.time()

# Function to preprocess face for gender detection
def preprocess_face(face, img_size=(150, 150)):
    face = cv2.resize(face, img_size)
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    return face

# Function to handle gender, emotion, and violence detection
def detect_face(frame):
    global male_count, female_count, frame_count, violence_count, start_time
    global ratio
    
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    
    frame_male_count = 0
    frame_female_count = 0
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]
            
            # Gender detection
            face_preprocessed = preprocess_face(face)
            gender_prediction = gender_model.predict(face_preprocessed)
            predicted_gender_prob = gender_prediction[0][0]
            
            if predicted_gender_prob > (1 - confidence_threshold):
                gender = 'Female'
                frame_female_count += 1
            elif predicted_gender_prob < confidence_threshold:
                gender = 'Male'
                frame_male_count += 1
            else:
                gender = 'Neutral'
            
            # Emotion detection
            roi_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype('float') / 255.0
            roi_gray = img_to_array(roi_gray)
            roi_gray = np.expand_dims(roi_gray, axis=0)
            emotion_prediction = emotion_model.predict(roi_gray)
            max_index = np.argmax(emotion_prediction[0])
            emotion = emotions[max_index]
            
            # Violence detection
            image_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
            image_array = (image_array / 127.5) - 1
            violence_prediction = violence_model.predict(image_array)
            violence_index = np.argmax(violence_prediction)
            violence_class = violence_labels[violence_index].strip()[2:]
            
            if violence_class == 'violence':
                violence_count += 1
                print(f"Class: {violence_class} | Confidence Score: {str(np.round(violence_prediction[0][violence_index] * 100))[:-2]}%, Count: {violence_count}")
            
            # Draw bounding box and labels
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, f"{gender}, {emotion}", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    male_count += frame_male_count
    female_count += frame_female_count
    frame_count += 1
    
    current_time = time.time()
    if current_time - start_time >= 1:
        avg_male = male_count / frame_count
        avg_female = female_count / frame_count
        if avg_female > 0:
            ratio = avg_male / avg_female
        else:
            ratio = 0
        male_count = 0
        female_count = 0
        frame_count = 0
        start_time = current_time
    
    return frame

# Function to handle pose detection using YOLO
def detect_pose(frame):
    results = pose_model(frame)
    plotted_frame = results[0].plot()
    return plotted_frame

# Generate frames for streaming
def generate_frames():
    global violence_count
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame = detect_face(frame)
        frame = detect_pose(frame)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if violence_count > 25:
            print("Threshold for violence crossed")
            # Trigger SOS sequence
            threading.Thread(target=start_sos_sequence, args=('', 'This is an SOS alert message', 'http://demo.twilio.com/docs/voice.xml')).start()
            violence_count = 0

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_averages')
def get_averages():
    global male_count, female_count, frame_count, ratio
    if frame_count > 0:
        avg_male = male_count / frame_count
        avg_female = female_count / frame_count
        if avg_female > 0:
            ratio = avg_male / avg_female
        else:
            ratio = 0
    else:
        avg_male = avg_female = 0
    return jsonify({'avg_male': avg_male, 'avg_female': avg_female, 'ratio': ratio})

@socketio.on('connect')
def handle_connect():
    socketio.start_background_task(generate_frames)
    # Start the clustering and mapping task
    socketio.start_background_task(run_show)

if __name__ == '__main__':
    socketio.run(app, debug=True)