# Working


from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
import time

app = Flask(__name__)

# Load the pre-trained gender detection model
model = tf.keras.models.load_model(r"D:\Personal\HackHeritage 2024\JavaScript\Project\gender_model_best.h5")

# Define labels and confidence threshold
labels = ['Male', 'Female']
neutral_label = 'Neutral'
confidence_threshold = 0.6

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load SSD model files for face detection
ssd_prototxt = r"D:\Personal\HackHeritage 2024\JavaScript\Project\deploy.prototxt.txt"
ssd_weights = r"D:\Personal\HackHeritage 2024\JavaScript\Project\res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(ssd_prototxt, ssd_weights)

# Variables for counting and averaging
male_count = 0
female_count = 0
frame_count = 0
start_time = time.time()

def preprocess_face(face, img_size=(150, 150)):
    face = cv2.resize(face, img_size)
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    return face

def generate_frames():
    global male_count, female_count, frame_count, start_time
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
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
                
                face_preprocessed = preprocess_face(face)
                prediction = model.predict(face_preprocessed)
                predicted_gender_prob = prediction[0][0]
                
                if predicted_gender_prob > (1 - confidence_threshold):
                    gender = 'Female'
                    frame_female_count += 1
                elif predicted_gender_prob < confidence_threshold:
                    gender = 'Male'
                    frame_male_count += 1
                else:
                    gender = neutral_label
                
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, gender, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        male_count += frame_male_count
        female_count += frame_female_count
        frame_count += 1
        
        current_time = time.time()
        if current_time - start_time >= 1:
            avg_male = male_count / frame_count
            avg_female = female_count / frame_count
            # Reset counters
            male_count = 0
            female_count = 0
            frame_count = 0
            start_time = current_time
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index_c.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_averages')
def get_averages():
    global male_count, female_count, frame_count
    if frame_count > 0:
        avg_male = male_count / frame_count
        avg_female = female_count / frame_count
    else:
        avg_male = avg_female = 0
    return jsonify({'avg_male': avg_male, 'avg_female': avg_female})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)