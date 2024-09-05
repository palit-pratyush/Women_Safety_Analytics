from flask import Flask, render_template, Response
from keras.models import load_model
import cv2
import numpy as np
from keras.layers import DepthwiseConv2D

# Custom DepthwiseConv2D layer to ignore unrecognized arguments
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super(CustomDepthwiseConv2D, self).__init__(*args, **kwargs)

# Initialize the Flask application
app = Flask(__name__)

# Load the gender detection model
model = load_model("gender.h5", custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}, compile=False)

# Load the labels
class_names = open("labels_gender.txt", "r").readlines()

# Initialize webcam
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        # Capture frame-by-frame from the webcam
        success, frame = camera.read()
        if not success:
            break
        else:
            # Resize the image to the required size
            image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            
            # Preprocess the image for the model
            image_array = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
            image_array = (image_array / 127.5) - 1
            
            # Predict the gender
            prediction = model.predict(image_array)
            index = np.argmax(prediction)
            class_name = class_names[index].strip()
            confidence_score = prediction[0][index]
            
            # Display prediction on the frame
            text = f"{class_name}: {confidence_score*100:.2f}%"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Use yield to stream the video
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
