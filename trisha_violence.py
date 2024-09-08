# Working


from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import load_model
from keras.layers import DepthwiseConv2D

# Custom DepthwiseConv2D layer to ignore unrecognized arguments
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super(CustomDepthwiseConv2D, self).__init__(*args, **kwargs)

# Load the violence detection model
model = load_model("violence.h5", custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}, compile=False)

class_names = open("labels_violence.txt", "r").readlines()

app = Flask(__name__)

# Global counter for violence detections
global a
a = 0

# Camera setup
camera = cv2.VideoCapture(0)

def generate_frames():
    global a
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()
        if not success:
            break

        # Resize the frame for the model
        image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        image_array = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image_array = (image_array / 127.5) - 1

        # Predict using the model
        prediction = model.predict(image_array)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        name = class_name[2:].strip()

        if name == 'violence':
            a += 1
            print(f"Class: {class_name[2:]} | Confidence Score: {str(np.round(prediction[0][index] * 100))[:-2]}%")

        # Draw predictions on the frame
        cv2.putText(frame, f"{name}: {str(np.round(prediction[0][index] * 100))[:-2]}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Break the loop if the threshold for violence is crossed
        if a > 30:
            print("Threshold for violence crossed")
            break

@app.route('/')
def index():
    # Video streaming home page
    return render_template('index_v.html')

@app.route('/video_feed')
def video_feed():
    # Video streaming route
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
