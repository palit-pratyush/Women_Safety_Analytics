from flask import Flask, render_template, Response, jsonify
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

# Correct file paths for SSD model files
ssd_prototxt = "deploy.prototxt.txt"  # Update this with the correct path
ssd_weights = "res10_300x300_ssd_iter_140000.caffemodel"  # Update this with the correct path

# Load the pre-trained SSD model
face_net = cv2.dnn.readNetFromCaffe(ssd_prototxt, ssd_weights)

# Initialize counters for each class
face_count = {"Male": 0, "Female": 0}

def generate_frames():
    global face_count  # Use the global variable to count faces
    
    while True:
        # Capture frame-by-frame from the webcam
        success, frame = camera.read()
        if not success:
            break
        else:
            # Prepare the frame for SSD face detection
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))

            # Pass the blob through the network to detect and predict faces
            face_net.setInput(blob)
            detections = face_net.forward()

            # Reset face count for each frame
            face_count = {"Male": 0, "Female": 0}

            # Loop over the detections
            for i in range(0, detections.shape[2]):
                # Extract the confidence (i.e., probability) associated with the prediction
                confidence = detections[0, 0, i, 2]

                # Filter out weak detections by ensuring the confidence is greater than a minimum threshold
                if confidence > 0.5:
                    # Compute the (x, y)-coordinates of the bounding box for the face
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Extract the face ROI
                    face = frame[startY:endY, startX:endX]

                    # Ensure the face ROI is of sufficient size
                    if face.size == 0:
                        continue

                    # Resize the face to the required size for the model
                    face_resized = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)

                    # Preprocess the face for the model
                    face_array = np.asarray(face_resized, dtype=np.float32).reshape(1, 224, 224, 3)
                    face_array = (face_array / 127.5) - 1

                    # Predict the gender of the face
                    prediction = model.predict(face_array)
                    index = np.argmax(prediction)
                    class_name = class_names[index].strip()
                    confidence_score = prediction[0][index]

                    # Increment the count for the detected gender
                    if class_name in face_count:
                        face_count[class_name] += 1

                    # Draw rectangle around the face
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

                    # Display label and confidence score
                    label = f"{class_name}: {confidence_score*100:.2f}%"
                    cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Use yield to stream the video
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for the home page
@app.route('/')
def index():
    return render_template('index2.html')

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to get face count data
@app.route('/face_count')
def get_face_count():
    total_faces = face_count["Male"] + face_count["Female"]
    ratio = {"Male": 0, "Female": 0}
    
    if total_faces > 0:
        ratio["Male"] = face_count["Male"] / total_faces
        ratio["Female"] = face_count["Female"] / total_faces

    data = {
        "face_count": face_count,
        "ratio": ratio
    }
    return jsonify(data)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
