from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("yolov8n-pose.pt")

# Open a connection to the webcam
cap = cv2.VideoCapture(0)  # 0 for the default webcam, change the index for other cameras

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the current frame
    results = model(frame)

    # Plot the results on the frame
    plotted_frame = results[0].plot()

    # Display the frame with the results
    cv2.imshow('YOLOv8 Pose Detection', plotted_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
