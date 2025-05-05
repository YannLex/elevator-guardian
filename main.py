import cv2
import time
import numpy as np
from ultralytics import YOLO

# Load the YOLOv11 model
model = YOLO('yolov8n.pt')  # Replace 'yolov8n.pt' with the correct YOLOv11 model file if available

# Load pre-trained models for age estimation and face detection
age_net = cv2.dnn.readNetFromCaffe(
    'deploy_age.prototxt',
    'age_net.caffemodel'
)
face_net = cv2.dnn.readNetFromCaffe(
    'deploy_face.prototxt',
    'res10_300x300_ssd_iter_140000.caffemodel'
)

# Define age categories
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]

# Define a function to estimate age based on face detection
def estimate_age(frame, x1, y1, x2, y2):
    face = frame[int(y1):int(y2), int(x1):int(x2)]
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_BUCKETS[age_preds[0].argmax()]
    return age

# Update the detect_children_and_adults function to use age estimation
def detect_children_and_adults(frame):
    results = model.predict(frame, save=False, conf=0.5)  # Use the predict method to get results
    children_count = 0
    adults_count = 0

    for result in results[0].boxes:  # Iterate through detected bounding boxes
        x1, y1, x2, y2 = result.xyxy[0]  # Extract bounding box coordinates
        conf = result.conf[0]  # Confidence score
        cls = int(result.cls[0])  # Class ID
        label = model.names[cls]

        if label == 'person':
            # Estimate age using the face detection model
            age = estimate_age(frame, x1, y1, x2, y2)
            if age in ["(0-2)", "(4-6)", "(8-12)"]:
                children_count += 1
                color = (255, 0, 0)  # Red for children
            else:
                adults_count += 1
                color = (0, 255, 0)  # Green for adults

            # Draw bounding box and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, age, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return children_count, adults_count

# Update the process_video function to include child/adult detection logic
def process_video(video_path):
    """
    Process the video file frame by frame to detect if a child is riding alone.
    :param video_path: Path to the video file.
    """
    # Load the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            print("End of video.")
            break

        # Detect children and adults in the frame
        children_count, adults_count = detect_children_and_adults(frame)

        # Check if a child is riding alone
        if children_count > 0 and adults_count == 0:
            print("Child detected riding alone. Stop signal sent to escalator.")
            return  # Stop further detection and exit the function

        # Display the frame with detection results
        cv2.imshow('Video Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = input("Enter the path to the video file: ")
    
    process_video(video_path)