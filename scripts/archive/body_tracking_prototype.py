import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import numpy as np
import time
import os
import urllib.request
from collections import deque

# Download the model if it doesn't exist
model_path = 'pose_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading Pose Landmarker model...")
    urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task', model_path)
    print("Download complete.")

# Initialize MediaPipe Pose Landmarker
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False)
detector = vision.PoseLandmarker.create_from_options(options)

# Variables for tracking state
hip_y_history = deque(maxlen=15)
shoulder_x_history = deque(maxlen=5)
last_time = time.time()

def calculate_distance(p1, p2):
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

def main():
    global last_time
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Camera started! Try the following movements:")
    print("1. Bending Left / Right")
    print("2. Jumping")
    print("Press 'q' to quit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        if dt == 0: dt = 0.001

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect pose
        detection_result = detector.detect(mp_image)
        
        current_action = "IDLE"

        if detection_result.pose_landmarks:
            # We only use the first detected person
            landmarks = detection_result.pose_landmarks[0]
            
            # --- Get Key Landmarks ---
            # Using landmark indices directly from the MediaPipe Pose topology
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            left_elbow = landmarks[13]
            right_elbow = landmarks[14]
            
            shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_mid_x = (left_hip.x + right_hip.x) / 2
            hip_mid_y = (left_hip.y + right_hip.y) / 2
            
            # --- 1. Jump Detection ---
            hip_y_history.append(hip_mid_y)
            if current_action == "IDLE" and len(hip_y_history) == 15:
                avg_hip_y = sum(hip_y_history) / 15
                if (avg_hip_y - hip_mid_y) > 0.05:
                    current_action = "JUMPING ⬆️"
            
            # --- 2. Bending Left / Right ---
            if current_action == "IDLE":
                dx = shoulder_mid_x - hip_mid_x
                if dx > 0.06:
                    current_action = "BENDING RIGHT ➡️"
                elif dx < -0.06:
                    current_action = "BENDING LEFT ⬅️"

            # Draw basic connections manually for visualization
            connections = [(11, 12), (11, 23), (12, 24), (23, 24), (11, 13), (13, 15), (12, 14), (14, 16)]
            h, w, c = image.shape
            for connection in connections:
                start_point = landmarks[connection[0]]
                end_point = landmarks[connection[1]]
                cv2.line(image, (int(start_point.x * w), int(start_point.y * h)),
                         (int(end_point.x * w), int(end_point.y * h)), (255, 0, 0), 2)
            for i in [11, 12, 13, 14, 15, 16, 23, 24]:
                cv2.circle(image, (int(landmarks[i].x * w), int(landmarks[i].y * h)), 5, (0, 0, 255), -1)

            # Display the Action
            cv2.putText(image, f"ACTION: {current_action}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255) if current_action != "IDLE" else (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('Naruto Run - Body Movement Prototype', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
