import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from muscle_exercises import get_exercise_for_muscle  # Import the exercise function
from anomoly import create_detector
import csv
import os

# Initialize Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# Function to calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# KNN initialization moved outside the function to avoid re-initializing
def initialize_knn(muscle_points):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(np.array(list(muscle_points.values())), list(muscle_points.keys()))
    return knn

# Function to check if the pointer is near a body part using KNN for pattern matching
def is_pointer_near_body_knn(finger_point, knn, muscle_points, threshold=50):
    muscle_name = knn.predict([finger_point])[0]
    distance = calculate_distance(finger_point, muscle_points[muscle_name])
    if distance < threshold:
        return muscle_name, distance
    return None, None

# Function to calculate the midpoint between two points
def calculate_midpoint(point1, point2):
    return int((point1[0] + point2[0]) / 2), int((point1[1] + point2[1]) / 2)

# Function to write data to CSV
def write_data_to_csv(finger_position, muscle_name, muscle_position):
    file_exists = os.path.isfile('knn_training_data.csv')

    try:
        with open('knn_training_data.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                # Write header if file is new
                writer.writerow(['finger_x', 'finger_y', 'muscle', 'muscle_x', 'muscle_y'])
            writer.writerow([finger_position[0], finger_position[1], muscle_name, muscle_position[0], muscle_position[1]])
    except IOError as e:
        print(f"Error writing to CSV: {e}")

# Main function
def main():
    try:
        # Initialize webcam
        cap = cv2.VideoCapture(0)

        # Set the width and height of the video capture (optimize performance)
        cap.set(3, 1280)  # Width
        cap.set(4, 720)   # Height
    except Exception as e:
        print(f"Error initializing webcam: {e}")
        return

    # Initialize both hands and pose models
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands, \
            mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5) as pose:

        frame_skip = 2  # Process every 2nd frame for performance
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error capturing video frame.")
                break

            # Process every N-th frame to reduce computation load
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            # Convert the frame to RGB format for processing with Mediapipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process hand landmarks
            hand_results = hands.process(image_rgb)

            # Process body landmarks
            pose_results = pose.process(image_rgb)

            # Initialize finger pointer position
            finger_position = None

            # Detect hands and find the index finger tip
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # Get index finger tip (landmark 8 is the index finger tip)
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    finger_x = int(index_finger_tip.x * frame.shape[1])
                    finger_y = int(index_finger_tip.y * frame.shape[0])
                    finger_position = (finger_x, finger_y)

                    # Draw the pointer on the frame (index finger tip)
                    cv2.circle(frame, finger_position, 10, (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, 'Pointer', (finger_x - 10, finger_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)

            # Detect body pose and check if the finger is near any body part
            if pose_results.pose_landmarks and finger_position:
                landmarks = pose_results.pose_landmarks.landmark

                # Ensure specific landmarks exist before using them
                if len(landmarks) > max(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER):
                    # Shoulder, elbow, and hip points
                    left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1]),
                                     int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0]))
                    right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1]),
                                      int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0]))

                    left_elbow = (int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * frame.shape[1]),
                                  int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * frame.shape[0]))
                    right_elbow = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * frame.shape[1]),
                                   int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * frame.shape[0]))

                    # Calculate biceps positions (midpoints between shoulders and elbows)
                    left_bicep_position = calculate_midpoint(left_shoulder, left_elbow)
                    right_bicep_position = calculate_midpoint(right_shoulder, right_elbow)

                    # Calculate triceps positions (elbows)
                    left_tricep_position = left_elbow
                    right_tricep_position = right_elbow

                    # Calculate chest position (midpoint between shoulders)
                    chest_position = calculate_midpoint(left_shoulder, right_shoulder)

                    # Calculate trapezius positions (approx. top of shoulders, mid-neck area)
                    left_trap_position = (int((left_shoulder[0] + right_shoulder[0]) / 2), int(left_shoulder[1] * 0.8))
                    right_trap_position = (int((left_shoulder[0] + right_shoulder[0]) / 2), int(right_shoulder[1] * 0.8))

                    # Define all upper body muscle points and names
                    muscle_points = {
                        "Left Bicep": left_bicep_position,
                        "Right Bicep": right_bicep_position,
                        "Left Tricep": left_tricep_position,
                        "Right Tricep": right_tricep_position,
                        "Chest (Pectorals)": chest_position,
                        "Left Shoulder (Deltoid)": left_shoulder,
                        "Right Shoulder (Deltoid)": right_shoulder,
                        "Left Trapezius": left_trap_position,
                        "Right Trapezius": right_trap_position
                    }

                    # Initialize KNN with the muscle points
                    knn = initialize_knn(muscle_points)

                    # Use KNN pattern matching to check if the pointer is near any muscle
                    muscle_name, distance = is_pointer_near_body_knn(finger_position, knn, muscle_points, threshold=50)

                    if muscle_name:
                        # Highlight the muscle point
                        cv2.circle(frame, muscle_points[muscle_name], 15, (255, 0, 0), cv2.FILLED)  # Highlight body part
                        cv2.putText(frame, muscle_name, (muscle_points[muscle_name][0] - 100, muscle_points[muscle_name][1] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                        # Get the exercise for the detected muscle and display it
                        exercise = get_exercise_for_muscle(muscle_name)
                        exercise_text = ', '.join(exercise)  # Convert list to string
                        cv2.putText(frame, f'Exercise: {exercise_text}', (muscle_points[muscle_name][0] - 100, muscle_points[muscle_name][1] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        # Write the data to CSV
                        write_data_to_csv(finger_position, muscle_name, muscle_points[muscle_name])

            # Draw body pose landmarks on the frame
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display the frame
            try:
                cv2.imshow('Finger Pointer and Upper Body Muscle Detection with KNN', frame)
            except cv2.error as e:
                print(f"OpenCV error: {e}")

            # Break the loop on 'q' key press
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
