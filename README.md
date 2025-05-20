# MuscleMap: A Smart System for Muscle Detection and Exercise Recommendation

MuscleMap is an intelligent system designed to detect specific muscle regions in the human body using finger-pointing gestures and provide personalized exercise recommendations. It integrates computer vision, machine learning, and gesture-based interaction to offer an interactive fitness assistant.

## 🚀 Features

- 👆 Finger Gesture Detection**: Uses webcam and MediaPipe to detect hand and finger positions in real-time.
- 💪 Muscle Region Identification**: Identifies which muscle group (e.g., chest, biceps, traps) the user is pointing to using a K-Nearest Neighbors (KNN) classifier.
- 🧠 Anomaly Detection: Utilizes the Isolation Forest algorithm to detect abnormal gestures or misclassifications.
- 📋 **Exercise Recommendation: Suggests appropriate exercises based on the detected muscle region using a dictionary-based approach.
- 📸 Visual Feedback: Displays pose landmarks, hand landmarks, detected muscle region, and recommended exercises on the screen.

## 🧰 Tech Stack

- Language: Python
- Computer Vision: OpenCV, MediaPipe
- Machine Learning: scikit-learn (KNN, Isolation Forest)
- Visualization: Real-time video feed with overlay
- Data Handling: Custom dictionary for exercise mapping

📌 Future Enhancements
Add voice assistance for exercise guidance
Incorporate more muscle groups (e.g., legs, calves)
Build a GUI or Streamlit interface for ease of use
Save user progress and history
