# Install dependencies first:
# pip install opencv-python pyttsx3 fer

import cv2
import pyttsx3
from fer import FER
import datetime
import time
import os
import pickle
import numpy as np

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion detector
emotion_detector = FER(mtcnn=True)

# Load known face images (one per person in 'known_faces/' folder)
known_faces = {}
if os.path.exists("faces_data.pkl"):
    with open("faces_data.pkl", "rb") as f:
        known_faces = pickle.load(f)
else:
    # Encode faces using simple grayscale resizing
    for file_name in os.listdir("known_faces"):
        img_path = os.path.join("known_faces", file_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(img, (100, 100))
        known_faces[file_name.split(".")[0]] = resized
    with open("faces_data.pkl", "wb") as f:
        pickle.dump(known_faces, f)

# Scheduled reminders (example)
reminders = {
    "09:00": "Take your morning medicine",
    "12:30": "Time for lunch",
    "18:00": "Evening walk time"
}

# Helper function: match detected face with known faces
def recognize_face(face_img):
    face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (100, 100))
    for name, known in known_faces.items():
        diff = np.sum((known - face_resized) ** 2)
        if diff < 5000000:  # threshold for matching
            return name
    return "Unknown"

# Open webcam
video_capture = cv2.VideoCapture(0)
last_reminder_checked = None

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        # Recognize face
        name = recognize_face(face_img)

        # Emotion detection
        result = emotion_detector.top_emotion(face_img)
        emotion = result[0] if result else "Neutral"

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} | {emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Voice greeting
        engine.say(f"Hello {name}, you look {emotion} today!")
        engine.runAndWait()

    # Scheduled reminders check
    current_time = datetime.datetime.now().strftime("%H:%M")
    if current_time in reminders and current_time != last_reminder_checked:
        engine.say(f"Reminder: {reminders[current_time]}")
        engine.runAndWait()
        last_reminder_checked = current_time

    cv2.imshow('Memory Guardian', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
