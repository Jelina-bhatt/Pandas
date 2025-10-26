# Install only OpenCV and pyttsx3
# pip install opencv-python pyttsx3

import cv2
import pyttsx3

# Initialize text-to-speech
engine = pyttsx3.init()

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Open webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]

        # Detect smile
        smiles = smile_cascade.detectMultiScale(face_gray, 1.8, 20)

        # Detect eyes for surprise/neutral
        eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 10)

        # Determine mood
        if len(smiles) > 0:
            mood = "Happy"
        elif len(eyes) >= 2:
            mood = "Surprised"
        else:
            mood = "Neutral"

        # Display rectangle and mood
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Mood: {mood}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Voice feedback
        engine.say(f"You seem {mood} today!")
        engine.runAndWait()

    cv2.imshow("Smart Mood Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
