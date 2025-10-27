import cv2
from fer import FER
import streamlit as st
from PIL import Image
import numpy as np


# ------------------------------
# App Title and Description
# ------------------------------
st.set_page_config(page_title="AI Life Mirror — Emotion-to-Action Predictor", layout="centered")
st.title("🪞 AI Life Mirror — Emotion-to-Action Predictor")
st.write("Upload an image to detect emotion and get a personalized suggestion. 💡")

# ------------------------------
# Tabs for Different Modes
# ------------------------------
tab1, tab2 = st.tabs(["🖼️ Image", "📊 Dashboard"])

with tab1:
    st.header("Image-based Emotion Detection")

    uploaded_file = st.file_uploader("Upload an image (face visible)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Convert image to RGB for FER
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        # Initialize FER detector
        detector = FER(mtcnn=True)
        results = detector.detect_emotions(img_rgb)

        if results:
            emotion, score = detector.top_emotion(img_rgb)
            st.subheader(f"🧠 Detected Emotion: **{emotion.capitalize()}** ({score*100:.2f}% confidence)")

            # Suggestion dictionary
            suggestions = {
                "happy": "Keep spreading positivity! 🌞 Maybe share your good vibe with a friend.",
                "sad": "Take a walk, breathe deeply, or listen to calm music. 🌧️",
                "angry": "Try journaling or meditation to release tension. 🔥",
                "surprise": "Channel that energy into creativity — start something new! ⚡",
                "fear": "You’re safe. Ground yourself with deep breaths. 🌿",
                "disgust": "Maybe distance yourself from the source and reset your focus. 💭",
                "neutral": "All balanced — maybe plan something small you enjoy. 🌸"
            }

            # Display suggestion
            suggestion = suggestions.get(emotion, "Stay calm and carry on. 🌼")
            st.info(suggestion)
        else:
            st.warning("No face detected. Please upload a clearer image.")

with tab2:
    st.header("📊 Emotion Analysis Dashboard (Prototype)")
    st.write("This section can later show emotion trends, logs, and AI-based insights.")

st.markdown("---")
st.caption("Made with ❤️ using Python, Streamlit, and FER — prototype version.")
