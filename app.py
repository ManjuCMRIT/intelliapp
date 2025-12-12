import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Configurations
TOTAL_IMAGES = 30
CAPTURE_INTERVAL = 2  # seconds

st.title("📸 Face Registration (Streamlit Cloud Compatible)")
st.markdown("Enter your name and click start to capture 30 face images (1 every 2 seconds).")

name = st.text_input("Enter your name:")
start_button = st.button("Start Capturing")

frame_placeholder = st.empty()
counter_placeholder = st.empty()

mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

if start_button and name.strip():
    st.success(f"Starting face capture for '{name}'...")

    folder_path = os.path.join("faces", name)
    os.makedirs(folder_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    captured = 0

    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detector:
        while captured < TOTAL_IMAGES:
            ret, frame = cap.read()
            if not ret:
                st.error("Unable to access webcam.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb)

            if results.detections:
                det = results.detections[0]
                bbox = det.location_data.relative_bounding_box

                h, w, _ = frame.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                face_crop = frame[y:y+bh, x:x+bw]
                if face_crop.size > 0:
                    img_filename = f"{name}_{captured + 1}.jpg"
                    img_path = os.path.join(folder_path, img_filename)
                    cv2.imwrite(img_path, face_crop)

                    captured += 1
                    counter_placeholder.markdown(f"**Captured {captured}/{TOTAL_IMAGES} images**")

                cv2.rectangle(rgb, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

            frame_placeholder.image(rgb, channels="RGB")

            time.sleep(CAPTURE_INTERVAL)

    cap.release()
    st.success(f"✅ Done! All images saved to `faces/{name}/`")
