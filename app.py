import streamlit as st
import cv2
import os
import time
import face_recognition

# Configurations
TOTAL_IMAGES = 30
CAPTURE_INTERVAL = 2  # seconds

# Streamlit UI
st.title("📸 Face Registration (Local Only)")
st.markdown("Enter your name and click start to capture 30 face images (1 every 2 seconds).")

name = st.text_input("Enter your name:")
start_button = st.button("Start Capturing")

frame_placeholder = st.empty()
counter_placeholder = st.empty()

# Main Capture Logic
if start_button and name.strip():
    st.success(f"Starting face capture for '{name}'...")
    
    folder_path = os.path.join("faces", name)
    os.makedirs(folder_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    captured = 0

    while captured < TOTAL_IMAGES:
        ret, frame = cap.read()
        if not ret:
            st.error("Unable to access webcam.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if face_locations:
            top, right, bottom, left = face_locations[0]
            face_crop = frame[top:bottom, left:right]

            # Save face image
            img_filename = f"{name}_{captured + 1}.jpg"
            img_path = os.path.join(folder_path, img_filename)
            cv2.imwrite(img_path, face_crop)

            # Draw box for display
            cv2.rectangle(rgb_frame, (left, top), (right, bottom), (0, 255, 0), 2)

            captured += 1
            counter_placeholder.markdown(f"**Captured {captured}/{TOTAL_IMAGES} face images**")

        # Show current frame
        frame_placeholder.image(rgb_frame, channels="RGB")

        time.sleep(CAPTURE_INTERVAL)

    cap.release()
    st.success(f"✅ Done! All images saved to `faces/{name}/`.")
