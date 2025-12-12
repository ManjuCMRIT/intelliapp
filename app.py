import streamlit as st
import cv2
import mediapipe as mp
import os
import time

# Config
TOTAL_IMAGES = 30
CAPTURE_INTERVAL = 1

st.title("📸 Face Registration (Local Save in GitHub Folder)")
st.markdown("Enter name to capture 30 cropped face images.")

name = st.text_input("Student Name:")
start = st.button("Start Capture")

frame_placeholder = st.empty()
count_placeholder = st.empty()

mp_face = mp.solutions.face_detection

if start and name.strip():
    save_folder = f"faces/{name}"
    os.makedirs(save_folder, exist_ok=True)

    st.success(f"Saving images to: {save_folder}")

    cap = cv2.VideoCapture(0)
    captured = 0

    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
        while captured < TOTAL_IMAGES:
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam not accessible")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)

            if results.detections:
                det = results.detections[0].location_data.relative_bounding_box
                h, w, _ = frame.shape

                x = int(det.xmin * w)
                y = int(det.ymin * h)
                bw = int(det.width * w)
                bh = int(det.height * h)

                face = frame[y:y+bh, x:x+bw]

                if face.size > 0:
                    filename = os.path.join(save_folder, f"{name}_{captured + 1}.jpg")
                    cv2.imwrite(filename, face)

                    captured += 1
                    count_placeholder.write(f"Captured {captured}/{TOTAL_IMAGES}")

                cv2.rectangle(rgb, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

            frame_placeholder.image(rgb)
            time.sleep(CAPTURE_INTERVAL)

    cap.release()
    st.success("🎉 Face registration completed!")
