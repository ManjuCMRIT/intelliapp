import streamlit as st
import cv2
import os
import time
import numpy as np
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import io
import face_recognition

# Google Drive folder ID
FOLDER_ID = "1blOQmXgEPzgbC1TV5tSlKh34xm0uwUz4"

# Load credentials from Streamlit Secrets
service_account_info = st.secrets["gdrive_service"]
creds = service_account.Credentials.from_service_account_info(
    service_account_info,
    scopes=["https://www.googleapis.com/auth/drive"]
)
drive_service = build("drive", "v3", credentials=creds)

# Streamlit UI
st.title("📸 Face Registration App")
st.markdown("Captures face images and uploads to Google Drive.")

name = st.text_input("Enter Name:")
start_button = st.button("Start Capture")

frame_placeholder = st.empty()
counter_placeholder = st.empty()

TOTAL_IMAGES = 30
CAPTURE_INTERVAL = 2

if start_button and name.strip():

    st.success(f"Capturing face for {name}...")

    cap = cv2.VideoCapture(0)
    captured = 0

    while captured < TOTAL_IMAGES:
        ret, frame = cap.read()
        if not ret:
            st.error("Cannot access webcam")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if face_locations:
            top, right, bottom, left = face_locations[0]
            face_crop = frame[top:bottom, left:right]

            # Convert to bytes
            _, buffer = cv2.imencode(".jpg", face_crop)
            img_bytes = io.BytesIO(buffer.tobytes())

            # Upload to Drive
            file_metadata = {
                "name": f"{name}_{captured+1}.jpg",
                "parents": [FOLDER_ID]
            }
            media = MediaIoBaseUpload(img_bytes, mimetype="image/jpeg")

            uploaded = drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id"
            ).execute()

            # Draw rectangle for display
            cv2.rectangle(rgb_frame, (left, top), (right, bottom), (0,255,0), 2)

            captured += 1
            counter_placeholder.markdown(f"**Captured {captured}/{TOTAL_IMAGES} images**")

        frame_placeholder.image(rgb_frame, channels="RGB")
        time.sleep(CAPTURE_INTERVAL)

    cap.release()
    st.success("✅ Capture completed! Images uploaded to Google Drive.")
