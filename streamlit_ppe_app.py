
import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from collections import Counter

# Load YOLOv8 model
model = YOLO(r'C:\Users\220250572\OneDrive - Regal Rexnord\Desktop\PPE\runs\detect\train3\weights\best.pt')

# Streamlit UI
st.title("Real-Time PPE Detection with YOLOv8")
st.sidebar.header("Settings")

# Confidence slider
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)

# Webcam toggle
start_webcam = st.sidebar.checkbox("Start Webcam")

# Snapshot button
snapshot = st.sidebar.button("Capture Snapshot")

# Display area
FRAME_WINDOW = st.image([])
DETECTED_CLASSES = st.empty()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Track snapshot frame
snapshot_frame = None

# Main loop
while start_webcam:
    ret, frame = cap.read()
    if not ret:
        st.warning("Failed to grab frame from webcam.")
        break

    # Run inference
    results = model.predict(source=frame, save=False, conf=conf_threshold)

    # Draw predictions and collect class names
    class_names = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{model.names[cls_id]} {conf:.2f}"
            class_names.append(model.names[cls_id])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Convert BGR to RGB for Streamlit
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(rgb_frame)

    # Display detected class counts
    class_counts = Counter(class_names)
    DETECTED_CLASSES.markdown(f"**Detected Classes:** {dict(class_counts)}")

    # Handle snapshot
    if snapshot:
        snapshot_frame = rgb_frame.copy()
        st.image(snapshot_frame, caption="Snapshot Captured")
        snapshot = False  # Reset snapshot button

cap.release()
cv2.destroyAllWindows()
