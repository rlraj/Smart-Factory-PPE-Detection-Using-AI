# streamlit_ppe_app.py

import time

from collections import Counter

import cv2

import numpy as np

import streamlit as st

from ultralytics import YOLO

st.set_page_config(page_title="Real-Time PPE Detection", layout="wide")

# ---- Model load ----

# Put your model path here (on Streamlit Cloud you should include the model in the repo or load from a URL)

MODEL_PATH = r"C:\Users\220250572\OneDrive - Regal Rexnord\Desktop\PPE\runs\detect\train3\weights\best.pt"

@st.cache_resource(show_spinner=True)

def load_model(path):

    return YOLO(path)

model = load_model(MODEL_PATH)

# ---- Sidebar UI ----

st.title("Real-Time PPE Detection with YOLOv8")

st.sidebar.header("Settings")

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)

start_stop = st.sidebar.button("Start / Stop Webcam")

snapshot_btn = st.sidebar.button("Capture Snapshot (once)")

# ---- Session state init ----

if "running" not in st.session_state:

    st.session_state.running = False

if "snapshot" not in st.session_state:

    st.session_state.snapshot = None

# Toggle running when button pressed

if start_stop:

    st.session_state.running = not st.session_state.running

# Layout

col1, col2 = st.columns([3, 1])

frame_window = col1.empty()

info_area = col2.empty()

# Open webcam (index 0). If deploying to cloud, webcam won't be available;

# you need to provide a video file or external stream URL instead.

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Main loop - only runs while session_state.running is True

try:

    prev_time = time.time()

    while st.session_state.running:

        ret, frame = cap.read()

        if not ret:

            st.warning("Failed to read from webcam. Check webcam index or permissions.")

            st.session_state.running = False

            break

        # Run inference. Ultralytics accepts numpy frames via `model.predict(source=frame)`

        results = model.predict(source=frame, conf=conf_threshold, imgsz=640, verbose=False)

        # Collect class names for counting, and draw boxes

        class_names = []

        for res in results:

            # res.boxes contains boxes for this frame

            # box.xyxy, box.cls, box.conf

            for box in getattr(res, "boxes", []):

                xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], "cpu") else box.xyxy[0]

                x1, y1, x2, y2 = map(int, xyxy)

                cls_id = int(box.cls[0].cpu().numpy() if hasattr(box.cls[0], "cpu") else box.cls[0])

                conf = float(box.conf[0].cpu().numpy() if hasattr(box.conf[0], "cpu") else box.conf[0])

                label = f"{model.names[cls_id]} {conf:.2f}"

                class_names.append(model.names[cls_id])

                # draw box and label

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

                cv2.putText(frame, label, (x1, max(15, y1 - 8)),

                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

        # Convert BGR -> RGB for display

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # FPS

        now = time.time()

        fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0

        prev_time = now

        # Display frame

        frame_window.image(rgb, channels="RGB", use_column_width=True)

        # Display info, counts, and last snapshot

        counts = dict(Counter(class_names))

        info_area.markdown("### Detection info")

        info_area.write(f"**FPS:** {fps:.1f}")

        info_area.write("**Detected classes (counts):**")

        info_area.write(counts if counts else "No detections")

        # Handle snapshot button (set snapshot in session state)

        if snapshot_btn:

            # Save current displayed image (RGB)

            st.session_state.snapshot = rgb.copy()

            # Show a confirmation in sidebar

            st.sidebar.success("Snapshot captured")

            # Important: break out of loop to let Streamlit process the button action change

            # (we will continue running unless user toggles Start/Stop)

            snapshot_btn = False

        # If a snapshot exists, show it below the detection info

        if st.session_state.snapshot is not None:

            info_area.image(st.session_state.snapshot, caption="Last Snapshot", width=240)

        # small sleep to yield to Streamlit and keep CPU reasonable

        time.sleep(0.02)

        # Streamlit will only update UI after this loop yields control. The loop

        # is fine for simple local runs; if this blocks your UI, change to a

        # different architecture (e.g., st.thread or streamlit-webrtc).

except Exception as e:

    st.error(f"Runtime error: {e}")

finally:

    cap.release()

# If not running, just display placeholder and snapshot if exists

if not st.session_state.running:

    frame_window.info("Webcam stopped. Click 'Start / Stop Webcam' in the sidebar to run.")

    if st.session_state.snapshot is not None:

        st.image(st.session_state.snapshot, caption="Last Snapshot (from session)")

