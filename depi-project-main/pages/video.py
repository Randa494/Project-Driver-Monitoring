import streamlit as st
import cv2
import numpy as np
import tempfile, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import (load_yolo_models, run_yolo_models, FaceAnalyzer,
                   draw_yolo_boxes, draw_face_overlay,
                   get_color, DANGER_LABELS, get_mediapipe_status)
from db import save_alert, check_connection

logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "logo.png")

with st.sidebar:
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
        st.divider()
    st.subheader("Detection Settings")
    conf_yolo  = st.slider("YOLO confidence", 0.05, 0.9, 0.25, 0.05)
    iou_thresh = st.slider("IoU threshold",    0.1,  0.9, 0.45, 0.05)
    show_conf  = st.checkbox("Show confidence scores", value=True)
    show_mesh  = st.checkbox("Show face landmarks",    value=True)
    frame_skip = st.slider("Process every N frames", 1, 5, 3)
    st.caption("Higher N = faster playback.")
    st.divider()
    st.markdown("**YOLO MODELS LOADED**")
    for name in load_yolo_models():
        st.markdown(f"🟢 `{name}.pt`")
    st.divider()
    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.rerun()

yolo_models     = load_yolo_models()
db_ok           = check_connection()
email_recipient = st.session_state.get("email_recipient", "")

if db_ok:
    st.success("MongoDB Connected")
else:
    st.error("MongoDB Offline")

st.title("Video Detection")
st.caption("Upload a video and run full DMS inference")

mp_ok, mp_msg = get_mediapipe_status()
if not mp_ok:
    st.warning(f"⚠️ Face analysis (Drowsiness/Yawning/Distraction) is currently disabled:\n\n{mp_msg}")

uploaded = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read())
    tfile.flush()

    cap          = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25

    st.info(f"Video loaded — {total_frames} frames @ {fps:.1f} FPS")
    run_btn = st.button("Run Detection", type="primary")

    if run_btn:
        frame_placeholder = st.empty()
        progress          = st.progress(0, text="Processing...")
        alert_placeholder = st.empty()

        analyzer    = FaceAnalyzer()
        frame_count = 0
        yolo_cache  = []
        face_cache  = {"face_found": False, "alerts": [], "state": "SAFE", "risk_score": 0.0}
        det_counts  = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            progress.progress(
                min(frame_count / max(total_frames, 1), 1.0),
                text=f"Frame {frame_count}/{total_frames}"
            )

            if frame_count % frame_skip == 0:
                yolo_cache = run_yolo_models(frame, yolo_models, conf=conf_yolo, iou=iou_thresh)
                face_cache = analyzer.analyze(frame)
                save_alert("video", yolo_cache, face_cache,
                           email_recipient=email_recipient)
                for (_, _, label) in yolo_cache:
                    det_counts[label] = det_counts.get(label, 0) + 1
                for alert in face_cache.get("alerts", []):
                    det_counts[alert] = det_counts.get(alert, 0) + 1

            annotated = draw_yolo_boxes(frame, yolo_cache, show_conf)
            annotated = draw_face_overlay(annotated, face_cache, show_mesh)
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

            state      = face_cache.get("state", "SAFE")
            all_alerts = list(set(
                [l for (_, _, l) in yolo_cache if l in DANGER_LABELS] +
                face_cache.get("alerts", [])
            ))
            if state == "HIGH_RISK" or any(l in DANGER_LABELS for l in all_alerts):
                alert_placeholder.error(" | ".join(all_alerts) if all_alerts else "HIGH RISK")
            elif state == "WARNING":
                alert_placeholder.warning(" | ".join(all_alerts) if all_alerts else "WARNING")
            else:
                alert_placeholder.success("Driver OK")

        cap.release()
        os.unlink(tfile.name)
        analyzer.close()
        progress.empty()

        st.success("Detection complete")
        if det_counts:
            st.markdown("### Detection Summary")
            for name, count in sorted(det_counts.items(), key=lambda x: -x[1]):
                st.metric(name, count)
