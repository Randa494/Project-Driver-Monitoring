import streamlit as st
import cv2
import numpy as np
import sys, os
from datetime import datetime
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
    conf_yolo     = st.slider("YOLO confidence",          0.05, 0.9, 0.25, 0.05)
    iou_thresh    = st.slider("IoU threshold",             0.1,  0.9, 0.45, 0.05)
    show_conf     = st.checkbox("Show confidence scores",  value=True)
    show_mesh     = st.checkbox("Show face landmarks",     value=True)
    process_every = st.slider("Run models every N frames", 1, 5, 3)
    st.divider()
    st.markdown("**YOLO MODELS LOADED**")
    for name in load_yolo_models():
        st.markdown(f"🟢 `{name}.pt`")
    st.divider()
    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.rerun()

yolo_models = load_yolo_models()
db_ok       = check_connection()

top_cols = st.columns([1, 1, 1, 4])
with top_cols[0]:
    if db_ok:
        st.success("MongoDB Connected")
    else:
        st.error("MongoDB Offline")
with top_cols[1]:
    cam_status = st.empty()
with top_cols[2]:
    clock_placeholder = st.empty()
    clock_placeholder.info(datetime.now().strftime("%H:%M:%S"))

st.markdown("## 🚗 Driver Monitoring System")
st.caption("Live webcam-based driver monitoring")

mp_ok, mp_msg = get_mediapipe_status()
if not mp_ok:
    st.warning(f"⚠️ Face analysis (Drowsiness/Yawning/Distraction) is currently disabled:\n\n{mp_msg}")

run = st.toggle("Start Camera", value=False)

col_feed, col_info = st.columns([3, 1])
with col_feed:
    st.markdown("##### LIVE CAMERA FEED")
    frame_placeholder = st.empty()
with col_info:
    st.markdown("##### RISK ASSESSMENT")
    risk_placeholder = st.empty()
    st.markdown("##### FACE ANALYSIS (MEDIAPIPE)")
    face_placeholder = st.empty()

st.markdown("##### OBJECT DETECTIONS (YOLO)")
yolo_placeholder = st.empty()

if run:
    cam_status.warning("Camera Active")
    analyzer = FaceAnalyzer()
    cap      = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        st.error("Cannot open webcam.")
    else:
        frame_count    = 0
        yolo_cache     = []
        face_cache     = {"face_found": False, "alerts": [], "state": "SAFE", "risk_score": 0.0}
        save_every     = 30
        email_recipient = st.session_state.get("email_recipient", "")

        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Lost camera feed.")
                break

            frame_count += 1
            clock_placeholder.info(datetime.now().strftime("%H:%M:%S"))

            if frame_count % process_every == 0:
                yolo_cache = run_yolo_models(frame, yolo_models, conf=conf_yolo, iou=iou_thresh)
                face_cache = analyzer.analyze(frame)
                if frame_count % save_every == 0:
                    save_alert("realtime", yolo_cache, face_cache,
                               email_recipient=email_recipient)

            annotated = draw_yolo_boxes(frame, yolo_cache, show_conf)
            annotated = draw_face_overlay(annotated, face_cache, show_mesh)
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

            state = face_cache.get("state", "SAFE")
            risk  = face_cache.get("risk_score", 0.0)
            state_badge = {"SAFE": "🟢", "WARNING": "🟡", "HIGH_RISK": "🔴"}
            risk_placeholder.markdown(
                f"{state_badge.get(state,'⚪')} **{state}**\n\n### {risk:.2f} / 1.00"
            )

            if face_cache.get("face_found"):
                face_placeholder.markdown(
                    f"👁 EAR `{face_cache.get('ear',0):.3f}`\n\n"
                    f"👄 MAR `{face_cache.get('mar',0):.3f}`\n\n"
                    f"👀 **{face_cache.get('direction','—')}**\n\n"
                    f"📐 **{face_cache.get('head_pose','—')}**\n\n"
                    f"🔢 Streak `{face_cache.get('ear_streak',0)}`"
                )
            else:
                face_placeholder.caption("No face detected")

            lines = []
            for (_, conf, label) in yolo_cache:
                color = get_color(label)
                lines.append(f"<span style='color:{color}'>■</span> **{label}** `{conf:.2f}`")
            yolo_placeholder.markdown(
                "\n\n".join(lines) if lines else "No detections",
                unsafe_allow_html=True
            )

        cap.release()
        analyzer.close()
        cam_status.info("Camera Idle")
else:
    cam_status.info("Camera Idle")
    frame_placeholder.markdown(
        "<div style='text-align:center;padding:80px;color:#555;'>"
        "<div style='font-size:48px;'>📷</div>"
        "<b>Camera is idle</b><br>"
        "<small>Toggle 'Start Camera' above to begin live detection.</small>"
        "</div>",
        unsafe_allow_html=True
    )
    risk_placeholder.markdown("🟢 **SAFE**\n\n### 0.00 / 1.00")
    face_placeholder.caption("No face detected")
    yolo_placeholder.caption("No detections")
