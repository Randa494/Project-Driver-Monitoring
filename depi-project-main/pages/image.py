import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import (load_yolo_models, run_yolo_models, FaceAnalyzerStatic,
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

st.title("Image Detection")
st.caption("Upload one or more images for DMS analysis")

mp_ok, mp_msg = get_mediapipe_status()
if not mp_ok:
    st.warning(f"⚠️ Face analysis (Drowsiness/Yawning/Distraction) is currently disabled:\n\n{mp_msg}")

uploaded_files = st.file_uploader(
    "Upload images",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True
)

if uploaded_files:
    analyzer = FaceAnalyzerStatic()

    for uploaded_file in uploaded_files:
        st.divider()
        st.markdown(f"**{uploaded_file.name}**")

        image = Image.open(uploaded_file).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        yolo_dets = run_yolo_models(frame, yolo_models, conf=conf_yolo, iou=iou_thresh)
        face      = analyzer.analyze(frame)
        save_alert("image", yolo_dets, face, email_recipient=email_recipient)

        annotated = draw_yolo_boxes(frame, yolo_dets, show_conf)
        annotated = draw_face_overlay(annotated, face, show_mesh)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        col_img, col_det = st.columns([2, 1])
        with col_img:
            st.image(rgb, use_container_width=True)

        with col_det:
            state = face.get("state", "SAFE")
            risk  = face.get("risk_score", 0.0)
            state_colors = {"SAFE": "green", "WARNING": "orange", "HIGH_RISK": "red"}
            sc = state_colors.get(state, "gray")
            st.markdown(f"**Risk: :{sc}[{state}]** `{risk:.2f}`")

            st.markdown("**YOLO**")
            if yolo_dets:
                for (_, conf, label) in yolo_dets:
                    color = get_color(label)
                    flag  = "🔴" if label in DANGER_LABELS else "🟢"
                    st.markdown(
                        f"{flag} <span style='color:{color}'>■</span> **{label}** `{conf:.3f}`",
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No YOLO detections")

            st.markdown("**Face Analysis**")
            if face.get("face_found"):
                st.markdown(f"👁 EAR `{face.get('ear',0):.3f}` → **{'Eyes Closed' if face.get('ear',1)<0.20 else 'Open'}**")
                st.markdown(f"👄 MAR `{face.get('mar',0):.3f}` → **{'Yawning' if face.get('mar',0)>0.55 else 'Closed'}**")
                st.markdown(f"👀 Direction: **{face.get('direction','—')}**")
                st.markdown(f"📐 Head Pose: **{face.get('head_pose','—')}**")
                if face.get("alerts"):
                    st.error("Alerts: " + ", ".join(face["alerts"]))
            else:
                st.caption("No face detected")

        out_buf = io.BytesIO()
        Image.fromarray(rgb).save(out_buf, format="PNG")
        st.download_button(
            label="Download annotated image",
            data=out_buf.getvalue(),
            file_name=f"dms_{uploaded_file.name}",
            mime="image/png",
        )

    analyzer.close()
else:
    st.info("Upload one or more images to run detection.")
