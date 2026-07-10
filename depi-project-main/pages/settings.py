import streamlit as st
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from db import check_connection

logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "logo.png")

with st.sidebar:
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
        st.divider()
    st.caption("These preferences are UI-only for now.")
    st.divider()
    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.rerun()

db_ok = check_connection()

top = st.columns([6, 1])
with top[0]:
    st.markdown("## 🚗 Settings")
    st.caption("Display and notification preferences")
with top[1]:
    if db_ok:
        st.success("MongoDB Connected")
    else:
        st.error("MongoDB Offline")

st.info("Detection confidence and thresholds are controlled by the sliders on each detection page.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### CAMERA")
    st.selectbox("Preferred camera device", ["Default (index 0)", "Camera 1 (index 1)", "Camera 2 (index 2)"])
    st.selectbox("Preferred resolution", ["1280 × 720", "1920 × 1080", "640 × 480"])
    st.toggle("Mirror preview", value=False)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### DETECTION DISPLAY")
    st.toggle("Show confidence scores by default",  value=True)
    st.toggle("Show face landmark mesh by default", value=True)
    st.toggle("Show bounding box labels",           value=True)
    st.caption("These only set the default checkbox state on other pages once connected.")

with col2:
    st.markdown("#### APPEARANCE")
    st.radio("Theme", ["Dark (current)", "Light", "Auto"], horizontal=True)
    st.select_slider("Card density", options=["Compact", "Comfortable", "Spacious"], value="Comfortable")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### NOTIFICATIONS")
    st.toggle("Sound alert on HIGH_RISK",          value=True)
    st.toggle("Desktop notification on new alert", value=False)

    st.markdown("**Email Alerts**")
    email_recipient = st.text_input(
        "Notification email",
        value=st.session_state.get("email_recipient", ""),
        placeholder="you@example.com"
    )
    st.caption("You will receive an email on WARNING and HIGH_RISK alerts.")

st.markdown("<br>", unsafe_allow_html=True)
if st.button("Save preferences", type="primary"):
    st.session_state["email_recipient"] = email_recipient
    st.success("Preferences saved.")