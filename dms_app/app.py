import streamlit as st
import os

st.set_page_config(
    page_title="Driver Monitoring System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not st.session_state.get("authenticated"):
    from config import APP_PASSWORD
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        if os.path.exists(logo_path):
            st.image(logo_path, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("## Driver Monitoring System")
        st.markdown("Enter your password to continue.")
        st.markdown("<br>", unsafe_allow_html=True)
        password = st.text_input("Password", type="password", placeholder="Enter password...")
        if st.button("Login", type="primary", use_container_width=True):
            if password == APP_PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    st.stop()

pg = st.navigation([
    st.Page("pages/realtime.py",  title="Real-Time Detection"),
    st.Page("pages/video.py",     title="Video Detection"),
    st.Page("pages/image.py",     title="Image Detection"),
    st.Page("pages/dashboard.py", title="Alerts Dashboard"),
    st.Page("pages/settings.py",  title="Settings"),
])

pg.run()