import streamlit as st

MONGO_URI    = st.secrets["MONGO_URI"]
APP_PASSWORD = st.secrets["APP_PASSWORD"]

ENABLE_TELEGRAM    = st.secrets.get("ENABLE_TELEGRAM", "false").lower() == "true"
TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = st.secrets.get("TELEGRAM_CHAT_ID", "")

ENABLE_EMAIL   = st.secrets.get("ENABLE_EMAIL", "false").lower() == "true"
EMAIL_ADDRESS  = st.secrets.get("EMAIL_ADDRESS", "")
EMAIL_PASSWORD = st.secrets.get("EMAIL_PASSWORD", "")
SMTP_SERVER    = st.secrets.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT      = int(st.secrets.get("SMTP_PORT", 587))