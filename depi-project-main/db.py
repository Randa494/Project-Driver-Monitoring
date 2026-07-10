from pymongo import MongoClient
from datetime import datetime
import streamlit as st
from config import (MONGO_URI, ENABLE_TELEGRAM, TELEGRAM_BOT_TOKEN,
                    TELEGRAM_CHAT_ID, ENABLE_EMAIL, EMAIL_ADDRESS,
                    EMAIL_PASSWORD, SMTP_SERVER, SMTP_PORT)

DB_NAME  = "dms_db"
COL_NAME = "alerts"


@st.cache_resource(show_spinner=False)
def get_collection():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    return client[DB_NAME][COL_NAME]


def check_connection():
    try:
        get_collection().database.client.admin.command("ping")
        return True
    except Exception:
        return False


def _send_telegram(message: str):
    try:
        import requests
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=5)
    except Exception as e:
        print(f"Telegram error: {e}")


def _send_email(subject: str, body: str, recipient: str):
    try:
        import smtplib
        from email.mime.text import MIMEText
        msg            = MIMEText(body)
        msg["Subject"] = subject
        msg["From"]    = EMAIL_ADDRESS
        msg["To"]      = recipient
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, recipient, msg.as_string())
    except Exception as e:
        print(f"Email error: {e}")


def send_alert_notification(alerts: list, risk_state: str, risk_score: float,
                             source: str, email_recipient: str = ""):
    if not alerts:
        return
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    message = (
        f"🚨 DMS Alert — {risk_state}\n"
        f"📍 Source: {source}\n"
        f"⚠️ Violations: {', '.join(alerts)}\n"
        f"📊 Risk Score: {risk_score:.2f}\n"
        f"🕐 Time: {timestamp}"
    )
    if ENABLE_TELEGRAM:
        _send_telegram(message)
    if ENABLE_EMAIL and email_recipient:
        _send_email(
            subject=f"DMS Alert — {risk_state}",
            body=message,
            recipient=email_recipient
        )


def save_alert(source: str, yolo_dets: list, face_result: dict,
               email_recipient: str = ""):
    from utils import DANGER_LABELS

    # All raw YOLO detections (Seatbelt + No Seatbelt + ...) — used for logging/display only
    yolo_raw = [label for (_, _, label) in yolo_dets]

    # Only real violations (e.g. "No Seatbelt"), not just anything that was detected.
    # A positive "Seatbelt" detection is never treated as a violation.
    yolo_violations = [l for l in yolo_raw if l in DANGER_LABELS]
    face_alerts     = face_result.get("alerts", [])
    all_alerts      = list(set(yolo_violations + face_alerts))

    # If there are no actual violations at all, don't save anything
    if not all_alerts:
        return False

    # Everything in all_alerts is already a real violation at this point
    danger_alerts = all_alerts

    risk_state = face_result.get("state", "SAFE")

    # If there's a real YOLO violation (e.g. No Seatbelt), bump to at least WARNING
    if yolo_violations and risk_state == "SAFE":
        risk_state = "WARNING"

    record = {
        "timestamp":       datetime.utcnow(),
        "source":          source,
        "alerts":          all_alerts,
        "risk_state":      risk_state,
        "risk_score":      round(face_result.get("risk_score", 0.0), 3),
        "yolo_detections": yolo_raw,
        "face_alerts":     face_alerts,
        "ear":             round(face_result.get("ear", 0.0), 3),
        "mar":             round(face_result.get("mar", 0.0), 3),
        "direction":       face_result.get("direction", "FORWARD"),
        "head_pose":       face_result.get("head_pose", "FORWARD"),
    }

    try:
        get_collection().insert_one(record)
    except Exception as e:
        print(f"MongoDB save error: {e}")
        return False

    if danger_alerts and risk_state in ("WARNING", "HIGH_RISK"):
        send_alert_notification(
            alerts=danger_alerts,
            risk_state=risk_state,
            risk_score=face_result.get("risk_score", 0.0),
            source=source,
            email_recipient=email_recipient
        )

    return True


def load_alerts(limit=500):
    try:
        col  = get_collection()
        docs = list(col.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
        return docs
    except Exception as e:
        print(f"MongoDB load error: {e}")
        return []
