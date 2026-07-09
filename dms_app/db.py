from pymongo import MongoClient
from datetime import datetime
import streamlit as st
from config import MONGO_URI

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


def save_alert(source: str, yolo_dets: list, face_result: dict):
    from utils import DANGER_LABELS
    yolo_alerts = [label for (_, _, label) in yolo_dets if label in DANGER_LABELS]
    face_alerts = [a for a in face_result.get("alerts", []) if a in DANGER_LABELS]
    all_alerts  = list(set(yolo_alerts + face_alerts))
    if not all_alerts:
        return False
    record = {
        "timestamp":       datetime.utcnow(),
        "source":          source,
        "alerts":          all_alerts,
        "risk_state":      face_result.get("state", "SAFE"),
        "risk_score":      round(face_result.get("risk_score", 0.0), 3),
        "yolo_detections": yolo_alerts,
        "face_alerts":     face_alerts,
        "ear":             round(face_result.get("ear", 0.0), 3),
        "mar":             round(face_result.get("mar", 0.0), 3),
        "direction":       face_result.get("direction", "FORWARD"),
        "head_pose":       face_result.get("head_pose", "FORWARD"),
    }
    try:
        get_collection().insert_one(record)
        return True
    except Exception as e:
        print(f"MongoDB save error: {e}")
        return False


def load_alerts(limit=500):
    try:
        col  = get_collection()
        docs = list(col.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
        return docs
    except Exception as e:
        print(f"MongoDB load error: {e}")
        return []