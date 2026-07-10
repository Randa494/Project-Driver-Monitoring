import streamlit as st
from ultralytics import YOLO
import os
import cv2
import numpy as np
from collections import deque

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
    MEDIAPIPE_OK = True
    MEDIAPIPE_IMPORT_ERROR = None
except Exception as e:
    MEDIAPIPE_OK = False
    MEDIAPIPE_IMPORT_ERROR = str(e)
    print(f"MediaPipe import error: {e}")

MODEL_DIR       = os.path.join(os.path.dirname(__file__), "models")
FACE_MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")

# Updated whenever the Face Landmarker model fails to load/init after a successful import
LANDMARKER_INIT_ERROR = None

YOLO_COLORS = {
    "Seatbelt":           "#22c55e",
    "No Seatbelt":        "#ef4444",
    "Eating or Drinking": "#eab308",
    "Mobile Phone":       "#f97316",
    "Smoking":            "#8b5cf6",
}

MEDIAPIPE_COLORS = {
    "Eyes Closed":     "#ec4899",
    "Drowsy":          "#dc2626",
    "Yawning":         "#f59e0b",
    "Looking Left":    "#06b6d4",
    "Looking Right":   "#06b6d4",
    "Looking Down":    "#06b6d4",
    "Looking Up":      "#06b6d4",
    "Looking Forward": "#22c55e",
    "No Face":         "#94a3b8",
    "SAFE":            "#22c55e",
    "WARNING":         "#f97316",
    "HIGH_RISK":       "#ef4444",
}

DANGER_LABELS = {
    "No Seatbelt", "Eating or Drinking", "Mobile Phone",
    "Smoking", "Eyes Closed", "Drowsy", "Yawning",
    "Looking Left", "Looking Right", "Looking Down",
}

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61,  291,  39, 181,   0,  17]
NOSE_TIP        = 1
LEFT_EYE_OUTER  = 263
RIGHT_EYE_OUTER = 33
NOSE_BRIDGE     = 168

EAR_THRESHOLD       = 0.20
EAR_CONSEC_FRAMES   = 60
MAR_THRESHOLD       = 0.55
WINDOW_SIZE         = 25
HEAD_CENTER_MARGIN  = 0.12
HEAD_DOWN_THRESHOLD = 0.15
HEAD_UP_THRESHOLD   = -0.05
FACE_MISSING_LIMIT  = 15
W_DROWSINESS        = 0.45
W_YAWNING           = 0.20
W_DISTRACTION       = 0.35
RISK_SAFE           = 0.35
RISK_WARNING        = 0.60


def display_name(raw: str) -> str:
    mapping = {
        "drink":        "Eating or Drinking",
        "eat_drink":    "Eating or Drinking",
        "eating":       "Eating or Drinking",
        "eat":          "Eating or Drinking",
        "food":         "Eating or Drinking",
        "mobile":       "Mobile Phone",
        "mobile_phone": "Mobile Phone",
        "phone":        "Mobile Phone",
        "cell_phone":   "Mobile Phone",
        "smoke":        "Smoking",
        "smoking":      "Smoking",
        "cigarette":    "Smoking",
        "no-seatbelt":  "No Seatbelt",
        "no_seatbelt":  "No Seatbelt",
        "seatbelt":     "Seatbelt",
    }
    return mapping.get(raw.lower(), raw)


def get_color(label: str) -> str:
    return {**YOLO_COLORS, **MEDIAPIPE_COLORS}.get(label, "#94a3b8")


@st.cache_resource(show_spinner="Loading YOLO models...")
def load_yolo_models():
    models = {}
    for name in ["seatbelt", "eat_drink", "mobile", "smoke"]:
        path = os.path.join(MODEL_DIR, f"{name}.pt")
        if os.path.exists(path):
            models[name] = YOLO(path)
    return models


def _iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter)


def suppress_conflicts(detections, iou_thresh=0.45):
    if len(detections) < 2:
        return detections
    detections = sorted(detections, key=lambda x: -x[1])
    kept, suppressed = [], set()
    for i, (bi, ci, ni) in enumerate(detections):
        if i in suppressed:
            continue
        kept.append(detections[i])
        for j, (bj, cj, nj) in enumerate(detections):
            if j <= i or j in suppressed:
                continue
            if ni != nj and _iou(bi, bj) > iou_thresh:
                suppressed.add(j)
    return kept


def run_yolo_models(frame, models, conf=0.25, iou=0.45):
    all_dets = []
    for model_name, model in models.items():
        results = model.predict(frame, imgsz=640, conf=conf, iou=iou, verbose=False)
        for result in results:
            for box in result.boxes:
                raw = model.names[int(box.cls[0])]
                all_dets.append((
                    box.xyxy[0].tolist(),
                    float(box.conf[0]),
                    display_name(raw)
                ))
    return suppress_conflicts(all_dets)


def _enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _make_landmarker():
    global LANDMARKER_INIT_ERROR
    LANDMARKER_INIT_ERROR = None

    if not MEDIAPIPE_OK:
        LANDMARKER_INIT_ERROR = MEDIAPIPE_IMPORT_ERROR or "MediaPipe import failed"
        return None
    if not os.path.exists(FACE_MODEL_PATH):
        try:
            import urllib.request
            os.makedirs(MODEL_DIR, exist_ok=True)
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, FACE_MODEL_PATH)
        except Exception as e:
            print(f"Failed to download face model: {e}")
            LANDMARKER_INIT_ERROR = f"Failed to download face model: {e}"
            return None
    try:
        options = FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        return FaceLandmarker.create_from_options(options)
    except Exception as e:
        print(f"FaceLandmarker init error: {e}")
        LANDMARKER_INIT_ERROR = str(e)
        return None


def get_mediapipe_status():
    """
    Returns (ok, message).
    If ok is False, message explains the real reason MediaPipe isn't working,
    so it can be shown in the UI instead of staying buried in the server logs.
    """
    if not MEDIAPIPE_OK:
        reason = MEDIAPIPE_IMPORT_ERROR or "unknown reason"
        hint = ""
        if "portaudio" in reason.lower() or "sounddevice" in reason.lower():
            hint = (
                "\n\nMost likely cause: the `sounddevice` package (a new dependency pulled in "
                "by mediapipe) requires a system library called `libportaudio2` that isn't "
                "available on Streamlit Community Cloud by default. Fix: add a `packages.txt` "
                "file in the project root containing the line `libportaudio2`, then Reboot the app."
            )
        return False, f"MediaPipe import failed: `{reason}`{hint}"

    if LANDMARKER_INIT_ERROR:
        return False, f"Face Landmarker model failed to load: `{LANDMARKER_INIT_ERROR}`"

    return True, ""


def _detect_face(landmarker, frame_bgr):
    for img in [frame_bgr, _enhance(frame_bgr)]:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        if result.face_landmarks:
            return result.face_landmarks[0]
    return None


def _get_coords(lm, indices, w, h):
    return np.array([[lm[i].x * w, lm[i].y * h] for i in indices], dtype=np.float64)


def _compute_ear(eye_lm):
    if eye_lm is None or len(eye_lm) < 6:
        return 0.0
    P1, P2, P3, P4, P5, P6 = eye_lm
    A = np.linalg.norm(P2 - P6)
    B = np.linalg.norm(P3 - P5)
    C = np.linalg.norm(P1 - P4)
    return float((A + B) / (2.0 * C + 1e-6))


def _compute_mar(mouth_lm):
    if mouth_lm is None or len(mouth_lm) < 6:
        return 0.0
    P1, P2, P3, P4, P5, P6 = mouth_lm
    A = np.linalg.norm(P3 - P6)
    B = np.linalg.norm(P4 - P5)
    C = np.linalg.norm(P1 - P2)
    return float((A + B) / (2.0 * C + 1e-6))


def _compute_direction(lm):
    nose_x = lm[NOSE_TIP].x
    mid_x  = (lm[LEFT_EYE_OUTER].x + lm[RIGHT_EYE_OUTER].x) / 2.0
    dev    = nose_x - mid_x
    if dev > HEAD_CENTER_MARGIN:
        return "RIGHT", dev
    elif dev < -HEAD_CENTER_MARGIN:
        return "LEFT", dev
    return "FORWARD", dev


def _compute_pitch(lm):
    nose_y       = lm[NOSE_TIP].y
    eye_center_y = (
        np.mean([lm[idx].y for idx in LEFT_EYE[:3]]) +
        np.mean([lm[idx].y for idx in RIGHT_EYE[:3]])
    ) / 2.0
    offset = nose_y - eye_center_y
    if abs(offset) < 0.03:
        return "FORWARD", offset
    if offset > HEAD_DOWN_THRESHOLD:
        return "LOOKING_DOWN", offset
    elif offset < HEAD_UP_THRESHOLD:
        return "LOOKING_UP", offset
    return "FORWARD", offset


class TemporalAnalyzer:
    def __init__(self, window_size=WINDOW_SIZE):
        self.window_size         = window_size
        self.ear_history         = deque(maxlen=window_size)
        self.mar_history         = deque(maxlen=window_size)
        self.dir_history         = deque(maxlen=window_size)
        self.head_pose_history   = deque(maxlen=window_size)
        self.ear_closed_streak   = 0
        self.head_down_streak    = 0
        self.face_missing_frames = 0

    def update(self, ear, mar, direction, head_pose="FORWARD", face_detected=True):
        self.ear_history.append(ear)
        self.mar_history.append(mar)
        self.dir_history.append(direction)
        self.head_pose_history.append(head_pose)
        if ear < EAR_THRESHOLD:
            self.ear_closed_streak += 1
        else:
            self.ear_closed_streak = 0
        if head_pose == "LOOKING_DOWN":
            self.head_down_streak += 1
        else:
            self.head_down_streak = 0
        if face_detected:
            self.face_missing_frames = 0
        else:
            self.face_missing_frames += 1

    def reset(self):
        self.ear_history.clear()
        self.mar_history.clear()
        self.dir_history.clear()
        self.head_pose_history.clear()
        self.ear_closed_streak   = 0
        self.head_down_streak    = 0
        self.face_missing_frames = 0

    @property
    def avg_ear(self):
        return float(np.mean(self.ear_history)) if self.ear_history else 0.30

    @property
    def avg_mar(self):
        return float(np.mean(self.mar_history)) if self.mar_history else 0.0

    @property
    def is_drowsy(self):
        return self.ear_closed_streak >= EAR_CONSEC_FRAMES

    @property
    def is_yawning(self):
        return self.avg_mar > MAR_THRESHOLD

    @property
    def is_distracted(self):
        if not self.dir_history:
            return False
        non_fwd = sum(1 for d in self.dir_history if d != "FORWARD")
        return (non_fwd / len(self.dir_history)) >= 0.75

    @property
    def current_direction(self):
        return self.dir_history[-1] if self.dir_history else "FORWARD"

    @property
    def is_head_down(self):
        return self.head_down_streak >= 45

    @property
    def is_face_missing(self):
        return self.face_missing_frames > FACE_MISSING_LIMIT

    @property
    def current_head_pose(self):
        return self.head_pose_history[-1] if self.head_pose_history else "FORWARD"


class RiskEngine:
    def compute(self, analyzer: TemporalAnalyzer):
        drowsiness_signal = 0.0
        if analyzer.is_head_down:
            drowsiness_signal = 0.95 if analyzer.avg_ear < EAR_THRESHOLD else 0.85
        else:
            ear_norm      = max(0.0, min(1.0, 1.0 - (analyzer.avg_ear / EAR_THRESHOLD)))
            streak_boost  = 0.3 if analyzer.is_drowsy else 0.0
            drowsiness_signal = min(1.0, ear_norm + streak_boost)
        if analyzer.is_face_missing:
            drowsiness_signal = max(drowsiness_signal, 0.90)

        yawning_signal = 0.0
        if analyzer.avg_mar > MAR_THRESHOLD:
            yawning_signal = min(1.0, (analyzer.avg_mar - MAR_THRESHOLD) / (1.0 - MAR_THRESHOLD))

        distraction_signal = 0.0
        if analyzer.dir_history:
            non_fwd = sum(1 for d in analyzer.dir_history if d != "FORWARD")
            distraction_signal = non_fwd / len(analyzer.dir_history)

        risk_score = float(np.clip(
            W_DROWSINESS  * drowsiness_signal +
            W_YAWNING     * yawning_signal    +
            W_DISTRACTION * distraction_signal,
            0.0, 1.0
        ))

        components = {
            "drowsiness":   round(drowsiness_signal, 3),
            "yawning":      round(yawning_signal, 3),
            "distraction":  round(distraction_signal, 3),
            "head_down":    1.0 if analyzer.is_head_down else 0.0,
            "face_missing": 1.0 if analyzer.is_face_missing else 0.0,
        }
        return risk_score, components

    def decide(self, risk_score):
        if risk_score < RISK_SAFE:
            return "SAFE"
        elif risk_score < RISK_WARNING:
            return "WARNING"
        return "HIGH_RISK"


class FaceAnalyzer:
    def __init__(self):
        self._landmarker = _make_landmarker()
        self._temporal   = TemporalAnalyzer()
        self._risk       = RiskEngine()

    def analyze(self, frame_bgr):
        if self._landmarker is None:
            return {"face_found": False, "alerts": [], "state": "SAFE",
                    "risk_score": 0.0, "error": "FaceLandmarker not initialized"}

        h, w = frame_bgr.shape[:2]
        lm   = _detect_face(self._landmarker, frame_bgr)

        if lm is None:
            self._temporal.update(0.30, 0.12, "FORWARD", "FORWARD", face_detected=False)
            risk_score, components = self._risk.compute(self._temporal)
            state = self._risk.decide(risk_score)
            return {
                "face_found":  False,
                "alerts":      ["No Face"] if self._temporal.is_face_missing else [],
                "state":       state,
                "risk_score":  risk_score,
                "components":  components,
            }

        left_eye_pts  = _get_coords(lm, LEFT_EYE,  w, h)
        right_eye_pts = _get_coords(lm, RIGHT_EYE, w, h)
        mouth_pts     = _get_coords(lm, MOUTH,     w, h)

        ear          = (_compute_ear(left_eye_pts) + _compute_ear(right_eye_pts)) / 2.0
        mar          = _compute_mar(mouth_pts)
        direction, _ = _compute_direction(lm)
        head_pose, _ = _compute_pitch(lm)

        self._temporal.update(ear, mar, direction, head_pose, face_detected=True)
        risk_score, components = self._risk.compute(self._temporal)
        state = self._risk.decide(risk_score)

        alerts = []
        if self._temporal.is_drowsy or self._temporal.is_head_down:
            alerts.append("Drowsy" if self._temporal.is_drowsy else "Eyes Closed")
        if self._temporal.is_yawning:
            alerts.append("Yawning")
        if direction != "FORWARD":
            alerts.append(f"Looking {direction.capitalize()}")
        if head_pose == "LOOKING_DOWN":
            alerts.append("Looking Down")
        elif head_pose == "LOOKING_UP":
            alerts.append("Looking Up")

        return {
            "face_found":  True,
            "ear":         ear,
            "mar":         mar,
            "direction":   direction,
            "head_pose":   head_pose,
            "risk_score":  risk_score,
            "state":       state,
            "components":  components,
            "alerts":      alerts,
            "landmarks":   lm,
            "img_w":       w,
            "img_h":       h,
            "ear_streak":  self._temporal.ear_closed_streak,
            "is_drowsy":   self._temporal.is_drowsy,
        }

    def reset(self):
        self._temporal.reset()

    def close(self):
        if self._landmarker:
            self._landmarker.close()


class FaceAnalyzerStatic:
    def __init__(self):
        self._landmarker = _make_landmarker()

    def analyze(self, frame_bgr):
        if self._landmarker is None:
            return {"face_found": False, "alerts": [], "state": "SAFE",
                    "risk_score": 0.0, "error": "FaceLandmarker not initialized"}

        h, w = frame_bgr.shape[:2]
        lm   = _detect_face(self._landmarker, frame_bgr)

        if lm is None:
            return {"face_found": False, "alerts": [], "state": "SAFE", "risk_score": 0.0}

        left_eye_pts  = _get_coords(lm, LEFT_EYE,  w, h)
        right_eye_pts = _get_coords(lm, RIGHT_EYE, w, h)
        mouth_pts     = _get_coords(lm, MOUTH,     w, h)

        ear          = (_compute_ear(left_eye_pts) + _compute_ear(right_eye_pts)) / 2.0
        mar          = _compute_mar(mouth_pts)
        direction, _ = _compute_direction(lm)
        head_pose, _ = _compute_pitch(lm)

        alerts = []
        if ear < EAR_THRESHOLD:
            alerts.append("Eyes Closed")
        if mar > MAR_THRESHOLD:
            alerts.append("Yawning")
        if direction != "FORWARD":
            alerts.append(f"Looking {direction.capitalize()}")
        if head_pose == "LOOKING_DOWN":
            alerts.append("Looking Down")
        elif head_pose == "LOOKING_UP":
            alerts.append("Looking Up")

        state = "HIGH_RISK" if len(alerts) >= 2 else ("WARNING" if alerts else "SAFE")

        return {
            "face_found": True,
            "ear": ear, "mar": mar,
            "direction": direction, "head_pose": head_pose,
            "risk_score": 1.0 if state == "HIGH_RISK" else (0.5 if state == "WARNING" else 0.1),
            "state": state,
            "alerts": alerts,
            "landmarks": lm, "img_w": w, "img_h": h,
        }

    def close(self):
        if self._landmarker:
            self._landmarker.close()


def draw_yolo_boxes(frame, yolo_dets, show_conf=True):
    frame = frame.copy()
    for (xyxy, conf, label) in yolo_dets:
        x1, y1, x2, y2 = map(int, xyxy)
        hex_c = get_color(label).lstrip("#")
        bgr = tuple(int(hex_c[i:i+2], 16) for i in (4, 2, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
        text = f"{label} {conf:.2f}" if show_conf else label
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), bgr, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame


def draw_face_overlay(frame, face_result, show_mesh=False):
    if not face_result.get("face_found"):
        return frame
    frame = frame.copy()
    h, w  = frame.shape[:2]
    lm    = face_result["landmarks"]

    for idx in LEFT_EYE + RIGHT_EYE:
        x = int(lm[idx].x * w)
        y = int(lm[idx].y * h)
        cv2.circle(frame, (x, y), 2, (255, 200, 0), -1)

    for idx in MOUTH:
        x = int(lm[idx].x * w)
        y = int(lm[idx].y * h)
        cv2.circle(frame, (x, y), 2, (200, 100, 255), -1)

    nx = int(lm[NOSE_TIP].x * w)
    ny = int(lm[NOSE_TIP].y * h)
    cv2.circle(frame, (nx, ny), 4, (0, 255, 255), -1)

    direction = face_result.get("direction", "FORWARD")
    head_pose = face_result.get("head_pose", "FORWARD")

    arrow_map = {
        "LEFT":         (-40, 0),
        "RIGHT":        (40, 0),
        "LOOKING_DOWN": (0, 40),
        "LOOKING_UP":   (0, -40),
        "FORWARD":      (0, 0),
    }
    dx, dy = arrow_map.get(direction, (0, 0))
    if head_pose == "LOOKING_DOWN":
        dy = 40
    elif head_pose == "LOOKING_UP":
        dy = -40
    if dx != 0 or dy != 0:
        cv2.arrowedLine(frame, (nx, ny), (nx + dx, ny + dy), (0, 255, 255), 2, tipLength=0.3)

    state = face_result.get("state", "SAFE")
    risk  = face_result.get("risk_score", 0.0)
    state_colors = {"SAFE": (60, 200, 0), "WARNING": (0, 165, 255), "HIGH_RISK": (0, 40, 220)}
    sc    = state_colors.get(state, (60, 200, 0))
    badge = f"{state}  {risk:.2f}"
    (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (w - bw - 20, 8), (w - 4, bh + 14), sc, -1)
    cv2.putText(frame, badge, (w - bw - 14, bh + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame
