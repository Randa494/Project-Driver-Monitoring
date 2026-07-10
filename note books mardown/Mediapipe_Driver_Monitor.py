
"""
================================================================================
NOTEBOOK 2 — Smart Driver Monitoring System (FULLY FIXED)
MediaPipe Face Mesh | EAR · MAR · Head Direction · Head Pose · Risk Engine
================================================================================

FIXED: When looking down + eyes closed → HIGH RISK (sleeping detection)

Pipeline:
  Camera → MediaPipe Face Mesh → Landmark Extraction
       → EAR → MAR → Head Direction → Head Pose
       → Temporal Analysis (sliding window)
       → Risk Engine (FIXED for head-down)
       → Decision (SAFE / WARNING / HIGH_RISK)
       → Real-time Visualization

Run:
  python notebook_02_mediapipe_driver_monitor.py

Controls (in the OpenCV window):
  Q  →  quit
  S  →  save screenshot
  R  →  reset temporal history
================================================================================
"""

# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — INSTALL DEPENDENCIES
# ══════════════════════════════════════════════════════════════════════════════
# Run once in your terminal before executing this script:
#
#   pip install mediapipe opencv-python numpy
#
# mediapipe ≥ 0.10   →  Face Mesh + drawing utilities
# opencv-python      →  webcam capture + display
# numpy              →  math / vector operations
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — IMPORTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_face_mesh   = mp.solutions.face_mesh
mp_drawing     = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ── Thresholds ───────────────────────────────────────────────────────────────
# EAR: Eye Aspect Ratio
#   Normal open eye  →  EAR ≈ 0.25 – 0.35
#   Closed / drowsy  →  EAR < 0.20
EAR_THRESHOLD        = 0.20    # below this → eyes closing
EAR_CONSEC_FRAMES    = 60       # consecutive frames below threshold → alert

# MAR: Mouth Aspect Ratio
#   Closed mouth     →  MAR ≈ 0.0 – 0.35
#   Yawning          →  MAR > 0.55
MAR_THRESHOLD        = 0.55    # above this → yawning

# Temporal window
WINDOW_SIZE          = 25      # sliding window in frames (~1 sec at 25 fps)

# Head pose: nose deviation from center (normalized 0–1 coords)
HEAD_CENTER_MARGIN   = 0.12    # ±7% from center is still "FORWARD"

# Head pitch (looking down/up) thresholds
HEAD_DOWN_THRESHOLD  = 0.15   # Nose-to-eye vertical offset for head down
HEAD_UP_THRESHOLD    = -0.05   # Negative offset for head up

# Face missing detection
FACE_MISSING_LIMIT   = 15      # Frames without face = high risk

# Risk weights (must sum to 1.0)
W_DROWSINESS         = 0.45
W_YAWNING            = 0.20
W_DISTRACTION        = 0.35

# Risk thresholds → decision
RISK_SAFE            = 0.35
RISK_WARNING         = 0.60


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — LANDMARK INDEX MAPS
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

# LEFT EYE  (6 landmarks for EAR — 2 horizontal, 4 vertical)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
#              P1   P2   P3   P4   P5   P6
#  P1 = left corner, P4 = right corner (horizontal)
#  P2, P3 = upper eyelid | P5, P6 = lower eyelid

# RIGHT EYE
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# MOUTH (6 landmarks for MAR)
MOUTH     = [61,  291, 39,  181, 0,   17]
#             P1   P2   P3   P4  P5   P6
#  P1 = left corner, P2 = right corner (horizontal)
#  P3, P4 = upper lip | P5, P6 = lower lip

# NOSE TIP (for head direction and pitch)
NOSE_TIP  = 1

# LEFT / RIGHT EYE OUTER CORNERS (for head centering reference)
LEFT_EYE_OUTER  = 263
RIGHT_EYE_OUTER = 33

# NOSE BRIDGE (for pitch detection - more stable than tip alone)
NOSE_BRIDGE = 168


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FEATURE EXTRACTION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_landmark_coords(landmarks, indices, frame_w, frame_h):
    """
    Convert normalized MediaPipe landmarks to pixel coordinates.

    Args:
        landmarks  : face_landmarks.landmark (list of NormalizedLandmark)
        indices    : list of landmark indices to extract
        frame_w    : frame width  in pixels
        frame_h    : frame height in pixels

    Returns:
        np.ndarray of shape (N, 2) — pixel (x, y) for each index
    """
    coords = []
    for idx in indices:
        lm = landmarks[idx]
        x  = int(lm.x * frame_w)
        y  = int(lm.y * frame_h)
        coords.append([x, y])
    return np.array(coords, dtype=np.float64)


def compute_ear(eye_landmarks):
    """
    Eye Aspect Ratio (EAR) — Soukupová & Čech, 2016.

    Formula:
        EAR = (||P2-P6|| + ||P3-P5||) / (2 × ||P1-P4||)

    Where:
        P1, P4  →  horizontal corners (left, right)
        P2, P5  →  upper eyelid landmarks
        P3, P6  →  lower eyelid landmarks

    Args:
        eye_landmarks : np.ndarray shape (6, 2)

    Returns:
        float  —  EAR value  (lower = more closed)
    """
    if eye_landmarks is None or len(eye_landmarks) < 6:
        return 0.0
    
    P1, P2, P3, P4, P5, P6 = eye_landmarks

    # Vertical distances
    A = np.linalg.norm(P2 - P6)   # upper–lower  (pair 1)
    B = np.linalg.norm(P3 - P5)   # upper–lower  (pair 2)

    # Horizontal distance
    C = np.linalg.norm(P1 - P4)

    # Avoid division by zero
    if C < 1e-6:
        return 0.0

    ear = (A + B) / (2.0 * C)
    return float(ear)


def compute_mar(mouth_landmarks):
    """
    Mouth Aspect Ratio (MAR) — analogous to EAR for yawning.

    Formula:
        MAR = (||P3-P6|| + ||P4-P5||) / (2 × ||P1-P2||)

    Where:
        P1, P2  →  horizontal corners (left, right)
        P3, P4  →  upper lip landmarks
        P5, P6  →  lower lip landmarks

    Args:
        mouth_landmarks : np.ndarray shape (6, 2)

    Returns:
        float  —  MAR value  (higher = more open / yawning)
    """
    if mouth_landmarks is None or len(mouth_landmarks) < 6:
        return 0.0
    
    P1, P2, P3, P4, P5, P6 = mouth_landmarks

    # Vertical distances (upper–lower lip pairs)
    A = np.linalg.norm(P3 - P6)
    B = np.linalg.norm(P4 - P5)

    # Horizontal distance (mouth width)
    C = np.linalg.norm(P1 - P2)

    if C < 1e-6:
        return 0.0

    mar = (A + B) / (2.0 * C)
    return float(mar)


def compute_head_direction(landmarks, frame_w, frame_h):
    """
    Estimate head direction based on the nose tip position
    relative to the midpoint of the two outer eye corners.

    Logic:
        - Find midpoint X of left/right outer eye corners
        - Compare nose tip X to that midpoint
        - Deviation > HEAD_CENTER_MARGIN  →  LEFT or RIGHT
        - Otherwise                       →  FORWARD

    Args:
        landmarks  : face landmark list
        frame_w    : frame width
        frame_h    : frame height

    Returns:
        direction  : str — 'FORWARD', 'LEFT', or 'RIGHT'
        deviation  : float — signed deviation in normalized coords
    """
    nose    = landmarks[NOSE_TIP]
    left_e  = landmarks[LEFT_EYE_OUTER]
    right_e = landmarks[RIGHT_EYE_OUTER]

    # All in normalized coords (0–1)
    nose_x    = nose.x
    mid_x     = (left_e.x + right_e.x) / 2.0
    deviation = nose_x - mid_x     # positive → nose right of center → looking LEFT

    if deviation > HEAD_CENTER_MARGIN:
        direction = 'RIGHT'
    elif deviation < -HEAD_CENTER_MARGIN:
        direction = 'LEFT'
    else:
        direction = 'FORWARD'

    return direction, float(deviation)


def compute_head_pitch(landmarks):
    """
    Detect head pitch (looking down/up) using nose and eye Y positions.
    
    When driver looks down (sleeping position), nose moves DOWN relative to eyes.
    
    Returns:
        pitch: 'LOOKING_DOWN', 'LOOKING_UP', or 'FORWARD'
        offset: vertical offset value
    """
    # Get key points
    nose_y = landmarks[NOSE_TIP].y
    
    # Eye center Y (average of both eyes)
    left_eye_y = np.mean([landmarks[idx].y for idx in LEFT_EYE[:3]])
    right_eye_y = np.mean([landmarks[idx].y for idx in RIGHT_EYE[:3]])
    eye_center_y = (left_eye_y + right_eye_y) / 2
    
    # Calculate vertical offset (nose relative to eyes)
    vertical_offset = nose_y - eye_center_y

# Ignore small head movements
    if abs(vertical_offset) < 0.03:
        return 'FORWARD', vertical_offset

    if vertical_offset > HEAD_DOWN_THRESHOLD:
        return 'LOOKING_DOWN', vertical_offset
    elif vertical_offset < HEAD_UP_THRESHOLD:
        return 'LOOKING_UP', vertical_offset
    else:
        return 'FORWARD', vertical_offset


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — TEMPORAL ANALYSIS (SLIDING WINDOW)
# ══════════════════════════════════════════════════════════════════════════════

class TemporalAnalyzer:
    """
    Maintains a sliding window of recent feature values.
    Smooths out noisy per-frame readings.
    
    Tracks head pose and face missing status.
    """

    def __init__(self, window_size=WINDOW_SIZE):
        self.window_size       = window_size
        self.ear_history       = deque(maxlen=window_size)
        self.mar_history       = deque(maxlen=window_size)
        self.dir_history       = deque(maxlen=window_size)
        self.head_pose_history = deque(maxlen=window_size)
        self.ear_closed_streak = 0
        self.head_down_streak = 0
        self.face_missing_frames = 0

    def update(self, ear, mar, direction, head_pose='FORWARD', face_detected=True):
        """Push new frame values into the sliding window."""
        self.ear_history.append(ear)
        self.mar_history.append(mar)
        self.dir_history.append(direction)
        self.head_pose_history.append(head_pose)

        # Track consecutive closed-eye frames
        if ear < EAR_THRESHOLD:
            self.ear_closed_streak += 1
        else:
            self.ear_closed_streak = 0

        # Track head down duration
        if head_pose == 'LOOKING_DOWN':
            self.head_down_streak += 1
        else:
            self.head_down_streak = 0
        
        # Track face missing frames
        if face_detected:
            self.face_missing_frames = 0
        else:
            self.face_missing_frames += 1

    def reset(self):
        """Clear all history (press R in the window)."""
        self.ear_history.clear()
        self.mar_history.clear()
        self.dir_history.clear()
        self.head_pose_history.clear()
        self.ear_closed_streak = 0
        self.head_down_streak = 0
        self.face_missing_frames = 0
    @property
    def avg_ear(self):
        return float(np.mean(self.ear_history)) if self.ear_history else 0.30

    @property
    def avg_mar(self):
        return float(np.mean(self.mar_history)) if self.mar_history else 0.0

    @property
    def is_drowsy(self):
        """True if eyes closed long enough to indicate drowsiness."""
        return self.ear_closed_streak >= EAR_CONSEC_FRAMES

    @property
    def is_yawning(self):
        """True if average MAR exceeds yawn threshold."""
        return self.avg_mar > MAR_THRESHOLD

    @property
    def is_distracted(self):
        """
        True if majority of recent frames show non-forward attention.
        Uses a 60% majority vote over the window.
        """
        if not self.dir_history:
            return False
        non_forward = sum(1 for d in self.dir_history if d != 'FORWARD')
        return (non_forward / len(self.dir_history)) >= 0.75

    @property
    def current_direction(self):
        """Most recent head direction."""
        return self.dir_history[-1] if self.dir_history else 'FORWARD'

    @property
    def is_head_down(self):
        """
        Head must stay down for a period
        before considering sleeping.
        """
        return self.head_down_streak >= 45
    @property
    def is_head_up(self):
        """Check if driver is looking up"""
        if not self.head_pose_history:
            return False
        up_frames = sum(1 for p in self.head_pose_history if p == 'LOOKING_UP')
        return (up_frames / len(self.head_pose_history)) >= 0.5

    @property
    def is_face_missing(self):
        """Check if face has been missing for too long"""
        return self.face_missing_frames > FACE_MISSING_LIMIT

    @property
    def current_head_pose(self):
        """Most recent head pose"""
        return self.head_pose_history[-1] if self.head_pose_history else 'FORWARD'


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — RISK ENGINE (FIXED FOR HEAD-DOWN DETECTION)
# ══════════════════════════════════════════════════════════════════════════════

class RiskEngine:
    """
    Combines temporal signals into a weighted risk score (0.0 – 1.0).
    
    FIXED: When head is down, automatically triggers HIGH RISK (sleeping position)
    """

    def compute(self, analyzer: TemporalAnalyzer):
        """
        Args:
            analyzer : TemporalAnalyzer with updated history

        Returns:
            risk_score : float in [0, 1]
            components : dict with individual signal values
        """

        # ── Drowsiness signal (FIXED for head-down detection) ─────────────────
        drowsiness_signal = 0.0

        # CASE 1: Head is down (looking down = sleeping position)
        if analyzer.is_head_down:
            # When head is down, driver is likely sleeping or about to sleep
            drowsiness_signal = 0.85  # Direct high risk
            
            # Additional: if eyes are also closed (detectable), even higher risk
            if analyzer.avg_ear < EAR_THRESHOLD:
                drowsiness_signal = 0.95
                print("⚠️ CRITICAL: Head down + eyes closed = SLEEPING!")
            else:
                print("⚠️ HEAD DOWN DETECTED - Sleeping position!")
        
        # CASE 2: Normal head position - use EAR-based detection
        else:
            ear_norm = max(0.0, min(1.0, 1.0 - (analyzer.avg_ear / EAR_THRESHOLD)))
            streak_boost = 0.3 if analyzer.is_drowsy else 0.0
            drowsiness_signal = min(1.0, ear_norm + streak_boost)
        
        # CASE 3: Face missing (head dropped out of frame)
        if analyzer.is_face_missing:
            drowsiness_signal = max(drowsiness_signal, 0.90)
            print("⚠️ DRIVER NOT VISIBLE - High risk!")

        # ── Yawning signal ───────────────────────────────────────────────────
        if analyzer.avg_mar > MAR_THRESHOLD:
            yawning_signal = min(1.0, (analyzer.avg_mar - MAR_THRESHOLD) / (1.0 - MAR_THRESHOLD))
        else:
            yawning_signal = 0.0

        # ── Distraction signal ───────────────────────────────────────────────
        if analyzer.dir_history:
            non_fwd = sum(1 for d in analyzer.dir_history if d != 'FORWARD')
            distraction_signal = non_fwd / len(analyzer.dir_history)
        else:
            distraction_signal = 0.0

        # ── Weighted risk score ──────────────────────────────────────────────
        risk_score = (
            W_DROWSINESS  * drowsiness_signal +
            W_YAWNING     * yawning_signal    +
            W_DISTRACTION * distraction_signal
        )
        risk_score = float(np.clip(risk_score, 0.0, 1.0))

        components = {
            'drowsiness' : round(drowsiness_signal, 3),
            'yawning'    : round(yawning_signal, 3),
            'distraction': round(distraction_signal, 3),
            'head_down'  : 1.0 if analyzer.is_head_down else 0.0,
            'face_missing': 1.0 if analyzer.is_face_missing else 0.0
        }

        return risk_score, components

    def decide(self, risk_score):
        """
        Map risk score to a human-readable state.

        Returns:
            state : str — 'SAFE', 'WARNING', or 'HIGH_RISK'
        """
        if risk_score < RISK_SAFE:
            return 'SAFE'
        elif risk_score < RISK_WARNING:
            return 'WARNING'
        else:
            return 'HIGH_RISK'


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

# Colors — BGR format for OpenCV
COLOR = {
    'SAFE'      : (0,   200,  60),    # green
    'WARNING'   : (0,   165, 255),    # orange
    'HIGH_RISK' : (0,    40, 220),    # red
    'white'     : (255, 255, 255),
    'gray'      : (160, 160, 160),
    'dark'      : ( 20,  20,  20),
    'panel_bg'  : ( 30,  30,  30),
    'eye_dot'   : (255, 200,   0),    # yellow for eye landmarks
    'mouth_dot' : (200, 100, 255),    # purple for mouth landmarks
}


def draw_landmarks_minimal(frame, face_landmarks, frame_w, frame_h):
    """
    Draw only the key landmarks we use (eyes + mouth) — clean and fast.
    Avoids drawing all 468 points which clutters the display.
    """
    # Eye landmarks
    for idx in LEFT_EYE + RIGHT_EYE:
        lm = face_landmarks.landmark[idx]
        x  = int(lm.x * frame_w)
        y  = int(lm.y * frame_h)
        cv2.circle(frame, (x, y), 2, COLOR['eye_dot'], -1)

    # Connect eye outline
    for eye_indices in [LEFT_EYE, RIGHT_EYE]:
        pts = []
        for idx in eye_indices:
            lm = face_landmarks.landmark[idx]
            pts.append((int(lm.x * frame_w), int(lm.y * frame_h)))
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True,
                      color=COLOR['eye_dot'], thickness=1)

    # Mouth landmarks
    for idx in MOUTH:
        lm = face_landmarks.landmark[idx]
        x  = int(lm.x * frame_w)
        y  = int(lm.y * frame_h)
        cv2.circle(frame, (x, y), 2, COLOR['mouth_dot'], -1)

    # Nose tip
    nose = face_landmarks.landmark[NOSE_TIP]
    nx   = int(nose.x * frame_w)
    ny   = int(nose.y * frame_h)
    cv2.circle(frame, (nx, ny), 4, (0, 255, 255), -1)


def draw_hud(frame, ear, mar, direction, head_pose, risk_score, state,
             components, analyzer, fps):
    """
    Draw the heads-up display (HUD) panel on the frame.
    Shows head pose status.
    """
    h, w = frame.shape[:2]

    state_color = COLOR[state]

    # Semi-transparent left panel
    panel_w = 240
    overlay  = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, h), COLOR['panel_bg'], -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Panel title
    cv2.putText(frame, 'DRIVER MONITOR',
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR['white'], 1)
    cv2.line(frame, (10, 36), (panel_w - 10, 36), COLOR['gray'], 1)

    def metric_row(y, label, value, value_color=COLOR['white']):
        cv2.putText(frame, label, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR['gray'], 1)
        cv2.putText(frame, value, (120, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, value_color, 1)

    # EAR
    ear_color = COLOR['HIGH_RISK'] if ear < EAR_THRESHOLD else COLOR['SAFE']
    metric_row(65,  'EAR',       f'{ear:.3f}',  ear_color)

    # MAR
    mar_color = COLOR['WARNING'] if mar > MAR_THRESHOLD else COLOR['SAFE']
    metric_row(90,  'MAR',       f'{mar:.3f}',  mar_color)

    # Head direction (left/right)
    dir_color = COLOR['SAFE'] if direction == 'FORWARD' else COLOR['WARNING']
    metric_row(115, 'DIRECTION', direction,      dir_color)

    # Head pose (up/down)
    pose_color = COLOR['HIGH_RISK'] if head_pose == 'LOOKING_DOWN' else COLOR['SAFE']
    metric_row(140, 'HEAD POSE',  head_pose,     pose_color)

    # Streak
    streak_color = COLOR['HIGH_RISK'] if analyzer.is_drowsy else COLOR['gray']
    metric_row(165, 'EYE STREAK', f'{analyzer.ear_closed_streak} fr', streak_color)

    # FPS
    metric_row(190, 'FPS',       f'{fps:.0f}',  COLOR['gray'])

    # Divider
    cv2.line(frame, (10, 205), (panel_w - 10, 205), COLOR['gray'], 1)

    # Sub-signals
    cv2.putText(frame, 'SIGNALS', (10, 225),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR['gray'], 1)

    def signal_bar(y, label, value):
        """Draw a mini bar for each sub-signal."""
        cv2.putText(frame, label, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR['gray'], 1)
        bar_x   = 90
        bar_w   = int(value * (panel_w - bar_x - 14))
        bar_h   = 8
        bar_y   = y - bar_h + 1

        cv2.rectangle(frame, (bar_x, bar_y),
                      (panel_w - 14, bar_y + bar_h), (60, 60, 60), -1)

        fill_color = _risk_color(value)
        if bar_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_h), fill_color, -1)

        cv2.putText(frame, f'{value:.2f}',
                    (panel_w - 40, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR['white'], 1)

    signal_bar(247, 'Drowsy', components['drowsiness'])
    signal_bar(267, 'Yawn',   components['yawning'])
    signal_bar(287, 'Distract', components['distraction'])

    # State badge (top right)
    badge_text = state
    (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    bx = w - bw - 24
    by = 14
    cv2.rectangle(frame, (bx - 8, by - bh - 6), (bx + bw + 8, by + 6),
                  state_color, -1)
    cv2.putText(frame, badge_text, (bx, by),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR['white'], 2)

    # Risk score bar (bottom)
    bar_y0     = h - 38
    bar_y1     = h - 18
    bar_x0     = panel_w + 10
    bar_x1     = w - 10
    bar_total  = bar_x1 - bar_x0

    cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x1, bar_y1), (60, 60, 60), -1)

    fill_end = bar_x0 + int(risk_score * bar_total)
    fill_c   = _risk_color(risk_score)
    if fill_end > bar_x0:
        cv2.rectangle(frame, (bar_x0, bar_y0), (fill_end, bar_y1), fill_c, -1)

    for thresh, label in [(RISK_SAFE, 'SAFE'), (RISK_WARNING, 'WARN')]:
        mx = bar_x0 + int(thresh * bar_total)
        cv2.line(frame, (mx, bar_y0 - 4), (mx, bar_y1 + 4), COLOR['white'], 1)

    cv2.putText(frame, f'RISK  {risk_score:.2f}',
                (bar_x0, bar_y0 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR['white'], 1)

    # Alerts (center overlay)
    if state == 'HIGH_RISK':
        _draw_alert_banner(frame, '⚠  HIGH RISK — PULL OVER', COLOR['HIGH_RISK'])
    elif state == 'WARNING':
        _draw_alert_banner(frame, '!  STAY ALERT', COLOR['WARNING'])
    
    # Head down alert
    if head_pose == 'LOOKING_DOWN' and state != 'HIGH_RISK':
        _draw_alert_banner(frame, '⚠  LOOKING DOWN', COLOR['WARNING'])

    return frame


def _risk_color(value):
    """Interpolate green → orange → red based on risk value 0–1."""
    if value < 0.5:
        r = int(value * 2 * 165)
        g = 200
    else:
        r = 165 + int((value - 0.5) * 2 * (220 - 165))
        g = int((1.0 - (value - 0.5) * 2) * 200)
    return (0, g, r)   # BGR


def _draw_alert_banner(frame, text, color):
    """Draw a centered alert banner near the top of the frame."""
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.80, 2)
    x = (w - tw) // 2
    y = 70
    cv2.putText(frame, text, (x + 2, y + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80, COLOR['dark'], 2)
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80, color, 2)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — NO-FACE FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

def draw_no_face(frame, missing_frames=0):
    """Display a placeholder when no face is detected."""
    h, w = frame.shape[:2]
    cv2.putText(frame, 'NO FACE DETECTED',
                (w // 2 - 140, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR['WARNING'], 2)
    cv2.putText(frame, 'Position your face in front of the camera',
                (w // 2 - 200, h // 2 + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR['gray'], 1)
    
    if missing_frames > 5:
        cv2.putText(frame, f'Face missing for {missing_frames} frames',
                    (w // 2 - 180, h // 2 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR['HIGH_RISK'], 1)
    return frame


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run():
    """
    Main real-time driver monitoring loop.
    """

    print('=' * 60)
    print('  Smart Driver Monitoring System — Notebook 2 (FIXED)')
    print('  MediaPipe Face Mesh | EAR · MAR · Head Direction · Head Pose')
    print('=' * 60)
    print('  FIXED: Head down + eyes closed → HIGH RISK (sleeping)')
    print('=' * 60)
    print('  Q → quit    S → screenshot    R → reset history')
    print('=' * 60)

    # Initialize MediaPipe Face Mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces       = 1,
        refine_landmarks    = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5,
    )

    # Initialize components
    analyzer    = TemporalAnalyzer(window_size=WINDOW_SIZE)
    risk_engine = RiskEngine()

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(' Cannot open webcam. Check camera connection.')
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_count  = 0
    prev_time    = time.time()
    screenshot_n = 0

    print('Webcam started. Processing...\n')
    print(' TEST: Look down + close eyes → Should trigger HIGH RISK!\n')

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('⚠  Frame grab failed. Exiting.')
                break

            frame_count += 1

            # FPS calculation
            now      = time.time()
            fps      = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            # MediaPipe inference
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results  = face_mesh.process(rgb)
            rgb.flags.writeable = True

            # Process detected face
            if results.multi_face_landmarks:
                face_lm = results.multi_face_landmarks[0].landmark

                # Extract coordinates
                left_eye_pts  = get_landmark_coords(face_lm, LEFT_EYE,  w, h)
                right_eye_pts = get_landmark_coords(face_lm, RIGHT_EYE, w, h)
                mouth_pts     = get_landmark_coords(face_lm, MOUTH,     w, h)

                # Compute features
                ear_left   = compute_ear(left_eye_pts)
                ear_right  = compute_ear(right_eye_pts)
                ear        = (ear_left + ear_right) / 2.0

                mar        = compute_mar(mouth_pts)

                direction, _ = compute_head_direction(face_lm, w, h)
                head_pose, _ = compute_head_pitch(face_lm)

                # Update temporal window
                analyzer.update(ear, mar, direction, head_pose, face_detected=True)

                # Risk score
                risk_score, components = risk_engine.compute(analyzer)
                state = risk_engine.decide(risk_score)

                # Draw landmarks
                draw_landmarks_minimal(frame, results.multi_face_landmarks[0], w, h)

                # Draw HUD
                draw_hud(frame, ear, mar, direction, head_pose, risk_score, 
                        state, components, analyzer, fps)

                # Additional visual alert for head down + sleeping
                if head_pose == 'LOOKING_DOWN' and ear < EAR_THRESHOLD:
                    h_frame, w_frame = frame.shape[:2]
                    cv2.rectangle(frame, (w_frame//2 - 200, h_frame//2 - 50),
                                (w_frame//2 + 200, h_frame//2 + 50), (0, 0, 255), -1)
                    cv2.putText(frame, " SLEEPING DETECTED! ",
                               (w_frame//2 - 170, h_frame//2 + 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)

                # Console output
                if frame_count % 15 == 0:
                    print(
                        f'Frame {frame_count:05d} | '
                        f'EAR={ear:.3f}  MAR={mar:.3f}  '
                        f'DIR={direction:<8}  POSE={head_pose:<10} '
                        f'RISK={risk_score:.2f}  STATE={state}'
                    )

            else:
                # No face found
                analyzer.update(0.30, 0.12, 'FORWARD', 'FORWARD', face_detected=False)
                risk_score, components = risk_engine.compute(analyzer)
                state = risk_engine.decide(risk_score)
                
                draw_no_face(frame, analyzer.face_missing_frames)

                cv2.putText(frame, f'RISK: {risk_score:.2f} - {state}',
                            (w // 2 - 100, h - 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, COLOR['HIGH_RISK'] if risk_score > 0.5 else COLOR['WARNING'], 2)
                cv2.putText(frame, f'FPS {fps:.0f}',
                            (w - 90, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, COLOR['gray'], 1)
                
                if analyzer.is_face_missing:
                    _draw_alert_banner(frame, ' DRIVER NOT VISIBLE', COLOR['HIGH_RISK'])

            # Show window
            cv2.imshow('Driver Monitor — Notebook 2 (Fixed)', frame)

            # Key handling
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print('\n🛑 Q pressed — stopping.')
                break

            elif key == ord('s'):
                screenshot_n += 1
                fname = f'screenshot_{screenshot_n:03d}.jpg'
                cv2.imwrite(fname, frame)
                print(f'💾 Screenshot saved: {fname}')

            elif key == ord('r'):
                analyzer.reset()
                print('History reset.')

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()
        print('\nSession ended cleanly.')


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    run()