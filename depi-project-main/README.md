# Driver Monitoring System — Streamlit App

Dual-model DMS dashboard: behavior detection (11 classes) + seatbelt detection (2 classes).

## Setup

1. Copy your model files into the `models/` folder:
   - `models/behavior.pt` → best__17_.pt (behavior model)
   - `models/seatbelt.pt` → best__16_.pt (seatbelt model)

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Pages

| Page | Description |
|------|-------------|
| 📹 Real-Time | Live webcam feed with frame-skip optimization for low latency |
| 🎞️ Video | Upload and process a video file |
| 🖼️ Image | Upload one or more images for detection |

## Speed Tips (Real-Time)

- Set inference width to 320 or 416 for faster inference
- Frame-skip is set to 2 by default (runs models every other frame)
- Models run with `half=True` (FP16) on GPU for extra speed
