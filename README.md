# SVD Gatekeeper

Real-time video novelty detection using SVD reconstruction error with AI object detection.

![Layout](https://img.shields.io/badge/Resolution-1920x1080-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![OpenCV](https://img.shields.io/badge/OpenCV-Required-orange)

## What It Does

Monitors a live video feed and detects **novel events** (changes in the scene) using Singular Value Decomposition (SVD). When something new appears, it triggers YOLO AI detection to identify objects.

**Key Concept:** Instead of running expensive AI on every frame, SVD acts as a "gatekeeper" - only running AI when something changes.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure video source** in `main.py`:
   ```python
   VIDEO_URL = "http://192.168.1.61:8080/video"  # Your IP camera URL
   ```

3. **Run:**
   ```bash
   python main.py
   ```

## Controls

| Control | Action |
|---------|--------|
| **Rank slider** | Adjust SVD compression (higher = more detail) |
| **Threshold slider** | Adjust novelty sensitivity |
| **F key** | Toggle fullscreen |
| **ESC** | Exit |
| **RECONNECT button** | Retry video connection (appears when disconnected) |

## How SVD Gatekeeper Works

```
Live Frame → Grayscale → SVD Decomposition → Reconstruction
                                                    ↓
                                            Error = ||Original - Reconstructed||
                                                    ↓
                                            Error changed significantly?
                                                    ↓
                                    YES → NOVELTY! → Run YOLO AI Detection
                                    NO  → Skip AI (save processing)
```

## UI Layout

```
┌─────────────────────────────┬──────────────┐
│   ORIGINAL FRAME            │  STATUS      │
│                             ├──────────────┤
├─────────────────────────────┤  METRICS     │
│   SVD RECONSTRUCTED         ├──────────────┤
│                             │  RANK SLIDER │
│                             ├──────────────┤
│                             │  THRESHOLD   │
│                             ├──────────────┤
│                             │  DETECTED    │
│                             │  OBJECTS     │
└─────────────────────────────┴──────────────┘
```

## Configuration

Edit the top of `main.py`:

```python
# Video source
VIDEO_URL = "http://192.168.1.61:8080/video"

# Initial parameters
INITIAL_RANK = 20          # SVD compression rank
INITIAL_THRESHOLD = 5      # Novelty threshold (÷100)

# Colors (BGR format)
COLOR_PRIMARY = (144, 119, 4)      # Panel backgrounds
COLOR_SECONDARY = (221, 233, 238)  # Main background
COLOR_ACCENT = (246, 252, 255)     # Highlights
```

## Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy
- Ultralytics (YOLO)

## Project Structure

```
├── main.py           # Main application
├── src/core/
│   ├── svd_detector.py    # SVD novelty detection
│   ├── ai_detector.py     # YOLO wrapper
│   └── preprocessor.py    # Frame preprocessing
├── requirements.txt
└── yolov8n.pt        # YOLO model weights
```

## License

See LICENSE file.