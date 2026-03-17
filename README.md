# picar-autonomous

An end-to-end behavioral cloning system for an autonomous RC car. A custom CNN is trained on manually driven data collected via a Raspberry Pi, then deployed back to the Pi for real-time inference. A Flutter mobile app provides live camera feed, manual joystick control, and one-tap autonomous mode toggle.

---

## Repository Structure

```
picar-autonomous/
├── pi/                       # Code that runs on the Raspberry Pi
│   ├── collect_data.py       # Manual driving data collection (keyboard + camera)
│   ├── drive_server.py       # Inference + control server (TCP, MJPEG, telemetry)
│   └── requirements.txt      # Pi Python dependencies
│
├── workstation/              # ML training pipeline (runs on dev machine)
│   ├── model.py              # CNN architecture (DrivingCNN)
│   ├── train.py              # Training loop with augmentation + early stopping
│   ├── export.py             # Quantize + TorchScript export → model.pt
│   ├── clean_dataset.py      # Dataset filtering utility
│   ├── show_data.py          # Steering distribution visualizer
│   └── requirements.txt      # Workstation Python dependencies
│
├── app/                      # Flutter mobile app (Android / iOS)
│   └── lib/main.dart         # App source (manual drive + autonomous mode)
│
├── dataset/                  # Training data
│   ├── labels.csv            # Steering/throttle labels (~20K rows)
│   └── images/               # Captured frames — gitignored, store externally
│
├── models/                   # Trained model outputs — gitignored, regenerate via pipeline
│
├── docs/                     # Documentation and diagrams
│   ├── diagrams.ipynb        # Generates standard diagrams
│   ├── diagrams_ieee.ipynb   # Generates IEEE-formatted diagrams
│   ├── behavioral_cloning_pipeline_highres.pdf
│   ├── behavioral_cloning_pipeline_highres.png
│   └── exports/
│       └── png/              # PNG diagram exports at 600 DPI (16 standard + 2 IEEE variants)
│
├── WORKFLOW.md               # Step-by-step training and deployment workflow
├── .gitignore
└── README.md
```

---

## System Overview

```
[Raspberry Pi]  ──WiFi AP (192.168.4.1)──  [Mobile App]
     │
     ├─ Port 5005  ← control commands  (TCP, newline-delimited JSON)
     ├─ Port 5006  → telemetry         (TCP, newline-delimited JSON)
     └─ Port 8080  → live camera feed  (HTTP MJPEG stream)
```

The Pi runs as a WiFi access point. The mobile app connects directly to it — no internet required.

---

## Prerequisites

### Raspberry Pi
- Raspberry Pi 4B
- Pi Camera Module 3
- 4× HC-SR04 ultrasonic sensors (front / left / right / rear)
- L298N motor driver + 4× DC motors
- Python 3.9+

```bash
pip install -r pi/requirements.txt
```

### Workstation (training)
- Python 3.9+
- PyTorch 2.x
- CUDA optional (CPU training works but is slow)

```bash
pip install -r workstation/requirements.txt
```

### Mobile App
- Flutter SDK ≥ 3.0.0
- Android SDK (for Android builds) or Xcode (for iOS builds)

```bash
cd app && flutter pub get
```

---

## Workflow

See [WORKFLOW.md](WORKFLOW.md) for the full step-by-step pipeline with commands.

1. **Collect data** on the Pi → `pi/collect_data.py`
2. **Transfer dataset** to workstation via SCP
3. **Clean dataset** → `workstation/clean_dataset.py`
4. **Train model** → `workstation/train.py`
5. **Export model** → `workstation/export.py` (produces `model.pt`)
6. **Deploy `model.pt`** to the Pi via SCP
7. **Start server** on Pi → `pi/drive_server.py`
8. **Connect** the mobile app and drive

---

## Model

**DrivingCNN** — NVIDIA PilotNet-style end-to-end convolutional network.

- Input: 3×224×224 RGB frame
- 5× Conv2d layers (24→36→48→64→64 filters) with ReLU
- 2× Fully connected layers (→100→50) with Dropout(0.5)
- Output: `[steering, throttle]`

Exported as dynamic INT8-quantized TorchScript for fast CPU inference on the Pi.

In autonomous mode the server uses **model steering + fixed throttle (0.5)**. Throttle prediction is intentionally disabled — human throttle during data collection is too inconsistent to train reliably, and a fixed moderate speed is safer.

---

## Network Protocol

### Control (TCP :5005) — app → Pi
```json
{"steering": 0.25, "throttle": 0.5, "mode": "manual"}
```
- `steering`: float in [-1.0, 1.0]
- `throttle`: float in [0.0, 1.0]
- `mode`: `"manual"` | `"collect"` | `"autonomous"`

### Telemetry (TCP :5006) — Pi → app
```json
{
  "fps": 19.8,
  "steering": 0.12,
  "throttle": 0.50,
  "mode": "autonomous",
  "obstacle": false,
  "sensors": {"front": 82.3, "left": 120.0, "right": 95.5, "rear": 999.0}
}
```
- Sensor values in cm; 999.0 = no obstacle detected (timeout)

### Camera stream (HTTP :8080)
- `GET /stream` — MJPEG stream consumed by the Flutter app
- `GET /` — Browser-viewable test page (useful for verifying the stream without the app)

---

## Notes

- `dataset/images/` is gitignored due to size (~20K frames). Store externally or use a data versioning tool (e.g. DVC).
- Model binaries (`*.pt`, `*.pth`) are gitignored. Run the training pipeline to regenerate them.
- The Pi IP `192.168.4.1` is hardcoded in `app/lib/main.dart` — update if using a different network setup.
- `drive_server.py` must be run from the directory where `model.pt` lives (typically `~/` on the Pi).
