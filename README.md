# picar-autonomous

An end-to-end behavioral cloning system for an autonomous RC car. A custom CNN is trained on manually driven data collected via a Raspberry Pi, then deployed back to the Pi for real-time inference. A Flutter mobile app provides live camera feed, manual joystick control, and one-tap autonomous mode toggle.

---

## Repository Structure

```
picar-autonomous/
├── pi/                     # Code that runs on the Raspberry Pi
│   ├── collect_data.py     # Manual driving data collection (keyboard + camera)
│   └── server.py           # Inference server (TCP control, MJPEG stream, telemetry)
│
├── workstation/            # ML training pipeline (runs on dev machine)
│   ├── model.py            # CNN architecture (DrivingCNN)
│   ├── train.py            # Training loop with augmentation + early stopping
│   ├── export.py           # Quantize + TorchScript export → model.pt
│   ├── clean_dataset.py    # Dataset filtering utility
│   └── show_data.py        # Steering distribution visualizer
│
├── app/                    # Flutter mobile app (Android / iOS)
│   └── lib/main.dart       # App source (manual drive + autonomous mode)
│
├── dataset/                # Training data
│   ├── labels.csv          # Steering/throttle labels (~20K rows)
│   └── images/             # Captured frames — gitignored, store externally
│
├── models/                 # Trained model outputs — gitignored, regenerate via pipeline
│
├── docs/                   # Documentation and diagrams
│   ├── diagrams.ipynb      # Generates standard diagrams
│   ├── diagrams_ieee.ipynb # Generates IEEE-formatted diagrams
│   ├── behavioral_cloning_pipeline_highres.pdf
│   ├── behavioral_cloning_pipeline_highres.png
│   └── exports/
│       ├── png/            # PNG diagram exports (16 standard + 2 IEEE variants)
│       └── tiff/           # TIFF diagram exports (same set, publication format)
│
├── WORKFLOW.md             # Step-by-step training and deployment workflow
├── .gitignore
└── README.md
```

---

## System Overview

```
[Raspberry Pi]  ──WiFi AP (192.168.4.1)──  [Mobile App]
     │
     ├─ Port 5005  ← control commands (TCP, JSON)
     ├─ Port 5006  → telemetry (TCP, JSON)
     └─ Port 8080  → live MJPEG camera stream (HTTP)
```

The Pi runs as a WiFi access point. The mobile app connects directly to it without internet.

---

## Prerequisites

### Raspberry Pi
- Raspberry Pi (tested on Pi 4)
- Pi Camera Module
- 4× HC-SR04 ultrasonic sensors (front / left / right / rear)
- GPIO motor driver
- Python 3.9+

```
pip install picamera2 opencv-python RPi.GPIO keyboard numpy
```

### Workstation (training)
- Python 3.9+
- PyTorch 2.x
- CUDA optional (CPU training works but is slow)

```
pip install torch torchvision albumentations scikit-learn pandas opencv-python matplotlib
```

### Mobile App
- Flutter SDK >=3.0.0
- Android SDK (for Android builds) or Xcode (for iOS builds)

```
cd app && flutter pub get
```

---

## Workflow

See [WORKFLOW.md](WORKFLOW.md) for the full step-by-step pipeline:

1. Collect data on the Pi → `pi/collect_data.py`
2. Transfer dataset to workstation via SCP
3. Clean dataset → `workstation/clean_dataset.py`
4. Train model → `workstation/train.py`
5. Export model → `workstation/export.py` (produces `models/model.pt`)
6. Deploy `model.pt` to the Pi
7. Run inference server on Pi → `pi/server.py`
8. Connect mobile app and drive

---

## Model

**DrivingCNN** — NVIDIA-style end-to-end convolutional network.

- Input: 3×224×224 RGB frame
- 5× Conv2d layers with ReLU
- 2× Fully connected layers with Dropout(0.5)
- Output: `[steering, throttle]` (both normalized to [-1, 1] and [0, 1])

Exported as dynamic INT8-quantized TorchScript for fast inference on the Pi.

---

## Notes

- `dataset/images/` is gitignored due to size (~20K frames). Store externally or use a data versioning tool (e.g., DVC).
- Model binaries (`*.pt`, `*.pth`) are gitignored. Run the training pipeline to regenerate them.
- The Pi IP `192.168.4.1` is hardcoded in `app/lib/main.dart` — update if using a different network setup.
