**picar-autonomous** is a complete, self-contained **end-to-end behavioral cloning** (imitation learning) pipeline for turning a standard Raspberry Pi-based RC car (SunFounder PiCar-style chassis with L298N driver) into a fully autonomous vehicle that can be controlled and monitored from a Flutter mobile app.  

The core idea is classic **NVIDIA PilotNet-style** behavioral cloning:  
- You manually drive the car while the Pi records synchronized camera frames + steering/throttle labels.  
- A custom CNN learns to map raw RGB images directly to steering commands (throttle is deliberately **not** predicted in autonomous mode).  
- The quantized TorchScript model is deployed back to the Pi for real-time inference (~20 Hz control loop).  
- A lightweight Flutter app acts as the “remote cockpit”: live MJPEG video feed, virtual joystick for manual drive / data collection, one-tap autonomous toggle, and real-time telemetry (FPS, sensors, state).  

Everything runs **offline** on a private Wi-Fi network where the Pi itself is the access point (IP `192.168.4.1` hardcoded in the app). No cloud, no internet, no ROS — pure Python + Flutter + bare-metal GPIO.

### 1. Purpose & Design Philosophy
**Why behavioral cloning?**  
It is the simplest way to get a vision-based autonomous car working on modest hardware (Pi 4B CPU only, no GPU, no Coral TPU). You don’t need reinforcement learning, simulation, or complex perception stacks. The model directly mimics human (your) driving behavior from raw pixels.  

**Key design decisions (and their trade-offs):**  
- **Throttle is fixed at 0.5 in autonomous mode.** Human throttle data is too noisy/inconsistent (people feather the throttle, brake unpredictably). Training on it leads to unstable speed. Fixed moderate speed + differential steering is safer and more reliable.  
- **Simple ultrasonic obstacle avoidance** layered on top of the CNN (not learned). The model predicts steering assuming clear path; if any sensor detects < ~30–40 cm obstacle, the server falls back to a hard-coded “reverse + turn-in-place” maneuver. This is a pragmatic hybrid: learned lateral control + rule-based longitudinal/safety layer.  
- **INT8 dynamic quantization + TorchScript** → runs comfortably on Pi 4B CPU at ~15–20 FPS inference + control. Full FP32 would be too slow.  
- **No end-to-end lane following or traffic rules** — it learns whatever driving style you demonstrate in your data (e.g., stay in the middle of a corridor, follow a line, avoid walls). Generalization depends entirely on the diversity of your ~20 k frame dataset.

**Target use cases**  
- Educational / hobbyist robotics projects  
- Quick prototyping of vision-based control  
- Indoor or controlled outdoor tracks (not public roads — safety note below)

**Limitations / edge cases**  
- No semantic understanding (stop signs, pedestrians, etc.).  
- Brittle to lighting changes, different floors, or unseen obstacles unless you collect diverse data.  
- Sensor fusion is minimal (four HC-SR04 only, no IMU, no wheel odometry).  
- Data collection is manual keyboard-only on the Pi (no app joystick during collection — you have to be physically near or SSH in).

### 2. Hardware Stack (Required)
- **Raspberry Pi 4B** (Python 3.9+)  
- **Pi Camera Module 3** (official, not V1/V2)  
- **4× HC-SR04** ultrasonic sensors (front/left/right/rear)  
- **L298N** dual H-bridge motor driver + 4× DC geared motors (standard PiCar-X / PiCar-V2 chassis)  
- Power: separate 18650 or LiPo battery pack for motors (Pi powered via USB-C)  
- MicroSD card with Raspberry Pi OS (Lite recommended) + Pi configured as Wi-Fi AP (`hostapd` + `dnsmasq`)

### 3. Software Stack & Dependencies
The repo is deliberately split into three isolated environments so you can keep the Pi lean and do heavy ML on a workstation (laptop/desktop with or without CUDA).

**Pi side** (`pi/requirements.txt`) — runs on the embedded device:
```
RPi.GPIO>=0.7.1
picamera2>=0.3.12
torch>=2.0.0          # CPU-only build
torchvision>=0.15.0
Pillow>=9.0.0
numpy>=1.23.0
opencv-python>=4.7.0
keyboard>=0.13.5
```
- `picamera2` for zero-copy RGB capture  
- `RPi.GPIO` + software PWM (1 kHz) for motor control  
- TorchScript runtime (no training)

**Workstation side** (`workstation/requirements.txt`) — training pipeline:
```
torch>=2.0.0
torchvision>=0.15.0
albumentations>=1.3.0     # strong augmentations
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
opencv-python>=4.7.0
matplotlib>=3.6.0
Pillow>=9.0.0
```
- Albumentations for geometric/color augmentations (critical for small dataset)  
- Pandas for CSV label handling  
- Matplotlib in `show_data.py` for steering histogram analysis

**Mobile app** (`app/`)  
- Flutter ≥ 3.0 (single `lib/main.dart` — surprisingly minimal; uses `http` for MJPEG, `web_socket_channel` or raw TCP sockets, joystick widget)  
- Hardcoded Pi IP `192.168.4.1` (easy to change)

**Model format**  
- Trained → dynamic INT8 quantized TorchScript (`model.pt`)  
- Inference on Pi is `torch.jit.load()` + `torch.no_grad()`

### 4. Repository Structure (as of latest commit)
```
picar-autonomous/
├── pi/                          # Embedded runtime
│   ├── collect_data.py          # Keyboard → motor + frame capture + CSV
│   ├── drive_server.py          # Full inference + TCP + MJPEG + telemetry + avoidance
│   └── requirements.txt
├── workstation/                 # Training pipeline (run on laptop/desktop)
│   ├── model.py                 # DrivingCNN definition
│   ├── train.py                 # Data loader, augmentation, training loop, early stopping
│   ├── export.py                # Quantize + TorchScript export
│   ├── clean_dataset.py         # Filter bad frames, balance steering distribution
│   ├── show_data.py             # Steering histogram + sample viewer
│   └── requirements.txt
├── app/                         # Flutter client
│   └── lib/main.dart            # Single-file app (joystick + video + mode toggle)
├── dataset/                     # ~20 k frames + labels.csv (images/ gitignored)
│   ├── labels.csv
│   └── images/                  # frame_000000.jpg … (224×224 resized)
├── models/                      # gitignored — contains model.pt after export
├── docs/                        # Diagrams (Jupyter + PDF/PNG exports)
│   ├── diagrams.ipynb
│   ├── diagrams_ieee.ipynb
│   ├── behavioral_cloning_pipeline_highres.pdf/png
│   └── exports/png/
├── WORKFLOW.md                  # (if present) or see README for steps
├── .gitignore                   # ignores images/, *.pt, *.pth, __pycache__
├── README.md
```

### 5. How Each Component Works (Deep Dive)

#### 5.1 Data Collection (`pi/collect_data.py`)
- Runs on Pi with keyboard attached (or SSH + keyboard forwarding).  
- Camera: `picamera2` at 640×480 RGB → resized to **224×224** on-the-fly (exact model input size).  
- Control loop: 10 Hz (sleep 0.1 s). WASD incremental steering/throttle with exponential decay (auto-centering).  
- Only saves frames when `throttle >= 0.15` (filters out stationary data).  
- Saves resized JPG + `(filename, steering, throttle)` row to `dataset/labels.csv`.  
- Motor control identical to server (differential drive: `left = throttle * (1 - steering)`, `right = throttle * (1 + steering)` with slight 1.2× steering gain for snappier feel).

#### 5.2 Model Architecture (`workstation/model.py`)
`DrivingCNN` — very close to original PilotNet but with modern touches:
```python
Input: 3 × 224 × 224 RGB
Conv1: 5×5 stride 2 → 24 ch
Conv2: 5×5 stride 2 → 36 ch
Conv3: 5×5 stride 2 → 48 ch
Conv4: 3×3 stride 1 → 64 ch
Conv5: 3×3 stride 1 → 64 ch
Flatten → FC(100) + Dropout(0.5) → FC(50) + Dropout(0.5) → Linear(2)
Output: [steering, throttle]  (but throttle ignored in deployment)
```
Dynamic feature-size calculation in `__init__` (no hard-coded flatten size). ReLU everywhere. Dropout added to fight overfitting on small datasets.

#### 5.3 Training Pipeline (`workstation/train.py` + helpers)
- Loads CSV + images.  
- Albumentations pipeline (random crop, flips, brightness/contrast, Gaussian blur, etc.).  
- Train/validation split (sklearn).  
- MSE loss on both outputs (but throttle weight can be lowered).  
- Early stopping + learning-rate scheduling.  
- `show_data.py` helps you visualize steering distribution before/after cleaning.  
- `clean_dataset.py` removes near-zero throttle frames, outliers, or balances the dataset.

#### 5.4 Model Export (`workstation/export.py`)
- Loads trained state dict.  
- Dynamic INT8 quantization (`torch.quantization.quantize_dynamic`).  
- Traces + scripts to TorchScript (`model.pt`).  
- This single file is SCP’d to the Pi.

#### 5.5 Runtime / Inference Server (`pi/drive_server.py`)
This is the heart of the system. Architecture (from source comments):
- **Three background daemon threads**:
  1. TCP control server (port 5005) — receives newline-delimited JSON commands from app.
  2. TCP telemetry server (port 5006) — pushes state at ~10 Hz.
  3. HTTP MJPEG server (port 8080) — serves `/stream` (and a browser test page at `/`).
- **Main control loop** (~20 Hz):
  - Grabs latest frame from picamera2.
  - If mode == "autonomous":
    - Preprocess (resize + ToTensor + normalize).
    - `model( )` → steering.
    - Throttle fixed at 0.5.
    - Check all four ultrasonics.
    - If any obstacle too close → trigger `go_backward()` + `turn_in_place()`.
  - Else: use joystick values from last control packet.
  - `set_motors(steering, throttle)` (same differential logic).
  - Update shared state for telemetry.

**Network protocol** (JSON over raw TCP — very lightweight):
- **Control** (app → Pi): `{"steering": -0.3, "throttle": 0.6, "mode": "autonomous"}`
- **Telemetry** (Pi → app): FPS, current steering/throttle, mode, obstacle flag, all four sensor distances (999.0 = timeout/no obstacle).

**Obstacle avoidance** is purely reactive (reverse 0.5 s → 180° turn → resume). Not learned; hard-coded for safety.

### 6. Full End-to-End Workflow
1. Hardware + Pi Wi-Fi AP setup.  
2. `python pi/collect_data.py` → drive manually → transfer `dataset/` via SCP.  
3. On workstation: `clean_dataset.py` → `train.py` → `export.py` → `model.pt`.  
4. SCP `model.pt` back to Pi (must be in same dir as `drive_server.py`).  
5. `python pi/drive_server.py`.  
6. Open Flutter app → connect → drive or toggle autonomous.

### 7. Nuances, Edge Cases & Engineering Considerations
- **Real-time performance**: Control loop targets 20 Hz; camera + inference on Pi 4B hits ~15–20 FPS comfortably. Telemetry FPS reported back to app.  
- **Data volume**: ~20 k frames ≈ 1–2 GB (224×224 JPGs). Use external SSD or DVC for versioning.  
- **GPIO conflicts**: Both `collect_data.py` and `drive_server.py` set up the exact same pins — never run them at the same time.  
- **Safety**: No emergency stop button in autonomous mode beyond the app’s manual override. Ultrasonic timeout returns 999 cm — treated as “clear”.  
- **Generalization**: The model will only be as good as your driving data. Collect laps in both directions, different lighting, slight perturbations.  
- **Extensibility**: Easy to add IMU, wheel encoders, or even switch to a full RL loop later because the server architecture is clean.  
- **Deployment gotchas**: `drive_server.py` must be launched from the directory containing `model.pt`. Pi IP is hardcoded in Dart — change before building the APK/IPA.  
- **Legal / safety note**: This is a toy RC car project. Do **not** put it on public roads.

This repo is a beautifully clean, production-grade example of how to ship a full behavioral cloning stack from data collection to mobile-controlled inference on $50–100 hardware. Everything is modular, well-commented, and split exactly where it should be (embedded vs. workstation).  

If you want me to walk through any single file line-by-line, suggest improvements (e.g., add ONNX export, ROS2 bridge, better avoidance, etc.), or help you set it up on your own PiCar, just say the word.
