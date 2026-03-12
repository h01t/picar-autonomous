# Workflow — picar-autonomous

End-to-end pipeline from data collection to autonomous driving.

---

## Hardware checklist (before starting)

- [ ] Raspberry Pi 4B powered and booted
- [ ] Pi Camera Module 3 connected and enabled (`raspi-config` → Interface Options → Camera)
- [ ] 4× HC-SR04 ultrasonic sensors wired (front / left / right / rear)
- [ ] L298N motor driver connected to GPIO pins (see `pi/drive_server.py` header for pin map)
- [ ] Pi configured as WiFi access point at `192.168.4.1`
- [ ] Pi dependencies installed: `pip install -r pi/requirements.txt`

---

## Step 1 — Collect training data (on Pi)

```bash
# On the Pi:
cd ~
python3 drive_server.py   # Not needed yet — just for reference
python3 collect_data.py
```

Controls:
| Key   | Action              |
|-------|---------------------|
| W / S | Throttle up / down  |
| A / D | Steer left / right  |
| Space | Emergency stop      |
| Q     | Quit                |

Frames are saved to `dataset/images/` and labels to `dataset/labels.csv`.
Run multiple sessions — aim for 20–30 minutes of varied driving total:
- Session 1: Corridor loops and figure-8 patterns at 0.4–0.7 throttle
- Session 2: Obstacle navigation (boxes/cones) with tight turns
- Session 3: Edge cases — close wall driving, recovery from near-collisions

---

## Step 2 — Transfer dataset to workstation

```bash
# On the workstation:
scp -r pi@192.168.4.1:~/dataset ./
```

This copies the `dataset/` folder (images + labels.csv) into the project root on your workstation.

---

## Step 3 — Clean dataset (on workstation)

```bash
cd workstation
python3 clean_dataset.py
```

Removes low-throttle frames (throttle < 0.2) and extreme-steering low-speed samples.
A backup of the original labels is saved as `dataset/labels_original.csv`.

Optionally inspect the steering distribution before/after:
```bash
python3 show_data.py
```

---

## Step 4 — Train model (on workstation)

```bash
cd workstation
python3 train.py
```

- Trains up to 100 epochs with early stopping (patience = 7)
- Saves `model_best.pth` whenever validation loss improves
- Saves `model_final.pth` at the end of training
- Run time: ~1–2 hours on a mid-range GPU; several hours on CPU

Output files are saved to the `workstation/` directory. Move them to `models/` when done:
```bash
mv model_best.pth model_final.pth ../models/
```

---

## Step 5 — Export model for Pi (on workstation)

```bash
cd workstation
python3 export.py
```

- Loads `model_best.pth`
- Applies dynamic INT8 quantization (Linear layers only)
- Exports to TorchScript → `model.pt`
- Prints inference speed benchmark and expected Pi FPS

Move the exported model to `models/`:
```bash
mv model.pt ../models/
```

---

## Step 6 — Deploy to Pi

```bash
# From the project root on the workstation:
scp models/model.pt pi@192.168.4.1:~/
```

The server loads `model.pt` from the directory it is run from (i.e. `~/` on the Pi).

---

## Step 7 — Start the drive server (on Pi)

```bash
# On the Pi:
cd ~
python3 drive_server.py
```

Expected startup output:
```
[GPIO] Motor and ultrasonic sensors initialized
[CAMERA] Initializing...
[CAMERA] Ready
[MODEL] Loaded successfully (TorchScript)
[SYSTEM] Starting all services...
[CONTROL] Listening on 5005
[TELEMETRY] Listening on 5006
[STREAM] Camera stream on port 8080
[SYSTEM] All services running. Starting control loop...
```

If `model.pt` is missing, autonomous mode is disabled but manual driving still works.

To verify the camera stream without the app, open a browser on any device connected to the Pi's WiFi:
```
http://192.168.4.1:8080
```

---

## Step 8 — Connect the mobile app and drive

1. Connect your phone to the Pi's WiFi network (`192.168.4.1`)
2. Open the app
3. **Manual mode** — use the joysticks to drive; the Pi relays commands directly to the motors
4. **Autonomous mode** — tap the toggle; the Pi runs model inference at ~20 Hz

---

## Re-training tip

If the car understeers or oversteers in autonomous mode, collect more data that emphasises the problem scenarios (tight turns, edge cases) and retrain from scratch — don't fine-tune the existing model on new data without revisiting the full dataset balance first.
