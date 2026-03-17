"""
pi/drive_server.py — Main inference and control server for the Raspberry Pi.

Run this on the Pi to enable both manual and autonomous driving via the Flutter app.

Architecture — three daemon threads + one main loop:
  - control_socket      (TCP :5005) — receives steering/throttle/mode commands from the app
  - telemetry_socket    (TCP :5006) — pushes sensor + state data back to the app at 10 Hz
  - camera_stream_server (HTTP :8080) — serves a live MJPEG stream at ~30 FPS
  - control_loop  (main thread) — drives motors at ~20 Hz; runs model inference in autonomous mode

Usage:
  # Copy model.pt to the same directory as this file, then:
  python3 drive_server.py
"""

import socket
import json
import time
import threading
import RPi.GPIO as GPIO
import torch
import torchvision.transforms as transforms
from PIL import Image
from picamera2 import Picamera2
import numpy as np
import io
from http.server import BaseHTTPRequestHandler, HTTPServer

# =============================
# GPIO SETUP
# =============================
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# L298N motor driver pins (BCM numbering).
# IN1/IN2 control left motor direction; IN3/IN4 control right motor direction.
# ENA/ENB are the PWM enable pins for speed control.
IN1, IN2, IN3, IN4 = 17, 24, 22, 27
ENA, ENB = 18, 23

# HC-SR04 ultrasonic sensor pins — one TRIG + one ECHO per sensor.
TRIG_FRONT, TRIG_LEFT, TRIG_RIGHT, TRIG_REAR = 12, 20, 5, 16
ECHO_FRONT, ECHO_LEFT, ECHO_RIGHT, ECHO_REAR = 13, 21, 6, 26

# Setup motor pins as outputs
GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)

# TRIG pins: output (we pulse them to fire the sensor)
# ECHO pins: input (we listen for the reflected pulse)
for trig_pin in [TRIG_FRONT, TRIG_LEFT, TRIG_RIGHT, TRIG_REAR]:
    GPIO.setup(trig_pin, GPIO.OUT)
    GPIO.output(trig_pin, GPIO.LOW)  # Start low — avoids spurious triggers on boot

for echo_pin in [ECHO_FRONT, ECHO_LEFT, ECHO_RIGHT, ECHO_REAR]:
    GPIO.setup(echo_pin, GPIO.IN)

# PWM at 1 kHz for smooth motor speed control; start at 0% duty cycle (stopped)
pwmA = GPIO.PWM(ENA, 1000)
pwmB = GPIO.PWM(ENB, 1000)
pwmA.start(0)
pwmB.start(0)

print("[GPIO] Motor and ultrasonic sensors initialized")


def stop():
    """Immediately cut power to both motors."""
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)


def set_motors(steering, throttle):
    """
    Drive both motors based on a differential steering model.

    Args:
        steering: float in [-1.0, 1.0] — negative = left, positive = right
        throttle: float in [0.0, 1.0] — overall forward speed

    The differential mixing formula:
        left_speed  = throttle * (1 - steering)
        right_speed = throttle * (1 + steering)

    When steering = 0:  both sides run at `throttle` speed (straight ahead).
    When steering = +1: left runs at 2×throttle, right at 0 (hard right).
    When steering = -1: right runs at 2×throttle, left at 0 (hard left).
    Values are clamped to [0, 100]% duty cycle before writing to PWM.
    """
    steering = max(-1.0, min(1.0, steering))
    throttle = max(0.0, min(1.0, throttle))

    left  = throttle * (1 - steering)
    right = throttle * (1 + steering)

    # Both motors forward direction (IN1/IN3 HIGH, IN2/IN4 LOW)
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

    pwmA.ChangeDutyCycle(max(0, min(100, left  * 100)))
    pwmB.ChangeDutyCycle(max(0, min(100, right * 100)))


def go_backward(speed=0.6, duration=0.5):
    """
    Drive both motors in reverse for a fixed duration.
    Used exclusively by the obstacle avoidance routine.

    Args:
        speed:    float in [0.0, 1.0] — reverse speed (duty cycle fraction)
        duration: float — seconds to reverse before stopping
    """
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)

    pwmA.ChangeDutyCycle(speed * 100)
    pwmB.ChangeDutyCycle(speed * 100)
    time.sleep(duration)
    stop()


def turn_in_place(direction, duration=0.8):
    """
    Spin the car in place by driving left and right motors in opposite directions.
    Used by the obstacle avoidance routine to reorient after reversing.

    Args:
        direction: "left" or "right"
        duration:  float — seconds to spin before stopping
    """
    if direction == "left":
        # Left motor backward, right motor forward → counter-clockwise spin
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)
    else:  # right
        # Left motor forward, right motor backward → clockwise spin
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)

    pwmA.ChangeDutyCycle(70)
    pwmB.ChangeDutyCycle(70)
    time.sleep(duration)
    stop()


# =============================
# ULTRASONIC SENSOR FUNCTIONS
# =============================
def get_distance(trig_pin, echo_pin):
    """
    Measure distance using a single HC-SR04 ultrasonic sensor.

    Protocol:
      1. Send a 10 µs HIGH pulse on TRIG to fire the ultrasonic burst.
      2. Measure how long ECHO stays HIGH — that duration is the round-trip
         travel time of the sound wave.
      3. Distance (cm) = pulse_duration × (speed_of_sound / 2)
                       = pulse_duration × 17150  (34300 cm/s ÷ 2)

    A 100 ms timeout guards against a permanently-stuck ECHO pin.
    Returns 999.9 cm on timeout (treated as "no obstacle in range").

    Args:
        trig_pin: BCM pin number for the TRIG output
        echo_pin: BCM pin number for the ECHO input

    Returns:
        Distance in cm, rounded to 1 decimal place.
    """
    GPIO.output(trig_pin, GPIO.HIGH)
    time.sleep(0.00001)  # 10 µs pulse
    GPIO.output(trig_pin, GPIO.LOW)

    timeout = time.time() + 0.1  # 100 ms timeout

    # Wait for ECHO to go HIGH (start of reflected pulse)
    while GPIO.input(echo_pin) == GPIO.LOW:
        pulse_start = time.time()
        if pulse_start > timeout:
            return 999.9

    # Wait for ECHO to go LOW (end of reflected pulse)
    while GPIO.input(echo_pin) == GPIO.HIGH:
        pulse_end = time.time()
        if pulse_end > timeout:
            return 999.9

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150  # Travel time → distance in cm

    return round(distance, 1)


def get_all_distances():
    """
    Read all four HC-SR04 sensors sequentially.

    Returns:
        dict with keys 'front', 'left', 'right', 'rear' — values in cm.
    """
    return {
        'front': get_distance(TRIG_FRONT, ECHO_FRONT),
        'left':  get_distance(TRIG_LEFT,  ECHO_LEFT),
        'right': get_distance(TRIG_RIGHT, ECHO_RIGHT),
        'rear':  get_distance(TRIG_REAR,  ECHO_REAR),
    }


def handle_obstacle_avoidance():
    """
    Emergency obstacle avoidance procedure — called when the control loop
    detects an imminent collision in autonomous mode.

    Strategy:
      1. Stop immediately.
      2. If rear is clear (>30 cm): reverse 0.8 s, then turn toward the
         clearer side (left vs right, with a 15 cm hysteresis threshold).
      3. If rear is also blocked: turn in place toward the clearer side
         for a longer duration (1.2 s).

    This function is blocking — the control loop waits while it executes.
    On return, the control loop resumes normal model-based steering.
    """
    print("\n[OBSTACLE] Emergency avoidance initiated!")

    stop()
    time.sleep(0.3)

    distances = get_all_distances()
    print(f"[OBSTACLE] Distances — "
          f"Front:{distances['front']:.1f} "
          f"Left:{distances['left']:.1f} "
          f"Right:{distances['right']:.1f} "
          f"Rear:{distances['rear']:.1f}")

    if distances['rear'] > 30:
        print("[OBSTACLE] Reversing...")
        go_backward(speed=0.6, duration=0.8)
        time.sleep(0.2)

        # Re-read after reversing — the environment has changed
        distances = get_all_distances()

        # 15 cm hysteresis: only prefer a side if it's clearly more open
        if distances['left'] > distances['right'] + 15:
            print("[OBSTACLE] Turning LEFT (clearer)")
            turn_in_place("left", duration=0.9)
        elif distances['right'] > distances['left'] + 15:
            print("[OBSTACLE] Turning RIGHT (clearer)")
            turn_in_place("right", duration=0.9)
        else:
            print("[OBSTACLE] Turning RIGHT (default — sides similar)")
            turn_in_place("right", duration=0.9)
    else:
        print("[OBSTACLE] Rear blocked — turning in place...")
        if distances['left'] > distances['right']:
            turn_in_place("left", duration=1.2)
        else:
            turn_in_place("right", duration=1.2)

    time.sleep(0.3)
    print("[OBSTACLE] Avoidance complete, resuming\n")


# =============================
# CAMERA SETUP
# =============================
print("[CAMERA] Initializing...")
picam2 = Picamera2()

# Video configuration at 640×480 RGB for the MJPEG stream.
# Frames are also fed to the model, resized to 224×224 inside
# get_model_prediction() via the torchvision transform pipeline.
config = picam2.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(2)  # Allow auto-exposure and auto-white-balance to stabilise
print("[CAMERA] Ready")


# =============================
# MODEL LOADING
# =============================
device = torch.device("cpu")  # Pi 4 has no GPU; all inference runs on CPU

model = None

# latest_frame is written by control_loop and read by get_model_prediction()
# and StreamingHandler. Always access it under frame_lock.
latest_frame = None
frame_lock = threading.Lock()

try:
    # model.pt must be in the same directory this script is run from.
    # Deploy it with: scp model.pt pi@192.168.4.1:~/
    model = torch.jit.load("model.pt", map_location=device)
    model.eval()
    print("[MODEL] Loaded successfully (TorchScript)")
except Exception as e:
    print(f"[MODEL] Failed to load: {e}")
    print("[MODEL] Autonomous mode will be disabled until model.pt is present")

# Preprocessing pipeline applied to every frame before inference.
# NOTE: mean/std here are the standard ImageNet normalisation values.
# The model was NOT pretrained on ImageNet — these values were chosen to
# keep pixel values in a similar numeric range to common PyTorch conventions.
# If you retrain from scratch you may want to compute the actual dataset
# mean/std and update these values accordingly.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def get_model_prediction():
    """
    Run one forward pass through the CNN on the latest camera frame.

    Grabs a thread-safe copy of latest_frame, preprocesses it, and returns
    the predicted (steering, throttle) pair clamped to their valid ranges.

    Returns:
        (steering, throttle) — both floats; (0.0, 0.0) on any error.
    """
    global latest_frame

    if model is None or latest_frame is None:
        return 0.0, 0.0

    try:
        with frame_lock:
            frame_copy = latest_frame.copy()

        # Strip alpha channel if the camera ever returns RGBA
        if frame_copy.shape[2] == 4:
            frame_copy = frame_copy[:, :, :3]

        img = Image.fromarray(frame_copy)
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            steering = float(output[0][0].item())
            throttle = float(output[0][1].item())

        # Hard clamp — don't let the model command unsafe values
        steering = max(-1.0, min(1.0, steering))
        throttle = max(0.0,  min(1.0, throttle))

        return steering, throttle

    except Exception as e:
        print(f"[MODEL] Prediction error: {e}")
        return 0.0, 0.0


# =============================
# SHARED STATE
# =============================
# All threads read/write this dict. Always acquire `lock` before accessing.
state = {
    "steering": 0.0,
    "throttle": 0.0,
    "mode": "manual",          # "manual" | "collect" | "autonomous"
    "last_cmd": time.time(),   # Timestamp of the most recent command from the app
    "obstacle_detected": False
}
lock = threading.Lock()


# =============================
# CONTROL SOCKET (5005)
# =============================
def control_socket():
    """
    TCP server that accepts one client at a time on port 5005.

    Expected message format — newline-delimited JSON:
        {"steering": 0.3, "throttle": 0.5, "mode": "manual"}

    On disconnect the server immediately starts listening for a new client.
    state["last_cmd"] is updated on every valid message — the watchdog in
    control_loop() stops the motors if no command arrives within 0.5 s.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", 5005))
        s.listen(5)
        print("[CONTROL] Listening on 5005")

        while True:
            try:
                conn, addr = s.accept()
                print(f"[CONTROL] Connected by {addr}")

                with conn, conn.makefile('r', encoding='utf-8') as f:
                    while True:
                        try:
                            line = f.readline()
                            if not line:
                                break

                            msg = json.loads(line.strip())

                            with lock:
                                state["steering"] = float(msg.get("steering", 0.0))
                                state["throttle"] = float(msg.get("throttle", 0.0))
                                state["mode"]     = msg.get("mode", state["mode"])
                                state["last_cmd"] = time.time()

                        except ValueError:
                            print("[CONTROL] JSON decode error — skipping line")
                        except Exception as e:
                            print(f"[CONTROL] Error: {e}")
                            break

                print(f"[CONTROL] {addr} disconnected")

            except Exception as e:
                print(f"[CONTROL] Accept error: {e}")
                time.sleep(1)


# =============================
# TELEMETRY SOCKET (5006)
# =============================
def telemetry_socket():
    """
    TCP server that pushes telemetry to one connected client at ~10 Hz on port 5006.

    Payload format — newline-delimited JSON:
        {
            "fps": 19.8,
            "steering": 0.12,
            "throttle": 0.50,
            "mode": "autonomous",
            "obstacle": false,
            "sensors": {"front": 82.3, "left": 120.0, "right": 95.5, "rear": 999.0}
        }

    Sensor distances are only read from hardware when in autonomous mode;
    in manual mode they are reported as 999 (no-obstacle sentinel) to avoid
    firing all four ultrasonic sensors on every telemetry cycle.

    NOTE: loop_fps is defined at module level (below) and updated by
    control_loop(). It is safe to read here without a lock because Python
    float assignment is atomic on CPython.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", 5006))
        s.listen(5)
        print("[TELEMETRY] Listening on 5006")

        while True:
            try:
                conn, addr = s.accept()
                print(f"[TELEMETRY] Connected by {addr}")

                with conn:
                    while True:
                        time.sleep(0.1)  # 10 Hz telemetry rate

                        # Only poll ultrasonic hardware in autonomous mode
                        sensor_data = {"front": 999, "left": 999, "right": 999, "rear": 999}
                        with lock:
                            if state["mode"] == "autonomous":
                                sensor_data = get_all_distances()

                        with lock:
                            payload = json.dumps({
                                "fps":      loop_fps,
                                "steering": round(state["steering"], 3),
                                "throttle": round(state["throttle"], 3),
                                "mode":     state["mode"],
                                "obstacle": state["obstacle_detected"],
                                "sensors": {
                                    "front": round(sensor_data['front'], 1),
                                    "left":  round(sensor_data['left'],  1),
                                    "right": round(sensor_data['right'], 1),
                                    "rear":  round(sensor_data['rear'],  1),
                                }
                            })
                        try:
                            conn.sendall((payload + "\n").encode())
                        except Exception:
                            break

                print(f"[TELEMETRY] {addr} disconnected")

            except Exception as e:
                print(f"[TELEMETRY] Accept error: {e}")
                time.sleep(1)


# =============================
# CAMERA STREAMING SERVER (8080)
# =============================
class StreamingHandler(BaseHTTPRequestHandler):
    """
    Minimal HTTP handler that serves a live MJPEG stream at /stream.

    MJPEG (Motion JPEG) is a simple streaming format: each frame is sent as
    a JPEG payload inside a multipart HTTP response. The Flutter app reads
    the stream chunk-by-chunk and renders each JPEG as it arrives.

    Routes:
        GET /stream — MJPEG stream (consumed by the Flutter app)
        GET /       — Static HTML test page with an <img> pointing at /stream.
                      Open http://192.168.4.1:8080 in a browser to verify the stream.
    """

    def do_GET(self):
        print(f"[STREAM] Request: {self.path} from {self.client_address}")

        if self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            print("[STREAM] Headers sent, starting frame loop")

            try:
                frame_count = 0
                while True:
                    with frame_lock:
                        if latest_frame is not None:
                            frame = latest_frame.copy()
                        else:
                            if frame_count == 0:
                                print("[STREAM] Waiting for first frame...")
                            time.sleep(0.1)
                            continue

                    # Strip alpha channel if present (defensive)
                    if len(frame.shape) == 3 and frame.shape[2] == 4:
                        frame = frame[:, :, :3]

                    # JPEG encode at quality=80 — good balance of visual
                    # quality vs. bandwidth over the local WiFi AP link
                    pil_img = Image.fromarray(frame)
                    buf = io.BytesIO()
                    pil_img.save(buf, format='JPEG', quality=80)
                    jpeg_data = buf.getvalue()

                    # MJPEG frame: boundary + per-frame headers + JPEG payload
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(f'Content-Type: image/jpeg\r\n'.encode())
                    self.wfile.write(f'Content-Length: {len(jpeg_data)}\r\n\r\n'.encode())
                    self.wfile.write(jpeg_data)
                    self.wfile.write(b'\r\n')

                    frame_count += 1
                    if frame_count % 30 == 0:
                        print(f"[STREAM] {frame_count} frames sent to {self.client_address}")

                    time.sleep(0.033)  # ~30 FPS cap

            except Exception as e:
                print(f"[STREAM] Client {self.client_address} disconnected: {e}")

        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"""
            <html><body>
            <h1>PiCar Camera Stream</h1>
            <img src="/stream" width="640" height="480">
            </body></html>
            """)
        else:
            self.send_error(404)
            self.end_headers()

    def log_message(self, format, *args):
        print(f"[STREAM] {format % args}")


def camera_stream_server():
    """Start the blocking MJPEG HTTP server — runs in its own daemon thread."""
    server = HTTPServer(('0.0.0.0', 8080), StreamingHandler)
    print("[STREAM] Camera stream on port 8080")
    server.serve_forever()


# =============================
# MAIN CONTROL LOOP + WATCHDOG
# =============================

# loop_fps is written by control_loop() and read by telemetry_socket().
# Defined at module level so telemetry can reference it before the first
# FPS measurement completes (avoids a NameError on early connections).
loop_fps = 0.0


def control_loop():
    """
    Main control loop — runs on the main thread at ~20 Hz.

    Each iteration:
      1. Capture a frame from the camera and update latest_frame.
      2. Safety watchdog — if no command arrived in the last 0.5 s
         (e.g. app disconnected), stop the motors and skip this cycle.
      3. Mode dispatch:
           autonomous    — read ultrasonic sensors, check obstacle thresholds,
                           run model inference, drive motors.
           manual/collect — pass app steering/throttle directly to motors,
                            no obstacle detection.
      4. Update the rolling FPS counter (averaged over 10 frames).

    Autonomous mode design notes:
      - FRONT_DANGER (25 cm): triggers full avoidance (stop → reverse → turn).
      - SIDE_DANGER  (15 cm): triggers avoidance only when the model is actively
        steering INTO that side (avoids false positives on wide open turns).
      - Throttle is fixed at 0.5 in autonomous mode — only steering comes from
        the model. The training data has high throttle variance (humans
        accelerate/brake inconsistently), so model throttle predictions are
        unreliable. A fixed moderate speed is safer for autonomous operation.
    """
    global loop_fps, latest_frame
    frames = 0
    t0 = time.time()

    while True:
        start = time.time()

        # --- 1. Capture frame ---
        try:
            frame = picam2.capture_array()
            with frame_lock:
                latest_frame = frame

            # Occasional shape sanity check (useful when debugging camera config)
            if frames % 100 == 0:
                print(f"[DEBUG] Frame shape: {frame.shape}, dtype: {frame.dtype}")

        except Exception as e:
            print(f"[CAMERA] Capture error: {e}")

        with lock:
            s    = state["steering"]
            t    = state["throttle"]
            mode = state["mode"]
            last = state["last_cmd"]

        # --- 2. Safety watchdog ---
        # Stop motors immediately if the app hasn't sent a command in 0.5 s
        if time.time() - last > 0.5:
            stop()
            frames += 1
            if frames >= 10:
                loop_fps = round(frames / (time.time() - t0), 2)
                frames = 0
                t0 = time.time()
            time.sleep(max(0, 0.05 - (time.time() - start)))
            continue

        # --- 3. Mode dispatch ---
        if mode == "autonomous":
            distances = get_all_distances()

            FRONT_DANGER = 25  # cm — full avoidance if front sensor reads below this
            SIDE_DANGER  = 15  # cm — avoidance only when steering into this side

            obstacle_detected = False

            if distances['front'] < FRONT_DANGER:
                obstacle_detected = True
                print(f"[SAFETY] Front obstacle at {distances['front']:.1f} cm!")

            elif distances['left'] < SIDE_DANGER and s < -0.3:
                # Steering left AND left sensor very close — about to clip the wall
                obstacle_detected = True
                print(f"[SAFETY] Left obstacle at {distances['left']:.1f} cm while turning left!")

            elif distances['right'] < SIDE_DANGER and s > 0.3:
                # Steering right AND right sensor very close
                obstacle_detected = True
                print(f"[SAFETY] Right obstacle at {distances['right']:.1f} cm while turning right!")

            with lock:
                state["obstacle_detected"] = obstacle_detected

            if obstacle_detected:
                handle_obstacle_avoidance()
                # Blocking call — resumes here after avoidance completes

            if model is not None:
                model_s, model_t = get_model_prediction()

                # Model controls steering; throttle is fixed (see docstring for rationale)
                s = model_s
                t = 0.5

                if frames % 20 == 0:
                    print(f"[AUTO] Steer={model_s:.3f} | "
                          f"F:{distances['front']:.0f} "
                          f"L:{distances['left']:.0f} "
                          f"R:{distances['right']:.0f} cm")

                set_motors(s, t)

                # Reflect actual commanded values in shared state for telemetry
                with lock:
                    state["steering"] = s
                    state["throttle"] = t
            else:
                print("[WARN] Autonomous mode requested but model not loaded — stopping")
                stop()

        else:
            # Manual or collect: pass app commands directly to motors
            set_motors(s, t)
            with lock:
                state["obstacle_detected"] = False

        # --- 4. FPS counter (rolling average over 10 frames) ---
        frames += 1
        if frames >= 10:
            loop_fps = round(frames / (time.time() - t0), 2)
            frames = 0
            t0 = time.time()

        # Sleep for the remainder of the 50 ms budget (~20 Hz target)
        time.sleep(max(0, 0.05 - (time.time() - start)))


# =============================
# ENTRY POINT
# =============================
print("[SYSTEM] Starting all services...")
try:
    threading.Thread(target=control_socket,       daemon=True).start()
    threading.Thread(target=telemetry_socket,     daemon=True).start()
    threading.Thread(target=camera_stream_server, daemon=True).start()
    print("[SYSTEM] All services running. Starting control loop...")
    control_loop()  # Blocks forever — runs on the main thread
except KeyboardInterrupt:
    print("\n[SYSTEM] Shutting down...")
finally:
    stop()
    pwmA.stop()
    pwmB.stop()
    picam2.stop()
    GPIO.cleanup()
    print("[SYSTEM] Cleanup complete")
