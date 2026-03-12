import os
import csv
import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
import keyboard
import re

# =============================
# PATHS
# =============================
IMAGE_DIR = "dataset/images"
CSV_PATH = "dataset/labels.csv"
os.makedirs(IMAGE_DIR, exist_ok=True)

# =============================
# MOTOR GPIO (same as before)
# =============================
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

IN1, IN2, IN3, IN4 = 17, 24, 22, 27
ENA, ENB = 18, 23

GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)

pwmA = GPIO.PWM(ENA, 1000)
pwmB = GPIO.PWM(ENB, 1000)
pwmA.start(0)
pwmB.start(0)

def set_motors(steering, throttle):
    steering *= 1.2
    steering = max(-1.0, min(1.0, steering))
    throttle = max(0.0, min(1.0, throttle))

    left_speed = throttle * (1 - steering)
    right_speed = throttle * (1 + steering)

    left_speed = max(0, min(1, left_speed))
    right_speed = max(0, min(1, right_speed))

    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

    pwmA.ChangeDutyCycle(left_speed * 100)
    pwmB.ChangeDutyCycle(right_speed * 100)

def stop():
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)

# =============================
# FRAME ID
# =============================
def get_next_frame_id():
    ids = []
    for f in os.listdir(IMAGE_DIR):
        m = re.match(r"frame_(\d+)\.jpg", f)
        if m:
            ids.append(int(m.group(1)))
    return max(ids) + 1 if ids else 0

frame_id = get_next_frame_id()
print(f"[INFO] Starting from frame_id {frame_id}")

# =============================
# CAMERA
# =============================
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(1)

# =============================
# CSV
# =============================
csv_exists = os.path.exists(CSV_PATH)
csv_file = open(CSV_PATH, "a", newline="")
writer = csv.writer(csv_file)

if not csv_exists:
    writer.writerow(["image", "steering", "throttle"])

# =============================
# CONTROL STATE
# =============================
steering = 0.0
throttle = 0.0
STEER_STEP = 0.08
THROTTLE_STEP = 0.05
MIN_THROTTLE_TO_SAVE = 0.15  # NEW: Skip nearly-stopped frames

saved_count = 0
skipped_count = 0

# =============================
# MAIN LOOP
# =============================
print("\n=== CONTROLS ===")
print("W/S: Throttle up/down")
print("A/D: Steer left/right")
print("SPACE: Emergency stop")
print("Q: Quit")
print("================\n")

try:
    while True:
        if keyboard.is_pressed("q"):
            print("\n[INFO] Quit requested")
            break

        frame = picam2.capture_array()
        frame_resized = cv2.resize(frame, (224, 224))

        # -------- KEYBOARD CONTROL --------
        if keyboard.is_pressed("space"):
            throttle = 0
            steering = 0
        
        if keyboard.is_pressed("a"):
            steering -= STEER_STEP
        elif keyboard.is_pressed("d"):
            steering += STEER_STEP
        else:
            steering *= 0.85  # auto-centering

        if keyboard.is_pressed("w"):
            throttle += THROTTLE_STEP
        elif keyboard.is_pressed("s"):
            throttle -= THROTTLE_STEP
        else:
            throttle *= 0.9  # gradual slowdown

        steering = max(-1, min(1, steering))
        throttle = max(0, min(1, throttle))

        # -------- SAVE ONLY IF MOVING --------
        if throttle >= MIN_THROTTLE_TO_SAVE:
            filename = f"frame_{frame_id:06d}.jpg"
            path = os.path.join(IMAGE_DIR, filename)

            if cv2.imwrite(path, cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)):
                writer.writerow([filename, steering, throttle])
                csv_file.flush()
                saved_count += 1
                frame_id += 1
        else:
            skipped_count += 1

        set_motors(steering, throttle)

        # Show stats every 50 frames
        if (saved_count + skipped_count) % 50 == 0:
            print(f"[STATS] Saved: {saved_count} | Skipped: {skipped_count} | "
                  f"Throttle: {throttle:.2f} | Steering: {steering:.2f}")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n[INFO] Stopped by user")

finally:
    stop()
    pwmA.stop()
    pwmB.stop()
    GPIO.cleanup()
    csv_file.close()
    picam2.stop()
    print(f"\n[INFO] Final stats:")
    print(f"  Saved frames: {saved_count}")
    print(f"  Skipped frames: {skipped_count}")
    print(f"  Final frame_id: {frame_id}")