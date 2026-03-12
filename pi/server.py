"""
pi/server.py — Inference & telemetry server for the Raspberry Pi.

Responsibilities:
  - Load the TorchScript model (model.pt) and run real-time inference
  - Serve a live MJPEG camera stream over HTTP on port 8080
  - Accept control commands (manual steering/throttle) via TCP on port 5005
  - Broadcast telemetry (sensor distances, speed, mode) via TCP on port 5006
  - Read 4× HC-SR04 ultrasonic sensors (front / left / right / rear)
  - Drive motors via GPIO based on either manual commands or model output

TODO: Implement this module.
      See collect_data.py for motor control and camera capture patterns.
      See workstation/export.py for how model.pt is loaded (torch.jit.load).
"""

raise NotImplementedError("pi/server.py is not yet implemented.")
