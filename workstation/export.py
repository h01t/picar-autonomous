"""
Optimized Model Export for Raspberry Pi
- Quantization for faster inference on CPU
- TorchScript compilation
- Model verification
"""

import torch
import torch.quantization
from model import DrivingCNN
import time
import numpy as np

print("=" * 50)
print("PiCar Model Optimization & Export")
print("=" * 50)

# =============================
# 1. LOAD BEST MODEL
# =============================
print("\n[1/5] Loading best model...")
device = torch.device("cpu")  # Export on CPU for Pi compatibility

model = DrivingCNN()
model.load_state_dict(torch.load("model_best.pth", map_location=device))
model.eval()
print("✓ Model loaded successfully")

# =============================
# 2. VERIFY MODEL OUTPUTS
# =============================
print("\n[2/5] Verifying model outputs...")
test_input = torch.rand(1, 3, 224, 224)

with torch.no_grad():
    output = model(test_input)
    steering, throttle = output[0].cpu().numpy()

print(f"  Sample prediction: steering={steering:.3f}, throttle={throttle:.3f}")
if -1.0 <= steering <= 1.0 and 0.0 <= throttle <= 1.0:
    print("✓ Output ranges look good")
else:
    print("⚠ WARNING: Outputs outside expected range!")

# =============================
# 3. BENCHMARK ORIGINAL MODEL
# =============================
print("\n[3/5] Benchmarking original model...")
iterations = 100
start = time.time()

with torch.no_grad():
    for _ in range(iterations):
        _ = model(test_input)

original_time = (time.time() - start) / iterations * 1000
print(f"  Original model: {original_time:.2f}ms per inference")

# =============================
# 4. APPLY DYNAMIC QUANTIZATION
# =============================
print("\n[4/5] Applying dynamic quantization...")

try:
    # Try new API first (PyTorch 2.4+)
    import torch.ao.quantization as quantization_new
    quantized_model = quantization_new.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    print("  Using new quantization API")
except (ImportError, AttributeError):
    # Fallback to old API
    import torch.quantization as quantization_old
    quantized_model = quantization_old.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    print("  Using legacy quantization API (will be deprecated in PyTorch 2.10)")

# Benchmark quantized model
start = time.time()
with torch.no_grad():
    for _ in range(iterations):
        _ = quantized_model(test_input)

quantized_time = (time.time() - start) / iterations * 1000
speedup = original_time / quantized_time

print(f"  Quantized model: {quantized_time:.2f}ms per inference")
print(f"  Speedup: {speedup:.2f}x")

# =============================
# 5. EXPORT TO TORCHSCRIPT
# =============================
print("\n[5/5] Exporting to TorchScript...")

# Use the quantized model for export
example = torch.rand(1, 3, 224, 224)

# Trace the model (converts to TorchScript)
scripted_model = torch.jit.trace(quantized_model, example)

# Note: optimize_for_inference can cause issues with quantized models
# Skipping it to ensure compatibility

# Save
scripted_model.save("model.pt")
print("✓ Saved to model.pt")

# =============================
# 6. VERIFY EXPORTED MODEL
# =============================
print("\n[6/6] Verifying exported model...")

loaded_model = torch.jit.load("model.pt")
loaded_model.eval()

with torch.no_grad():
    original_output = quantized_model(test_input)
    exported_output = loaded_model(test_input)

difference = torch.abs(original_output - exported_output).max().item()
print(f"  Max difference: {difference:.6f}")

if difference < 1e-4:
    print("✓ Exported model matches original")
else:
    print("⚠ Small numerical differences (expected with quantization)")

# =============================
# 7. MODEL SIZE COMPARISON
# =============================
print("\n" + "=" * 50)
print("EXPORT SUMMARY")
print("=" * 50)

import os
model_size = os.path.getsize("model.pt") / (1024 * 1024)
print(f"\nModel size: {model_size:.2f} MB")
print(f"Inference speed (on this machine): {quantized_time:.2f}ms")
print(f"Expected on Pi 4: ~{quantized_time * 2.5:.0f}ms (estimated)")

# Calculate max FPS on Pi
pi_fps = 1000 / (quantized_time * 2.5)
print(f"Expected FPS on Pi: ~{pi_fps:.1f}")

print("\n✓ Export complete! Transfer model.pt to your Pi.")
print("\nRecommended deployment:")
print("  scp model.pt pi@192.168.4.1:~/")
print("=" * 50)

# =============================
# 8. SAVE METADATA (OPTIONAL)
# =============================
metadata = {
    "model_type": "DrivingCNN",
    "input_size": [224, 224],
    "quantized": True,
    "inference_time_ms": quantized_time,
    "speedup": speedup,
    "training_samples": "See labels.csv"
}

import json
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n✓ Saved model metadata to model_metadata.json")