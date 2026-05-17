"""Test all models individually"""
import subprocess
import sys

MODELS = [
    ("LSTM", "test_lstm.py"),
    ("VAE", "test_vae.py"),
    ("CGAN", "test_cgan.py"),
    ("Diffusion", "test_diffusion.py"),
]

print("=" * 60)
print("  Testing Individual Models")
print("=" * 60)

for model_name, script in MODELS:
    print(f"\n[Testing {model_name}]")
    print("-" * 60)
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"✗ {model_name} test failed")
    else:
        print(f"✓ {model_name} test passed")

print("\n" + "=" * 60)
print("  All tests completed")
print("=" * 60)
