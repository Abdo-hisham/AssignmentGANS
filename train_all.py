"""
train_all.py – Train all four models sequentially.
Run: python train_all.py
"""

import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODELS = ["lstm", "vae", "cgan", "diffusion"]

for model in MODELS:
    script = ROOT / "model" / model / "train.py"
    print(f"\n{'='*60}")
    print(f"  Training {model.upper()}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"[ERROR] {model} training failed with code {result.returncode}")
        sys.exit(result.returncode)

print("\n✓ All models trained successfully!")
