"""Test VAE model individually"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model.vae.model import VAEDateGenerator
from utils.tokenizer import PAD_ID

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 64
EMBED_DIM = 128
HIDDEN_DIM = 256
WEIGHT_PATH = Path("model/vae/weights/best.pt")

print(f"Device: {DEVICE}")
print(f"Testing VAE Model...")

# Create model
model = VAEDateGenerator(
    latent_dim=LATENT_DIM,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
).to(DEVICE)

# Load weights if exists
if WEIGHT_PATH.exists():
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
    print(f"✓ Loaded weights from {WEIGHT_PATH}")
else:
    print(f"⚠ No weights found at {WEIGHT_PATH}")

model.eval()

# Test inference
print("\n--- Testing Inference ---")
with torch.no_grad():
    # Random latent vector
    z = torch.randn(1, LATENT_DIM).to(DEVICE)
    output = model.decode(z)
    print(f"Generated shape: {output.shape}")
    print(f"✓ VAE generated output successfully")

print("\n✓ VAE test completed")
