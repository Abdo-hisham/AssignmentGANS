"""Test Diffusion model individually"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model.diffusion.model import DiffusionDateGenerator
from utils.tokenizer import PAD_ID

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 20
VOCAB_SIZE = 128
EMBED_DIM = 128
HIDDEN_DIM = 256
TIMESTEPS = 100
WEIGHT_PATH = Path("model/diffusion/weights/best.pt")

print(f"Device: {DEVICE}")
print(f"Testing Diffusion Model...")

# Create model
model = DiffusionDateGenerator(
    seq_len=SEQ_LEN,
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    timesteps=TIMESTEPS,
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
    # Random condition (4 condition tokens)
    cond = torch.randint(0, VOCAB_SIZE, (1, 4)).to(DEVICE)
    
    # Generate sample (reverse diffusion)
    x_t = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN)).to(DEVICE)
    for t in range(TIMESTEPS-1, -1, -1):
        t_tensor = torch.tensor([t], dtype=torch.long).to(DEVICE)
        output = model(x_t, t_tensor, cond)
    
    print(f"Generated shape: {output.shape}")
    print(f"✓ Diffusion inference successful")

print("\n✓ Diffusion test completed")
