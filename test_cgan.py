"""Test CGAN model individually"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model.cgan.model import Generator, Discriminator
from utils.tokenizer import PAD_ID

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NOISE_DIM = 100
COND_DIM = 4
SEQ_LEN = 20
VOCAB_SIZE = 128
WEIGHT_PATH_G = Path("model/cgan/weights/generator.pt")
WEIGHT_PATH_D = Path("model/cgan/weights/discriminator.pt")

print(f"Device: {DEVICE}")
print(f"Testing CGAN Model...")

# Create models
generator = Generator(
    noise_dim=NOISE_DIM,
    cond_dim=COND_DIM,
    seq_len=SEQ_LEN,
    vocab_size=VOCAB_SIZE,
).to(DEVICE)

discriminator = Discriminator(
    seq_len=SEQ_LEN,
    vocab_size=VOCAB_SIZE,
    cond_dim=COND_DIM,
).to(DEVICE)

# Load weights if exist
if WEIGHT_PATH_G.exists():
    generator.load_state_dict(torch.load(WEIGHT_PATH_G, map_location=DEVICE))
    print(f"✓ Loaded generator from {WEIGHT_PATH_G}")

if WEIGHT_PATH_D.exists():
    discriminator.load_state_dict(torch.load(WEIGHT_PATH_D, map_location=DEVICE))
    print(f"✓ Loaded discriminator from {WEIGHT_PATH_D}")

generator.eval()
discriminator.eval()

# Test inference
print("\n--- Testing Inference ---")
with torch.no_grad():
    z = torch.randn(1, NOISE_DIM).to(DEVICE)
    cond = torch.randint(0, VOCAB_SIZE, (1, COND_DIM)).to(DEVICE)
    
    fake_dates = generator(z, cond)
    print(f"Generator output shape: {fake_dates.shape}")
    
    disc_score = discriminator(fake_dates, cond)
    print(f"Discriminator score shape: {disc_score.shape}")
    print(f"✓ CGAN inference successful")

print("\n✓ CGAN test completed")
