"""Test LSTM model individually"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model.lstm.model import LSTMDateGenerator
from utils.tokenizer import SOS_ID, PAD_ID

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 128
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.2
WEIGHT_PATH = Path("model/lstm/weights/best.pt")

print(f"Device: {DEVICE}")
print(f"Testing LSTM Model...")

# Create model
model = LSTMDateGenerator(
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
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
    test_cond = torch.tensor([1, 2, 3, 4], dtype=torch.long).to(DEVICE)
    output = model.generate(test_cond.tolist(), DEVICE)
    print(f"Generated date: {output}")

print("\n✓ LSTM test completed")
