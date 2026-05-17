"""
Debug CGAN output - show raw tokens and decoded dates
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model.cgan.model import CGANGenerator
from utils.tokenizer import TOKEN2ID, ID2TOKEN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_token_id(token_str):
    """Convert token string to ID"""
    if token_str in TOKEN2ID:
        return TOKEN2ID[token_str]
    try:
        token_id = int(token_str)
        if 0 <= token_id < 77:
            return token_id
        return None
    except:
        return None

# Test with first condition line
cond_input = "[SAT] [FEB] [False] [212]".split()
cond_ids = [get_token_id(t) for t in cond_input]

print(f"Input tokens: {cond_input}")
print(f"Condition IDs: {cond_ids}")
print()

# Load CGAN
generator = CGANGenerator(
    vocab_size=77, noise_dim=64, cond_dim=64, hidden_dim=512
).to(DEVICE)

weight_path = Path("model/cgan/weights/G_best.pt")
if weight_path.exists():
    generator.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    print("✓ Loaded pre-trained generator")
else:
    print(f"✗ Weights not found at {weight_path}")
    sys.exit(1)

generator.eval()

with torch.no_grad():
    cond_t = torch.tensor([cond_ids], dtype=torch.long, device=DEVICE)
    z = torch.randn(1, 64, device=DEVICE)
    soft = generator.forward(z, cond_t, temperature=0.8, hard=True)
    token_ids = soft.argmax(dim=-1).squeeze(0).tolist()

print(f"Raw token IDs: {token_ids}")
print(f"Token ID range: min={min(token_ids)}, max={max(token_ids)}")
print()

# Try to decode each token
print("Token decoding:")
for i, tok_id in enumerate(token_ids):
    tok_name = ID2TOKEN.get(tok_id, f"UNK_{tok_id}")
    print(f"  Position {i}: ID {tok_id:2d} → {tok_name}")

print()

# Full decode
decoded = ""
for i in token_ids:
    tok = ID2TOKEN.get(i, "UNK")
    if tok not in ("<PAD>", "<SOS>", "<EOS>", "<UNK>", "<SEP>"):
        decoded += tok

print(f"Decoded: {decoded}")

# Now test generate method
print("\n--- Using generate() method ---")
output = generator.generate(cond_ids, DEVICE)
print(f"Generated: {output}")
