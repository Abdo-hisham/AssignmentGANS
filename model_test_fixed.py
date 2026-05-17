"""
Fixed Interactive Model Tester
Tests each model with proper token validation and error handling
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model.lstm.model import LSTMDateGenerator
from model.vae.model import CVAEDateGenerator
from model.cgan.model import CGANGenerator
from model.diffusion.model import DiffusionDenoiser
from utils.tokenizer import TOKEN2ID, ID2TOKEN, PAD_ID, SOS_ID

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
print(f"\n{'='*70}")
print("Available Tokens in Vocabulary ({} total):".format(len(TOKEN2ID)))
print(f"{'='*70}")
for idx, token in enumerate(sorted(TOKEN2ID.keys())):
    print(f"  {TOKEN2ID[token]:2d}: {token}", end="   ")
    if (idx + 1) % 4 == 0:
        print()
print("\n")

def parse_tokens(input_str):
    """Parse token input string to token IDs"""
    tokens = input_str.strip().split()
    token_ids = []
    
    for t in tokens:
        # Try exact match first
        if t in TOKEN2ID:
            token_ids.append(TOKEN2ID[t])
        else:
            # Try integer parsing
            try:
                token_id = int(t)
                if 0 <= token_id < 77:
                    token_ids.append(token_id)
                else:
                    print(f"✗ Token ID {token_id} out of range [0-76]")
                    return None
            except ValueError:
                print(f"✗ Invalid token: '{t}'")
                return None
    
    return token_ids if len(token_ids) == 4 else None

def test_lstm():
    """Test LSTM model"""
    print("\n" + "="*70)
    print("  LSTM Model Tester")
    print("="*70)
    
    try:
        model = LSTMDateGenerator(
            embed_dim=128, hidden_dim=512, num_layers=2, dropout=0.2
        ).to(DEVICE)
        
        weight_path = Path("model/lstm/weights/best.pt")
        if weight_path.exists():
            model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
            print("✓ Loaded pre-trained weights")
        else:
            print("⚠ No pre-trained weights found")
        
        model.eval()
        
        print("\nEnter 4 condition token IDs (0-76, separated by space):")
        print("Example: '5 12 15 25' or use token names like '[MON] [JAN] 15 25'")
        cond_input = input("Enter conditions: ").strip()
        
        cond_ids = parse_tokens(cond_input)
        if cond_ids is None or len(cond_ids) != 4:
            print("✗ Please enter exactly 4 valid tokens")
            return
        
        print(f"Condition tokens: {cond_ids}")
        
        with torch.no_grad():
            output = model.generate(cond_ids, DEVICE)
        
        print(f"\n✓ Generated Date: {output}")
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")

def test_vae():
    """Test VAE model"""
    print("\n" + "="*70)
    print("  VAE Model Tester")
    print("="*70)
    
    try:
        # Use hyperparameters that match the trained weights
        model = CVAEDateGenerator(
            vocab_size=77, cond_dim=64, date_dim=64, hidden_dim=512, latent_dim=64
        ).to(DEVICE)
        
        weight_path = Path("model/vae/weights/best.pt")
        if weight_path.exists():
            model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
            print("✓ Loaded pre-trained weights")
        else:
            print("⚠ No pre-trained weights found")
        
        model.eval()
        
        print("\nEnter 4 condition token IDs (0-76, separated by space):")
        cond_input = input("Enter conditions: ").strip()
        
        cond_ids = parse_tokens(cond_input)
        if cond_ids is None or len(cond_ids) != 4:
            print("✗ Please enter exactly 4 valid tokens")
            return
        
        print(f"Condition tokens: {cond_ids}")
        
        cond_tensor = torch.tensor([cond_ids], dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            # Generate random latent vector
            z = torch.randn(1, 64).to(DEVICE)
            logits = model.decode(z, cond_tensor)
            output_tokens = torch.argmax(logits[0], dim=-1).tolist()
        
        print(f"\n✓ Generated date tokens: {output_tokens}")
        
        # Try to decode tokens to token names
        token_names = [ID2TOKEN.get(t, f"[tok_{t}]") for t in output_tokens]
        print(f"Token names: {token_names}")
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

def test_cgan():
    """Test CGAN model"""
    print("\n" + "="*70)
    print("  CGAN Model Tester")
    print("="*70)
    
    try:
        generator = CGANGenerator(
            vocab_size=77, noise_dim=64, cond_dim=64, hidden_dim=512
        ).to(DEVICE)
        
        weight_path = Path("model/cgan/weights/generator.pt")
        if weight_path.exists():
            generator.load_state_dict(torch.load(weight_path, map_location=DEVICE))
            print("✓ Loaded pre-trained generator")
        else:
            print("⚠ No pre-trained weights found")
        
        generator.eval()
        
        print("\nEnter 4 condition token IDs (0-76, separated by space):")
        cond_input = input("Enter conditions: ").strip()
        
        cond_ids = parse_tokens(cond_input)
        if cond_ids is None or len(cond_ids) != 4:
            print("✗ Please enter exactly 4 valid tokens")
            return
        
        print(f"Condition tokens: {cond_ids}")
        
        cond_tensor = torch.tensor([cond_ids], dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            z = torch.randn(1, 64).to(DEVICE)
            output = generator(z, cond_tensor)
            output_tokens = torch.argmax(output[0], dim=-1).tolist()
        
        print(f"\n✓ Generated output shape: {output.shape}")
        print(f"Generated date tokens: {output_tokens}")
        
        # Decode to token names
        token_names = [ID2TOKEN.get(t, f"[tok_{t}]") for t in output_tokens]
        print(f"Token names: {token_names}")
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")

def test_diffusion():
    """Test Diffusion model"""
    print("\n" + "="*70)
    print("  Diffusion Model Tester")
    print("="*70)
    
    try:
        # DiffusionDenoiser signature: vocab_size, embed_dim, num_heads, num_layers, ff_dim, time_dim, dropout
        model = DiffusionDenoiser(
            vocab_size=77, embed_dim=128, num_heads=4, num_layers=3, ff_dim=256
        ).to(DEVICE)
        
        weight_path = Path("model/diffusion/weights/best.pt")
        if weight_path.exists():
            model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
            print("✓ Loaded pre-trained weights")
        else:
            print("⚠ No pre-trained weights found")
        
        model.eval()
        
        print("\nEnter 4 condition token IDs (0-76, separated by space):")
        cond_input = input("Enter conditions: ").strip()
        
        cond_ids = parse_tokens(cond_input)
        if cond_ids is None or len(cond_ids) != 4:
            print("✗ Please enter exactly 4 valid tokens")
            return
        
        print(f"Condition tokens: {cond_ids}")
        print("Generating... (simulating reverse diffusion)")
        
        cond_tensor = torch.tensor([cond_ids], dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            # Start from random noise (B=1, DATE_LEN=10)
            x_t = torch.randint(0, 77, (1, 10)).to(DEVICE)
            
            # Reverse diffusion (simulate a few steps)
            for t in range(50, 0, -5):
                t_tensor = torch.tensor([t], dtype=torch.long).to(DEVICE)
                # model.forward returns list of DATE_LEN tensors, each (B, vocab_size)
                output_list = model(cond_tensor, x_t, t_tensor)
                # Convert to tokens (take argmax of each position)
                x_t = torch.stack([torch.argmax(o, dim=-1) for o in output_list], dim=1)
                print(f"  Step t={t}")
        
        output_tokens = x_t[0].tolist()
        print(f"\n✓ Generated date tokens: {output_tokens}")
        
        # Decode to token names
        token_names = [ID2TOKEN.get(t, f"[tok_{t}]") for t in output_tokens]
        print(f"Token names: {token_names}")
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main menu"""
    while True:
        print("\n" + "="*70)
        print("  Interactive Model Tester - GPU Enabled")
        print("="*70)
        print("\nAvailable Models:")
        print("1. LSTM")
        print("2. VAE (CVAE)")
        print("3. CGAN")
        print("4. Diffusion")
        print("5. Exit")
        
        choice = input("\nSelect model (1-5): ").strip()
        
        if choice == "1":
            test_lstm()
        elif choice == "2":
            test_vae()
        elif choice == "3":
            test_cgan()
        elif choice == "4":
            test_diffusion()
        elif choice == "5":
            print("\n✓ Goodbye!")
            break
        else:
            print("✗ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()
