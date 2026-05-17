"""
Interactive Model Tester
Choose a model and provide inputs manually to test inference
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

def get_token_id(token_str):
    """Convert token string to ID"""
    if token_str in TOKEN2ID:
        return TOKEN2ID[token_str]
    try:
        return int(token_str)
    except:
        return None

def test_lstm():
    """Test LSTM model"""
    print("\n" + "=" * 60)
    print("  LSTM Model Tester")
    print("=" * 60)
    
    model = LSTMDateGenerator(
        embed_dim=128, hidden_dim=512, num_layers=2, dropout=0.2
    ).to(DEVICE)
    
    weight_path = Path("model/lstm/weights/best.pt")
    if weight_path.exists():
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        print("✓ Loaded pre-trained weights")
    
    model.eval()
    
    # Get condition input
    print("\nEnter 4 condition tokens (separated by space):")
    print("Example: 'month day year_lo year_hi' or 'jan 15 20 25'")
    cond_input = input("Enter conditions: ").strip().split()
    
    if len(cond_input) != 4:
        print("✗ Please enter exactly 4 tokens")
        return
    
    try:
        cond_ids = [get_token_id(t) for t in cond_input]
        if None in cond_ids:
            print("✗ Invalid token(s)")
            return
        
        print(f"\nCondition tokens: {cond_ids}")
        
        with torch.no_grad():
            output = model.generate(cond_ids, DEVICE)
        
        print(f"\n✓ Generated Date: {output}")
    except Exception as e:
        print(f"✗ Error: {e}")

def test_vae():
    """Test VAE model"""
    print("\n" + "=" * 60)
    print("  VAE Model Tester")
    print("=" * 60)
    
    # Use hyperparameters that match the trained weights
    model = CVAEDateGenerator(
        vocab_size=77, cond_dim=64, date_dim=64, hidden_dim=512, latent_dim=64
    ).to(DEVICE)
    
    weight_path = Path("model/vae/weights/best.pt")
    if weight_path.exists():
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        print("✓ Loaded pre-trained weights")
    
    model.eval()
    
    print("\nEnter 4 condition tokens (separated by space):")
    print("Example: '[MON] [JAN] [15] [20]' or '[SAT] [FEB] [01] [21]'")
    cond_input = input("Enter conditions: ").strip().split()
    
    if len(cond_input) != 4:
        print("✗ Please enter exactly 4 tokens")
        return
    
    try:
        cond_ids = torch.tensor([get_token_id(t) for t in cond_input], dtype=torch.long).to(DEVICE)
        if None in cond_ids.tolist():
            print("✗ Invalid token(s)")
            return
        
        print(f"Condition tokens: {cond_ids.tolist()}")
        
        with torch.no_grad():
            # Generate random latent vector
            z = torch.randn(1, 64).to(DEVICE)
            logits = model.decode(z, cond_ids.unsqueeze(0))
        
        # Convert logits to tokens
        output_tokens = [torch.argmax(logit, dim=-1).item() for logit in logits]
        print(f"\n✓ Generated date tokens: {output_tokens}")
        
        # Try to decode tokens to date
        token_names = [ID2TOKEN.get(t, f"tok_{t}") for t in output_tokens]
        print(f"Token names: {token_names}")
    except Exception as e:
        print(f"✗ Error: {e}")

def test_cgan():
    """Test CGAN model"""
    print("\n" + "=" * 60)
    print("  CGAN Model Tester")
    print("=" * 60)
    
    try:
        generator = CGANGenerator(
            vocab_size=77, noise_dim=64, cond_dim=64, hidden_dim=512
        ).to(DEVICE)
        
        weight_path = Path("model/cgan/weights/generator.pt")
        if weight_path.exists():
            generator.load_state_dict(torch.load(weight_path, map_location=DEVICE))
            print("✓ Loaded pre-trained generator")
        
        generator.eval()
        
        print("\nEnter 4 condition tokens (separated by space):")
        print("Example: '[MON] [JAN] [15] [20]'")
        cond_input = input("Enter conditions: ").strip().split()
        
        if len(cond_input) != 4:
            print("✗ Please enter exactly 4 tokens")
            return
        
        cond_ids = torch.tensor([get_token_id(t) for t in cond_input], dtype=torch.long).to(DEVICE)
        if None in cond_ids.tolist():
            print("✗ Invalid token(s)")
            return
        
        print(f"Condition tokens: {cond_ids.tolist()}")
        
        with torch.no_grad():
            z = torch.randn(1, 64).to(DEVICE)
            output = generator(z, cond_ids.unsqueeze(0))
        
        print(f"\n✓ Generated output shape: {output.shape}")
        print(f"Output (argmax): {torch.argmax(output[0], dim=-1)[:10]}")
    except Exception as e:
        print(f"✗ Error: {e}")

def test_diffusion():
    """Test Diffusion model"""
    print("\n" + "=" * 60)
    print("  Diffusion Model Tester")
    print("=" * 60)
    
    try:
        model = DiffusionDenoiser(
            seq_len=10, vocab_size=77, embed_dim=128, hidden_dim=512
        ).to(DEVICE)
        
        weight_path = Path("model/diffusion/weights/best.pt")
        if weight_path.exists():
            model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
            print("✓ Loaded pre-trained weights")
        
        model.eval()
        
        print("\nEnter 4 condition tokens (separated by space):")
        print("Example: '[MON] [JAN] [15] [20]'")
        cond_input = input("Enter conditions: ").strip().split()
        
        if len(cond_input) != 4:
            print("✗ Please enter exactly 4 tokens")
            return
        
        cond_ids = torch.tensor([get_token_id(t) for t in cond_input], dtype=torch.long).to(DEVICE)
        if None in cond_ids.tolist():
            print("✗ Invalid token(s)")
            return
        
        print(f"Condition tokens: {cond_ids.tolist()}")
        print("Generating... (this may take a moment)")
        
        with torch.no_grad():
            # Start from random noise
            x_t = torch.randint(0, 77, (1, 10)).to(DEVICE)
            
            # Reverse diffusion (simulate a few steps)
            for t in range(min(10, 50), 0, -5):
                t_tensor = torch.tensor([t], dtype=torch.long).to(DEVICE)
                output = model(x_t, t_tensor, cond_ids.unsqueeze(0))
                print(f"  Step t={t}")
        
        print(f"\n✓ Generated output shape: {output.shape}")
        output_tokens = torch.argmax(output, dim=-1)[0].tolist()
        print(f"Output tokens: {output_tokens}")
    except Exception as e:
        print(f"✗ Error: {e}")

def main():
    """Main menu"""
    while True:
        print("\n" + "=" * 60)
        print("  Interactive Model Tester")
        print("=" * 60)
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
            print("\nGoodbye!")
            break
        else:
            print("✗ Invalid choice")

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Available tokens: {list(TOKEN2ID.keys())[:20]}... (total: {len(TOKEN2ID)})")
    main()
