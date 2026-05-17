"""
Interactive manual prediction - input condition tokens, save outputs to file
"""
import torch
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from model.lstm.model import LSTMDateGenerator
from model.vae.model import CVAEDateGenerator
from model.cgan.model import CGANGenerator
from model.diffusion.model import DiffusionDenoiser, ddpm_sample, NUM_STEPS
from utils.tokenizer import TOKEN2ID

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

def load_models():
    """Load all 4 models"""
    print("Loading models on GPU...")
    
    # LSTM
    lstm = LSTMDateGenerator(
        embed_dim=128, hidden_dim=512, num_layers=2, dropout=0.2
    ).to(DEVICE)
    lstm_path = Path("model/lstm/weights/best.pt")
    if lstm_path.exists():
        lstm.load_state_dict(torch.load(lstm_path, map_location=DEVICE))
    lstm.eval()
    
    # VAE
    vae = CVAEDateGenerator(
        vocab_size=77, cond_dim=64, date_dim=64, hidden_dim=512, latent_dim=64
    ).to(DEVICE)
    vae_path = Path("model/vae/weights/best.pt")
    if vae_path.exists():
        vae.load_state_dict(torch.load(vae_path, map_location=DEVICE))
    vae.eval()
    
    # CGAN
    cgan = CGANGenerator(
        vocab_size=77, noise_dim=64, cond_dim=64, hidden_dim=512
    ).to(DEVICE)
    cgan_path = Path("model/cgan/weights/G_best.pt")
    if cgan_path.exists():
        cgan.load_state_dict(torch.load(cgan_path, map_location=DEVICE))
    cgan.eval()
    
    # Diffusion
    diffusion = DiffusionDenoiser(
        vocab_size=77, embed_dim=128, num_heads=4, num_layers=3, ff_dim=256
    ).to(DEVICE)
    diffusion_path = Path("model/diffusion/weights/best.pt")
    if diffusion_path.exists():
        diffusion.load_state_dict(torch.load(diffusion_path, map_location=DEVICE))
    diffusion.eval()
    
    print("✓ All models loaded\n")
    return lstm, vae, cgan, diffusion

def validate_and_fix_date(date_str):
    """Validate date format DD-MM-YYYY and fix invalid values"""
    try:
        parts = date_str.split('-')
        if len(parts) != 3:
            return date_str
        
        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
        
        # Fix invalid day
        if day < 1 or day > 31:
            day = max(1, min(31, day))
        
        # Fix invalid month
        if month < 1 or month > 12:
            month = max(1, min(12, month))
        
        # Fix invalid year
        if year < 1800 or year > 2200:
            year = 2000
        
        return f"{day:02d}-{month:02d}-{year}"
    except:
        return date_str

def predict_with_models(cond_ids, models):
    """Get predictions from all 4 models"""
    lstm, vae, cgan, diffusion = models
    
    results = {}
    
    try:
        with torch.no_grad():
            output = lstm.generate(cond_ids, DEVICE)
            results['LSTM'] = validate_and_fix_date(output)
    except Exception as e:
        results['LSTM'] = f"ERROR: {str(e)}"
    
    try:
        with torch.no_grad():
            output = vae.generate(cond_ids, DEVICE)
            results['VAE'] = validate_and_fix_date(output)
    except Exception as e:
        results['VAE'] = f"ERROR: {str(e)}"
    
    try:
        with torch.no_grad():
            output = cgan.generate(cond_ids, DEVICE)
            results['CGAN'] = validate_and_fix_date(output)
    except Exception as e:
        results['CGAN'] = f"ERROR: {str(e)}"
    
    try:
        with torch.no_grad():
            output = ddpm_sample(diffusion, cond_ids, DEVICE, num_steps=NUM_STEPS)
            results['Diffusion'] = validate_and_fix_date(output)
    except Exception as e:
        results['Diffusion'] = f"ERROR: {str(e)}"
    
    return results

def main():
    """Main interactive loop"""
    print(f"Device: {DEVICE}\n")
    print("=" * 70)
    print("  Manual Date Prediction - Input & Save to File")
    print("=" * 70)
    
    models = load_models()
    
    # Create output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"manual_predictions_{timestamp}.txt")
    
    print(f"Output will be saved to: {output_file}\n")
    
    output_lines = []
    
    while True:
        print("\n" + "-" * 70)
        print("Enter 4 condition tokens (or 'quit' to exit):")
        print("Format: [DAY_OF_WEEK] [MONTH] [BOOL_FLAG] [YEAR_VALUE]")
        print("Example: [SAT] [FEB] [False] [212]")
        print("         or: 10 13 24 58 (token IDs)")
        
        user_input = input("\n> ").strip()
        
        if user_input.lower() in ('quit', 'q', 'exit'):
            break
        
        if not user_input:
            print("✗ Empty input, try again")
            continue
        
        tokens = user_input.split()
        
        if len(tokens) != 4:
            print(f"✗ Expected 4 tokens, got {len(tokens)}")
            continue
        
        cond_ids = [get_token_id(t) for t in tokens]
        
        if None in cond_ids:
            print("✗ Invalid token(s)")
            continue
        
        print(f"\n✓ Condition IDs: {cond_ids}")
        print("\nGenerating predictions...")
        
        results = predict_with_models(cond_ids, models)
        
        # Format output line
        input_str = " ".join(tokens)
        output_line = f"{input_str} | LSTM: {results['LSTM']} | VAE: {results['VAE']} | CGAN: {results['CGAN']} | Diffusion: {results['Diffusion']}"
        output_lines.append(output_line)
        
        # Display results
        print("\nResults:")
        print(f"  LSTM:      {results['LSTM']}")
        print(f"  VAE:       {results['VAE']}")
        print(f"  CGAN:      {results['CGAN']}")
        print(f"  Diffusion: {results['Diffusion']}")
        
        # Save to file
        with open(output_file, 'a') as f:
            f.write(output_line + "\n")
        
        print(f"\n✓ Saved to {output_file}")
    
    print("\n" + "=" * 70)
    print(f"Total predictions saved: {len(output_lines)}")
    print(f"Output file: {output_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
