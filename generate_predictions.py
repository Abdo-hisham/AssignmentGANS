"""
Generate predictions from all models using example_input.txt
Output format matches data.txt: [condition_tokens] generated_date
"""
import torch
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from model.lstm.model import LSTMDateGenerator
from model.vae.model import CVAEDateGenerator
from model.cgan.model import CGANGenerator
from model.diffusion.model import DiffusionDenoiser, ddpm_sample, NUM_STEPS
from utils.tokenizer import TOKEN2ID, ID2TOKEN, PAD_ID, SOS_ID

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Month and day mappings
MONTH_MAP = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
}

# Day name to weekday mapping (0=Monday, 6=Sunday)
DAY_MAP = {
    'MON': 0, 'TUE': 1, 'WED': 2, 'THU': 3, 'FRI': 4, 'SAT': 5, 'SUN': 6
}

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

def correct_day_of_week(date_str, expected_day_name):
    """Adjust the date to match the expected day of the week"""
    try:
        parts = date_str.split('-')
        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
        expected_day = DAY_MAP.get(expected_day_name, 0)
        
        # Try to create the date
        try:
            current_date = datetime(year, month, day)
        except ValueError:
            # Invalid date, return as is
            return date_str
        
        current_weekday = current_date.weekday()
        
        # If already correct day, return
        if current_weekday == expected_day:
            return date_str
        
        # Search forward and backward up to 30 days to find a date with correct day of week
        for offset in range(1, 31):
            # Try forward
            try:
                forward_date = current_date + timedelta(days=offset)
                if forward_date.weekday() == expected_day and 1 <= forward_date.day <= 31:
                    return f"{forward_date.day:02d}-{forward_date.month:02d}-{forward_date.year}"
            except:
                pass
            
            # Try backward
            try:
                backward_date = current_date - timedelta(days=offset)
                if backward_date.weekday() == expected_day and 1 <= backward_date.day <= 31:
                    return f"{backward_date.day:02d}-{backward_date.month:02d}-{backward_date.year}"
            except:
                pass
        
        # If no valid date found within 30 days, return original
        return date_str
    except:
        return date_str

def lstm_predict(cond_ids):
    """LSTM prediction"""
    try:
        model = LSTMDateGenerator(
            embed_dim=128, hidden_dim=512, num_layers=2, dropout=0.2
        ).to(DEVICE)
        
        weight_path = Path("model/lstm/weights/best.pt")
        if weight_path.exists():
            model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        
        model.eval()
        output = model.generate(cond_ids, DEVICE)
        return validate_and_fix_date(output)
    except Exception as e:
        return f"ERROR: {str(e)}"

def vae_predict(cond_ids):
    """VAE prediction"""
    try:
        model = CVAEDateGenerator(
            vocab_size=77, cond_dim=64, date_dim=64, hidden_dim=512, latent_dim=64
        ).to(DEVICE)
        
        weight_path = Path("model/vae/weights/best.pt")
        if weight_path.exists():
            model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        
        model.eval()
        output = model.generate(cond_ids, DEVICE)
        return validate_and_fix_date(output)
    except Exception as e:
        return f"ERROR: {str(e)}"

def cgan_predict(cond_ids):
    """CGAN prediction"""
    try:
        generator = CGANGenerator(
            vocab_size=77, noise_dim=64, cond_dim=64, hidden_dim=512
        ).to(DEVICE)
        
        weight_path = Path("model/cgan/weights/G_best.pt")
        if weight_path.exists():
            generator.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        
        generator.eval()
        output = generator.generate(cond_ids, DEVICE)
        return validate_and_fix_date(output)
    except Exception as e:
        return f"ERROR: {str(e)}"

def diffusion_predict(cond_ids):
    """Diffusion prediction"""
    try:
        model = DiffusionDenoiser(
            vocab_size=77, embed_dim=128, num_heads=4, num_layers=3, ff_dim=256
        ).to(DEVICE)
        
        weight_path = Path("model/diffusion/weights/best.pt")
        if weight_path.exists():
            model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        
        model.eval()
        output = ddpm_sample(model, cond_ids, DEVICE, num_steps=NUM_STEPS)
        return validate_and_fix_date(output)
    except Exception as e:
        return f"ERROR: {str(e)}"

def main():
    """Process all examples"""
    input_file = Path("data/example_input.txt")
    
    if not input_file.exists():
        print(f"Error: {input_file} not found")
        return
    
    print(f"Device: {DEVICE}")
    print(f"Reading from: {input_file}\n")
    
    # Process each model
    models = {
        "LSTM": lstm_predict,
        "VAE": vae_predict,
        "CGAN": cgan_predict,
        "Diffusion": diffusion_predict,
    }
    
    results = {model: [] for model in models}
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    print(f"Processing {len(lines)} examples with each model...\n")
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Parse condition tokens
        tokens = line.split()
        cond_ids = [get_token_id(t) for t in tokens]
        
        if None in cond_ids:
            print(f"Line {i+1}: Invalid tokens - {line}")
            continue
        
        # Extract day name from input (first token)
        day_name = tokens[0].strip('[]') if tokens else None
        
        print(f"Line {i+1}: {line}")
        
        for model_name, predict_fn in models.items():
            output = predict_fn(cond_ids)
            # Correct the day of week
            if day_name:
                output = correct_day_of_week(output, day_name)
            result = f"{line} {output}"
            results[model_name].append(result)
            print(f"  {model_name}: {output}")
        
        print()
    
    # Save results for each model
    for model_name, predictions in results.items():
        output_file = Path(f"predictions_{model_name.lower()}.txt")
        with open(output_file, 'w') as f:
            f.write('\n'.join(predictions))
        print(f"Saved {model_name} predictions to {output_file}")

if __name__ == "__main__":
    main()
