"""
predict.py – Master inference script for the Dates Generator assignment.

Usage:
    python predict.py -i data/example_input.txt -o predictions.txt [--model lstm|vae|cgan|diffusion]

Default model: lstm (best balance of speed and accuracy).
Fallback: if model output is invalid, brute-force a valid date.
"""

import argparse
import sys
import os
from pathlib import Path

import torch

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.tokenizer import (
    parse_line, encode_condition, format_output_line, ID2TOKEN,
)
from utils.date_utils import validate_date, extract_conditions, find_valid_date

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_lstm(weights_path: str):
    from model.lstm.model import LSTMDateGenerator
    m = LSTMDateGenerator().to(DEVICE)
    m.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    m.eval()
    return m


def load_vae(weights_path: str):
    from model.vae.model import CVAEDateGenerator
    m = CVAEDateGenerator().to(DEVICE)
    m.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    m.eval()
    return m


def load_cgan(weights_path: str):
    from model.cgan.model import CGANGenerator
    m = CGANGenerator().to(DEVICE)
    m.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    m.eval()
    return m


def load_diffusion(weights_path: str):
    from model.diffusion.model import DiffusionDenoiser
    m = DiffusionDenoiser().to(DEVICE)
    m.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    m.eval()
    return m


MODEL_CONFIGS = {
    "lstm":      ("model/lstm/weights/best.pt",      load_lstm),
    "vae":       ("model/vae/weights/best.pt",       load_vae),
    "cgan":      ("model/cgan/weights/G_best.pt",    load_cgan),
    "diffusion": ("model/diffusion/weights/best.pt", load_diffusion),
}


def generate(model, model_name: str, cond_ids: list) -> str:
    if model_name == "lstm":
        return model.generate(cond_ids, DEVICE)
    elif model_name == "vae":
        return model.generate(cond_ids, DEVICE)
    elif model_name == "cgan":
        return model.generate(cond_ids, DEVICE)
    elif model_name == "diffusion":
        from model.diffusion.model import ddpm_sample, NUM_STEPS
        return ddpm_sample(model, cond_ids, DEVICE, num_steps=NUM_STEPS)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Dates Generator – inference")
    parser.add_argument("-i", "--input",  required=True, help="Path to input file")
    parser.add_argument("-o", "--output", required=True, help="Path to output file")
    parser.add_argument("--model", default="lstm",
                        choices=["lstm", "vae", "cgan", "diffusion"],
                        help="Which model to use for generation (default: lstm)")
    args = parser.parse_args()

    weights_rel, loader_fn = MODEL_CONFIGS[args.model]
    weights_abs = ROOT / weights_rel

    if not weights_abs.exists():
        print(f"[ERROR] Weights not found at {weights_abs}")
        print(f"  Please train first:  python model/{args.model}/train.py")
        sys.exit(1)

    print(f"Loading {args.model.upper()} from {weights_abs} …")
    model = loader_fn(str(weights_abs))

    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = input_path.read_text().strip().splitlines()
    results = []
    fallback_count = 0

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        cond_tokens, _ = parse_line(line)
        if len(cond_tokens) != 4:
            print(f"  [WARN] Line {i+1}: could not parse conditions, skipping.")
            continue

        cond_ids = encode_condition(cond_tokens)

        # Try model generation (up to 5 attempts)
        date_str = None
        for attempt in range(5):
            candidate = generate(model, args.model, cond_ids)
            if validate_date(candidate, cond_tokens):
                date_str = candidate
                break

        # Fallback: brute-force
        if date_str is None:
            day_name, month_num, want_leap, decade_start = extract_conditions(cond_tokens)
            date_str = find_valid_date(day_name, month_num, want_leap, decade_start)
            if date_str:
                fallback_count += 1
            else:
                date_str = "IMPOSSIBLE"   # no valid date exists for these conditions

        out_line = format_output_line(cond_tokens, date_str)
        results.append(out_line)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(lines)} …")

    output_path.write_text("\n".join(results) + "\n")
    print(f"\n✓ Done. {len(results)} predictions written to {output_path}")
    if fallback_count:
        print(f"  (Fallback used for {fallback_count} samples where model output was invalid)")


if __name__ == "__main__":
    main()
