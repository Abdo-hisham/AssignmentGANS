"""
evaluate.py – Compute condition pass rate + generate loss plots.
Run: python model/evaluate.py --model lstm
"""

import sys, os, argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import matplotlib.pyplot as plt

from utils.dataset import split_dataset, get_dataloader
from utils.date_utils import validate_date
from utils.tokenizer import ID2TOKEN, encode_condition

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name: str, weights_path: str):
    if model_name == "lstm":
        from model.lstm.model import LSTMDateGenerator
        m = LSTMDateGenerator().to(DEVICE)
    elif model_name == "vae":
        from model.vae.model import CVAEDateGenerator
        m = CVAEDateGenerator().to(DEVICE)
    elif model_name == "cgan":
        from model.cgan.model import CGANGenerator
        m = CGANGenerator().to(DEVICE)
    elif model_name == "diffusion":
        from model.diffusion.model import DiffusionDenoiser
        m = DiffusionDenoiser().to(DEVICE)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    m.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    m.eval()
    return m


def generate_one(model, model_name, cond_ids):
    if model_name in ("lstm", "vae", "cgan"):
        return model.generate(cond_ids, DEVICE)
    elif model_name == "diffusion":
        from model.diffusion.model import ddpm_sample, NUM_STEPS
        return ddpm_sample(model, cond_ids, DEVICE, num_steps=NUM_STEPS)


def evaluate(model, model_name: str, loader, n: int = 1000):
    ok, tot = 0, 0
    examples_pass, examples_fail = [], []

    for cond_ids, _ in loader:
        if tot >= n:
            break
        cond_ids = cond_ids.to(DEVICE)
        for b in range(min(cond_ids.size(0), n - tot)):
            cids  = cond_ids[b].tolist()
            ctok  = [ID2TOKEN[i] for i in cids]
            out   = generate_one(model, model_name, cids)
            passed = validate_date(out, ctok)
            if passed:
                ok += 1
                if len(examples_pass) < 5:
                    examples_pass.append((ctok, out))
            else:
                if len(examples_fail) < 5:
                    examples_fail.append((ctok, out))
            tot += 1

    rate = ok / max(tot, 1)
    return rate, examples_pass, examples_fail


def plot_training_log(log_path: str, model_name: str, save_dir: Path):
    if not Path(log_path).exists():
        print(f"  No log found at {log_path}, skipping plot.")
        return
    with open(log_path) as f:
        log = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{model_name.upper()} Training Curves", fontsize=14)

    if "train_loss" in log and "val_loss" in log:
        axes[0].plot(log["train_loss"], label="Train Loss")
        axes[0].plot(log["val_loss"],   label="Val Loss")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss"); axes[0].legend()

    if model_name == "cgan" and "g_loss" in log:
        axes[0].plot(log["g_loss"], label="G Loss")
        axes[0].plot(log["d_loss"], label="D Loss")
        axes[0].legend()

    pr = [x for x in log.get("pass_rate", []) if x > 0]
    if pr:
        x_vals = [i * (len(log.get("train_loss", pr)) // len(pr))
                  for i in range(1, len(pr)+1)]
        axes[1].plot(x_vals, pr, marker="o", color="green")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Pass Rate")
        axes[1].set_title("Condition Pass Rate"); axes[1].set_ylim(0, 1)

    plt.tight_layout()
    out = save_dir / f"{model_name}_curves.png"
    plt.savefig(out, dpi=150)
    print(f"  Plot saved to {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm",
                        choices=["lstm","vae","cgan","diffusion"])
    parser.add_argument("--n",    type=int, default=500,
                        help="Number of samples to evaluate")
    args = parser.parse_args()

    weights_map = {
        "lstm":      "model/lstm/weights/best.pt",
        "vae":       "model/vae/weights/best.pt",
        "cgan":      "model/cgan/weights/G_best.pt",
        "diffusion": "model/diffusion/weights/best.pt",
    }
    log_map = {
        "lstm":      "model/lstm/train_log.json",
        "vae":       "model/vae/train_log.json",
        "cgan":      "model/cgan/train_log.json",
        "diffusion": "model/diffusion/train_log.json",
    }

    weights = ROOT / weights_map[args.model]
    if not weights.exists():
        print(f"[ERROR] No weights at {weights}. Train first.")
        sys.exit(1)

    model = load_model(args.model, str(weights))

    _, test_set = split_dataset("data/data.txt", seed=42, seq_mode=False)
    loader = get_dataloader(test_set, batch_size=64, shuffle=False)

    print(f"Evaluating {args.model.upper()} on {args.n} samples …")
    rate, passes, fails = evaluate(model, args.model, loader, n=args.n)
    print(f"\n  Condition Pass Rate: {rate*100:.2f}%")

    print("\n  ✓ Passing examples:")
    for ctok, out in passes:
        print(f"    {' '.join(ctok)} → {out}")

    print("\n  ✗ Failing examples:")
    for ctok, out in fails:
        print(f"    {' '.join(ctok)} → {out}  (invalid)")

    plots_dir = ROOT / "model" / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_training_log(str(ROOT / log_map[args.model]), args.model, plots_dir)


if __name__ == "__main__":
    main()
