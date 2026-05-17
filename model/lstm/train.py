"""
Training script for the LSTM date generator.
Run: python model/lstm/train.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import json, time

from utils.dataset import split_dataset, get_dataloader
from utils.tokenizer import PAD_ID, SOS_ID, SEP_ID, EOS_ID, encode_condition
from utils.date_utils import validate_date, extract_conditions, find_valid_date
from model.lstm.model import LSTMDateGenerator

# ── Hyper-parameters ──────────────────────────────────────────────────────────
DATA_PATH   = "data/data.txt"
SAVE_DIR    = Path("model/lstm/weights")
LOG_PATH    = Path("model/lstm/train_log.json")
EPOCHS      = 30
BATCH_SIZE  = 512
LR          = 3e-3
EMBED_DIM   = 128
HIDDEN_DIM  = 512
NUM_LAYERS  = 2
DROPOUT     = 0.2
SEED        = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────────


def condition_pass_rate(model, test_loader, device, n_samples=500):
    """Fraction of generated dates that pass all conditions."""
    from utils.dataset import DatesDatasetSeq
    model.eval()
    correct = 0
    total   = 0
    for cond_ids, date_ids in test_loader:
        if total >= n_samples:
            break
        for b in range(min(cond_ids.size(0), n_samples - total)):
            c_ids = cond_ids[b].tolist()          # may be seq-format; take first 4
            # get raw condition ids (positions 1..4 after SOS)
            raw_cond = c_ids[1:5]                 # [SOS cond[0..3] SEP ...]
            gen = model.generate(raw_cond, device)
            # rebuild cond tokens for validation
            from utils.tokenizer import ID2TOKEN
            cond_toks = [ID2TOKEN[i] for i in raw_cond]
            if validate_date(gen, cond_toks):
                correct += 1
            total += 1
    return correct / max(total, 1)


def train():
    torch.manual_seed(SEED)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    train_set, test_set = split_dataset(DATA_PATH, train_ratio=0.9,
                                        seed=SEED, seq_mode=True)
    train_loader = get_dataloader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = get_dataloader(test_set,  batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMDateGenerator(
        embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, dropout=DROPOUT,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    log = {"train_loss": [], "val_loss": [], "pass_rate": []}
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        total_loss = 0.0
        for inp, tgt in train_loader:
            inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(inp)          # (B, T, V)
            B, T, V  = logits.shape
            loss = criterion(logits.view(B * T, V), tgt.view(B * T))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, tgt in test_loader:
                inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
                logits, _ = model(inp)
                B, T, V = logits.shape
                val_loss += criterion(logits.view(B * T, V), tgt.view(B * T)).item()
        val_loss /= len(test_loader)

        scheduler.step(val_loss)

        # ── Condition pass rate (every 5 epochs) ──
        pr = 0.0
        if epoch % 5 == 0:
            pr = condition_pass_rate(model, test_loader, DEVICE)
            print(f"Epoch {epoch:3d} | train={train_loss:.4f} val={val_loss:.4f} pass={pr:.3f}")
        else:
            print(f"Epoch {epoch:3d} | train={train_loss:.4f} val={val_loss:.4f}")

        log["train_loss"].append(train_loss)
        log["val_loss"].append(val_loss)
        log["pass_rate"].append(pr)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), SAVE_DIR / "best.pt")

    torch.save(model.state_dict(), SAVE_DIR / "last.pt")
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print("✓ LSTM training complete. Weights saved to", SAVE_DIR)


if __name__ == "__main__":
    train()
