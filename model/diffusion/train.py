"""
Training script for Diffusion date generator.
Run: python model/diffusion/train.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import json

from utils.dataset import split_dataset, get_dataloader
from utils.date_utils import validate_date
from utils.tokenizer import ID2TOKEN, VOCAB_SIZE
from model.diffusion.model import DiffusionDenoiser, corrupt_tokens, ddpm_sample, NUM_STEPS

# ── Hyper-parameters ──────────────────────────────────────────────────────────
DATA_PATH  = "data/data.txt"
SAVE_DIR   = Path("model/diffusion/weights")
LOG_PATH   = Path("model/diffusion/train_log.json")
EPOCHS     = 40
BATCH_SIZE = 256
LR         = 1e-3
EMBED_DIM  = 128
SEED       = 42
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────────


def pass_rate(model, loader, device, n=300):
    model.eval()
    ok, tot = 0, 0
    for cond_ids, _ in loader:
        if tot >= n:
            break
        cond_ids = cond_ids.to(device)
        for b in range(min(cond_ids.size(0), n - tot)):
            cids = cond_ids[b].tolist()
            out  = ddpm_sample(model, cids, device, num_steps=NUM_STEPS)
            ctok = [ID2TOKEN[i] for i in cids]
            if validate_date(out, ctok):
                ok += 1
            tot += 1
    return ok / max(tot, 1)


def train():
    torch.manual_seed(SEED)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    train_set, test_set = split_dataset(DATA_PATH, seed=SEED, seq_mode=False)
    train_loader = get_dataloader(train_set, BATCH_SIZE, shuffle=True)
    test_loader  = get_dataloader(test_set,  BATCH_SIZE, shuffle=False)

    model = DiffusionDenoiser(embed_dim=EMBED_DIM).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    log = {"train_loss": [], "val_loss": [], "pass_rate": []}
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss = 0.0
        for cond_ids, date_ids in train_loader:
            cond_ids, date_ids = cond_ids.to(DEVICE), date_ids.to(DEVICE)
            B = cond_ids.size(0)

            # Sample random timesteps
            t = torch.randint(1, NUM_STEPS + 1, (B,), device=DEVICE)
            noisy = corrupt_tokens(date_ids, t, NUM_STEPS, VOCAB_SIZE)

            logits_list = model(cond_ids, noisy, t)

            loss = torch.tensor(0.0, device=DEVICE)
            for pos, logits in enumerate(logits_list):
                loss = loss + criterion(logits, date_ids[:, pos])

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()

        t_loss /= len(train_loader)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for cond_ids, date_ids in test_loader:
                cond_ids, date_ids = cond_ids.to(DEVICE), date_ids.to(DEVICE)
                B = cond_ids.size(0)
                t = torch.randint(1, NUM_STEPS + 1, (B,), device=DEVICE)
                noisy = corrupt_tokens(date_ids, t, NUM_STEPS, VOCAB_SIZE)
                logits_list = model(cond_ids, noisy, t)
                loss = torch.tensor(0.0, device=DEVICE)
                for pos, logits in enumerate(logits_list):
                    loss = loss + criterion(logits, date_ids[:, pos])
                v_loss += loss.item()
        v_loss /= len(test_loader)
        scheduler.step()

        pr = 0.0
        if epoch % 5 == 0:
            pr = pass_rate(model, test_loader, DEVICE)
            print(f"Epoch {epoch:3d} | train={t_loss:.4f} val={v_loss:.4f} pass={pr:.3f}")
            if v_loss < best_val:
                best_val = v_loss
                torch.save(model.state_dict(), SAVE_DIR / "best.pt")
        else:
            print(f"Epoch {epoch:3d} | train={t_loss:.4f} val={v_loss:.4f}")

        log["train_loss"].append(t_loss)
        log["val_loss"].append(v_loss)
        log["pass_rate"].append(pr)

    torch.save(model.state_dict(), SAVE_DIR / "last.pt")
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print("✓ Diffusion training complete.")


if __name__ == "__main__":
    train()
