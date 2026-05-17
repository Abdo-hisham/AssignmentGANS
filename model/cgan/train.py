"""
Training script for CGAN date generator.
Run: python model/cgan/train.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from pathlib import Path
import json

from utils.dataset import split_dataset, get_dataloader, DatesDataset
from utils.date_utils import validate_date
from utils.tokenizer import (
    ID2TOKEN, VOCAB_SIZE, TOKEN2ID,
    encode_condition, tokenize_date, encode_date,
)
from model.cgan.model import CGANGenerator, CGANDiscriminator, DATE_LEN, NOISE_DIM

# ── Hyper-parameters ──────────────────────────────────────────────────────────
DATA_PATH   = "data/data.txt"
SAVE_DIR    = Path("model/cgan/weights")
LOG_PATH    = Path("model/cgan/train_log.json")
EPOCHS      = 40
BATCH_SIZE  = 512
LR_G        = 2e-4
LR_D        = 1e-4
N_D_STEPS   = 2          # discriminator updates per generator update
TEMP_START  = 2.0        # Gumbel-softmax temperature (anneal down)
TEMP_END    = 0.5
SEED        = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────────


def real_date_to_soft(date_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Convert integer date ids (B,10) to one-hot soft (B,10,V)."""
    return F.one_hot(date_ids, num_classes=vocab_size).float()


def pass_rate(gen, loader, device, n=500):
    gen.eval()
    ok, tot = 0, 0
    with torch.no_grad():
        for cond_ids, _ in loader:
            if tot >= n:
                break
            cond_ids = cond_ids.to(device)
            for b in range(min(cond_ids.size(0), n - tot)):
                cids = cond_ids[b].tolist()
                out  = gen.generate(cids, device)
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

    G = CGANGenerator(hidden_dim=512).to(DEVICE)
    D = CGANDiscriminator().to(DEVICE)

    opt_G = Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.999))
    opt_D = Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    log = {"g_loss": [], "d_loss": [], "pass_rate": []}
    best_pr = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Anneal Gumbel temperature
        temp = TEMP_START + (TEMP_END - TEMP_START) * (epoch / EPOCHS)

        G.train(); D.train()
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0

        for cond_ids, date_ids in train_loader:
            cond_ids, date_ids = cond_ids.to(DEVICE), date_ids.to(DEVICE)
            B = cond_ids.size(0)

            real_soft = real_date_to_soft(date_ids, VOCAB_SIZE)  # (B,10,V)

            # ── Train Discriminator ──
            for _ in range(N_D_STEPS):
                z = torch.randn(B, NOISE_DIM, device=DEVICE)
                fake_soft = G(z, cond_ids, temperature=temp, hard=False).detach()

                real_label = torch.ones(B,  1, device=DEVICE) * 0.9   # label smoothing
                fake_label = torch.zeros(B, 1, device=DEVICE)

                d_real = D(real_soft, cond_ids)
                d_fake = D(fake_soft, cond_ids)
                d_loss = criterion(d_real, real_label) + criterion(d_fake, fake_label)

                opt_D.zero_grad()
                d_loss.backward()
                nn.utils.clip_grad_norm_(D.parameters(), 1.0)
                opt_D.step()

            # ── Train Generator ──
            z = torch.randn(B, NOISE_DIM, device=DEVICE)
            fake_soft = G(z, cond_ids, temperature=temp, hard=False)
            g_loss = criterion(D(fake_soft, cond_ids),
                               torch.ones(B, 1, device=DEVICE))

            opt_G.zero_grad()
            g_loss.backward()
            nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt_G.step()

            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()

        g_loss_epoch /= len(train_loader)
        d_loss_epoch /= len(train_loader)

        pr = 0.0
        if epoch % 5 == 0:
            pr = pass_rate(G, test_loader, DEVICE)
            print(f"Epoch {epoch:3d} | G={g_loss_epoch:.4f} D={d_loss_epoch:.4f}"
                  f" temp={temp:.2f} pass={pr:.3f}")
            if pr > best_pr:
                best_pr = pr
                torch.save(G.state_dict(), SAVE_DIR / "G_best.pt")
                torch.save(D.state_dict(), SAVE_DIR / "D_best.pt")
        else:
            print(f"Epoch {epoch:3d} | G={g_loss_epoch:.4f} D={d_loss_epoch:.4f} temp={temp:.2f}")

        log["g_loss"].append(g_loss_epoch)
        log["d_loss"].append(d_loss_epoch)
        log["pass_rate"].append(pr)

    torch.save(G.state_dict(), SAVE_DIR / "G_last.pt")
    torch.save(D.state_dict(), SAVE_DIR / "D_last.pt")
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print("✓ CGAN training complete.")


if __name__ == "__main__":
    train()
