"""
Discrete Diffusion model for date generation (D3PM-inspired, simplified).

We treat the date as 10 discrete tokens. Forward process corrupts tokens
with noise (random uniform replacement). Reverse process denoises conditioned
on conditions.

Architecture: condition + noisy-date → Transformer → clean-date logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List

from utils.tokenizer import (
    VOCAB_SIZE, PAD_ID, TOKEN2ID, ID2TOKEN,
    encode_condition, tokenize_date, encode_date, decode_date_ids,
)

DATE_LEN    = 10
NUM_STEPS   = 200      # diffusion timesteps


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        angles = t[:, None].float() * freqs[None]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class DiffusionDenoiser(nn.Module):
    """
    Transformer-based denoiser.
    Input  : noisy date tokens + condition tokens + timestep
    Output : clean date logits  (DATE_LEN × vocab_size)
    """

    def __init__(
        self,
        vocab_size:   int = VOCAB_SIZE,
        embed_dim:    int = 128,
        num_heads:    int = 4,
        num_layers:   int = 3,
        ff_dim:       int = 256,
        time_dim:     int = 64,
        dropout:      float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        self.date_len   = DATE_LEN

        self.tok_emb  = nn.Embedding(vocab_size, embed_dim)
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_proj = nn.Linear(time_dim, embed_dim)

        # Sequence: 4 cond tokens + DATE_LEN noisy tokens = 14 tokens
        SEQ_LEN = 4 + DATE_LEN
        self.pos_emb = nn.Embedding(SEQ_LEN, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.heads = nn.ModuleList([
            nn.Linear(embed_dim, vocab_size) for _ in range(DATE_LEN)
        ])

    def forward(self, cond_ids: torch.Tensor,
                noisy_date: torch.Tensor,
                t: torch.Tensor) -> List[torch.Tensor]:
        """
        cond_ids:   (B, 4)
        noisy_date: (B, DATE_LEN)  integer token ids
        t:          (B,)           integer timesteps
        Returns: list of DATE_LEN tensors, each (B, V)
        """
        B = cond_ids.size(0)
        seq = torch.cat([cond_ids, noisy_date], dim=1)   # (B, 14)
        pos = torch.arange(seq.size(1), device=seq.device).unsqueeze(0)

        x = self.tok_emb(seq) + self.pos_emb(pos)        # (B, 14, E)
        te = self.time_proj(self.time_emb(t)).unsqueeze(1)  # (B, 1, E)
        x = x + te

        out = self.transformer(x)                         # (B, 14, E)
        date_out = out[:, 4:, :]                          # (B, DATE_LEN, E)

        return [self.heads[i](date_out[:, i, :]) for i in range(self.date_len)]


# ── Forward / Reverse diffusion helpers ───────────────────────────────────────

def corrupt_tokens(date_ids: torch.Tensor, t: torch.Tensor,
                   num_steps: int, vocab_size: int) -> torch.Tensor:
    """
    Apply forward diffusion: at step t, each token is independently
    replaced by a uniform random token with probability t/T.
    """
    noise_prob = (t.float() / num_steps).view(-1, 1)   # (B, 1)
    mask = torch.bernoulli(noise_prob.expand_as(date_ids))
    random_toks = torch.randint(0, vocab_size, date_ids.shape, device=date_ids.device)
    return torch.where(mask.bool(), random_toks, date_ids)


@torch.no_grad()
def ddpm_sample(model: DiffusionDenoiser, cond_ids: List[int],
                device: torch.device, num_steps: int = NUM_STEPS,
                temperature: float = 0.8) -> str:
    """Ancestral sampling from T→0."""
    model.eval()
    cond_t = torch.tensor([cond_ids], dtype=torch.long, device=device)
    vocab_size = model.vocab_size

    # Start from pure noise
    x = torch.randint(0, vocab_size, (1, DATE_LEN), device=device)

    step_size = max(1, num_steps // 50)  # use 50 denoising steps
    for t_val in range(num_steps, 0, -step_size):
        t = torch.tensor([t_val], dtype=torch.long, device=device)
        logits_list = model(cond_t, x, t)
        new_x = []
        noise_prob = t_val / num_steps
        for i, logits in enumerate(logits_list):
            logits = logits / temperature
            tok = torch.multinomial(torch.softmax(logits, -1), 1).item()
            # With some probability keep noise (annealing)
            if torch.rand(1).item() < noise_prob * 0.1:
                tok = torch.randint(0, vocab_size, (1,)).item()
            new_x.append(tok)
        x = torch.tensor([new_x], dtype=torch.long, device=device)

    return decode_date_ids(x.squeeze(0).tolist())
