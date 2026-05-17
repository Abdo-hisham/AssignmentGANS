"""
Conditional GAN (CGAN) for date generation.

Generator  : [z (noise) + cond_embed] → MLP → 10 × softmax (Gumbel-Softmax)
Discriminator: [date_embed + cond_embed] → MLP → real/fake score

The date is represented as 10 one-hot-like soft vectors (Gumbel-Softmax trick)
to allow gradient flow through discrete tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from utils.tokenizer import (
    VOCAB_SIZE, TOKEN2ID, ID2TOKEN, encode_condition,
    tokenize_date, encode_date, decode_date_ids,
)

DATE_LEN  = 10
NOISE_DIM = 64


class CGANGenerator(nn.Module):

    def __init__(
        self,
        vocab_size:  int   = VOCAB_SIZE,
        noise_dim:   int   = NOISE_DIM,
        cond_dim:    int   = 64,
        hidden_dim:  int   = 512,
        dropout:     float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.date_len   = DATE_LEN
        self.noise_dim  = noise_dim

        self.cond_emb  = nn.Embedding(vocab_size, cond_dim)
        self.cond_proj = nn.Linear(cond_dim * 4, hidden_dim // 2)

        gen_in = noise_dim + hidden_dim // 2
        self.net = nn.Sequential(
            nn.Linear(gen_in, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size) for _ in range(DATE_LEN)
        ])

    def forward(self, z: torch.Tensor, cond_ids: torch.Tensor,
                temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
        """
        Returns (B, DATE_LEN, V) Gumbel-softmax soft one-hots.
        """
        c = F.relu(self.cond_proj(
            self.cond_emb(cond_ids).view(cond_ids.size(0), -1)
        ))
        h = self.net(torch.cat([z, c], dim=-1))
        soft_tokens = []
        for head in self.heads:
            logits = head(h)
            soft = F.gumbel_softmax(logits, tau=temperature, hard=hard)
            soft_tokens.append(soft)
        return torch.stack(soft_tokens, dim=1)   # (B, 10, V)

    @torch.no_grad()
    def generate(self, cond_ids: List[int], device: torch.device,
                 temperature: float = 0.8) -> str:
        self.eval()
        cond_t = torch.tensor([cond_ids], dtype=torch.long, device=device)
        z = torch.randn(1, self.noise_dim, device=device)
        soft = self.forward(z, cond_t, temperature=temperature, hard=True)
        token_ids = soft.argmax(dim=-1).squeeze(0).tolist()
        return decode_date_ids(token_ids)


class CGANDiscriminator(nn.Module):

    def __init__(
        self,
        vocab_size: int   = VOCAB_SIZE,
        cond_dim:   int   = 64,
        date_dim:   int   = 32,
        hidden_dim: int   = 256,
        dropout:    float = 0.2,
    ) -> None:
        super().__init__()
        self.cond_emb  = nn.Embedding(vocab_size, cond_dim)
        self.cond_proj = nn.Linear(cond_dim * 4, hidden_dim // 2)

        # Date input can be soft (continuous) or hard (one-hot); use linear projection
        self.date_proj = nn.Linear(vocab_size * DATE_LEN, hidden_dim // 2)

        dis_in = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(dis_in, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, date_soft: torch.Tensor,
                cond_ids: torch.Tensor) -> torch.Tensor:
        """
        date_soft: (B, DATE_LEN, V)  soft one-hots
        returns:   (B, 1)  logit (no sigmoid — use BCEWithLogitsLoss)
        """
        c = F.relu(self.cond_proj(
            self.cond_emb(cond_ids).view(cond_ids.size(0), -1)
        ))
        d = F.relu(self.date_proj(
            date_soft.view(date_soft.size(0), -1)
        ))
        return self.net(torch.cat([c, d], dim=-1))
