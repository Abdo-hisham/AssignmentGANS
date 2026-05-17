"""
Conditional Variational Autoencoder (CVAE) for date generation.

Architecture:
  Encoder : [cond_embed + date_embed] → MLP → (mu, log_var)
  Decoder : [z + cond_embed]          → MLP → 10 × (logits over date-digit vocab)

The date is represented as 10 discrete tokens: dd-mm-yyyy
  positions: [d0,d1,'-',m0,m1,'-',y0,y1,y2,y3]
Each position is a separate classification head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

from utils.tokenizer import (
    VOCAB_SIZE, PAD_ID, TOKEN2ID, ID2TOKEN,
    encode_condition, tokenize_date, encode_date, decode_date_ids,
)

DATE_LEN = 10   # characters in dd-mm-yyyy tokenised form


class CVAEDateGenerator(nn.Module):

    def __init__(
        self,
        vocab_size:   int   = VOCAB_SIZE,
        cond_dim:     int   = 64,
        date_dim:     int   = 64,
        hidden_dim:   int   = 256,
        latent_dim:   int   = 32,
        dropout:      float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.date_len   = DATE_LEN

        # Condition embedding (4 tokens → single vector)
        self.cond_emb = nn.Embedding(vocab_size, cond_dim)
        self.cond_proj = nn.Linear(cond_dim * 4, hidden_dim // 2)

        # Date embedding (10 tokens → single vector) — encoder only
        self.date_emb = nn.Embedding(vocab_size, date_dim)
        self.date_proj = nn.Linear(date_dim * DATE_LEN, hidden_dim // 2)

        # Encoder
        enc_in = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu      = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        dec_in = latent_dim + hidden_dim // 2
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # One head per date position
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size) for _ in range(DATE_LEN)
        ])

    def encode(self, cond_ids: torch.Tensor,
               date_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """cond_ids: (B,4)  date_ids: (B,10)"""
        c = self.cond_emb(cond_ids).view(cond_ids.size(0), -1)
        c = F.relu(self.cond_proj(c))                   # (B, H/2)
        d = self.date_emb(date_ids).view(date_ids.size(0), -1)
        d = F.relu(self.date_proj(d))                   # (B, H/2)
        h = self.encoder(torch.cat([c, d], dim=-1))     # (B, H)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu: torch.Tensor,
                       log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor,
               cond_ids: torch.Tensor) -> List[torch.Tensor]:
        """Returns list of DATE_LEN logit tensors, each (B, V)."""
        c = self.cond_emb(cond_ids).view(cond_ids.size(0), -1)
        c = F.relu(self.cond_proj(c))
        h = self.decoder(torch.cat([z, c], dim=-1))
        return [head(h) for head in self.heads]

    def forward(self, cond_ids: torch.Tensor,
                date_ids: torch.Tensor):
        mu, log_var = self.encode(cond_ids, date_ids)
        z = self.reparameterize(mu, log_var)
        logits_list = self.decode(z, cond_ids)
        return logits_list, mu, log_var

    @torch.no_grad()
    def generate(self, cond_ids: List[int], device: torch.device,
                 temperature: float = 0.8) -> str:
        self.eval()
        cond_t = torch.tensor([cond_ids], dtype=torch.long, device=device)
        z = torch.randn(1, self.latent_dim, device=device)
        logits_list = self.decode(z, cond_t)
        tokens: List[int] = []
        for logits in logits_list:
            logits = logits / temperature
            tok = torch.multinomial(torch.softmax(logits, -1), 1).item()
            tokens.append(tok)
        return decode_date_ids(tokens)


def cvae_loss(logits_list: List[torch.Tensor],
              date_ids: torch.Tensor,
              mu: torch.Tensor,
              log_var: torch.Tensor,
              beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ELBO loss: reconstruction + beta * KL."""
    recon = torch.tensor(0.0, device=date_ids.device)
    for pos, logits in enumerate(logits_list):
        recon = recon + F.cross_entropy(logits, date_ids[:, pos])
    kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon + beta * kl, recon, kl
