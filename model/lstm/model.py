"""
LSTM-based autoregressive sequence model for date generation.
Architecture: Embedding → LSTM → Linear projection → Softmax over vocab.
The model sees [SOS + conditions + SEP] then generates date tokens autoregressively.
"""

import torch
import torch.nn as nn
from typing import List, Optional

from utils.tokenizer import (
    VOCAB_SIZE, PAD_ID, SOS_ID, EOS_ID, SEP_ID,
    TOKEN2ID, ID2TOKEN, encode_condition,
    tokenize_date, encode_date, decode_date_ids,
)


class LSTMDateGenerator(nn.Module):

    def __init__(
        self,
        vocab_size:   int = VOCAB_SIZE,
        embed_dim:    int = 128,
        hidden_dim:   int = 512,
        num_layers:   int = 2,
        dropout:      float = 0.2,
        pad_id:       int = PAD_ID,
    ) -> None:
        super().__init__()
        self.pad_id     = pad_id
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout  = nn.Dropout(dropout)
        self.fc_out   = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,                          # (B, T)
        hidden: Optional[tuple] = None,
    ):
        emb = self.dropout(self.embedding(x))     # (B, T, E)
        out, hidden = self.lstm(emb, hidden)       # (B, T, H)
        logits = self.fc_out(self.dropout(out))    # (B, T, V)
        return logits, hidden

    @torch.no_grad()
    def generate(
        self,
        cond_ids: List[int],
        device: torch.device,
        max_new_tokens: int = 12,
        temperature: float = 0.8,
    ) -> str:
        self.eval()
        # Build prompt: [SOS] + cond + [SEP]
        prompt = [SOS_ID] + cond_ids + [SEP_ID]
        x = torch.tensor([prompt], dtype=torch.long, device=device)

        _, hidden = self.lstm(self.dropout(self.embedding(x)))
        generated: List[int] = []

        # Autoregressive decode
        next_tok = torch.tensor([[SEP_ID]], dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            emb = self.dropout(self.embedding(next_tok))
            out, hidden = self.lstm(emb, hidden)
            logits = self.fc_out(self.dropout(out))  # (1,1,V)
            logits = logits[:, -1, :] / temperature
            probs  = torch.softmax(logits, dim=-1)
            tok    = torch.multinomial(probs, 1).item()
            if tok == EOS_ID:
                break
            generated.append(tok)
            next_tok = torch.tensor([[tok]], dtype=torch.long, device=device)

        return decode_date_ids(generated)
