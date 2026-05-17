"""
PyTorch Dataset for the Dates Generator problem.
"""

import random
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from utils.tokenizer import (
    parse_line, build_sequence, encode_condition, tokenize_date,
    encode_date, VOCAB_SIZE, PAD_ID, SOS_ID, EOS_ID, SEP_ID,
    TOKEN2ID,
)
from utils.date_utils import extract_conditions


class DatesDataset(Dataset):
    """
    Returns (condition_ids, date_ids) tensors for each sample.
    condition_ids: length-4 tensor of condition token ids.
    date_ids:      length-10 tensor of date character token ids (dd-mm-yyyy).
    """

    def __init__(self, filepath: str, max_samples: Optional[int] = None) -> None:
        self.samples: List[Tuple[List[int], List[int]]] = []
        path = Path(filepath)
        with open(path, "r") as f:
            lines = f.readlines()

        if max_samples:
            lines = lines[:max_samples]

        for line in lines:
            line = line.strip()
            if not line:
                continue
            cond_tokens, date_str = parse_line(line)
            if date_str is None or len(cond_tokens) != 4:
                continue
            cond_ids = encode_condition(cond_tokens)
            date_toks = tokenize_date(date_str)
            date_ids = encode_date(date_toks)
            self.samples.append((cond_ids, date_ids))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cond_ids, date_ids = self.samples[idx]
        return (
            torch.tensor(cond_ids, dtype=torch.long),
            torch.tensor(date_ids, dtype=torch.long),
        )


class DatesDatasetSeq(Dataset):
    """
    Sequence version used by LSTM.
    Returns (input_seq, target_seq) where:
      input_seq  = [SOS] + cond_ids + [SEP] + date_ids (teacher-forced input)
      target_seq = cond_ids + [SEP] + date_ids + [EOS]
    """

    def __init__(self, filepath: str, max_samples: Optional[int] = None) -> None:
        self.samples: List[Tuple[List[int], List[int]]] = []
        path = Path(filepath)
        with open(path, "r") as f:
            lines = f.readlines()

        if max_samples:
            lines = lines[:max_samples]

        for line in lines:
            line = line.strip()
            if not line:
                continue
            cond_tokens, date_str = parse_line(line)
            if date_str is None or len(cond_tokens) != 4:
                continue
            full_seq = build_sequence(cond_tokens, date_str,
                                      add_sos=True, add_eos=True)
            input_seq  = full_seq[:-1]
            target_seq = full_seq[1:]
            self.samples.append((input_seq, target_seq))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inp, tgt = self.samples[idx]
        return (
            torch.tensor(inp,  dtype=torch.long),
            torch.tensor(tgt,  dtype=torch.long),
        )


def get_dataloader(dataset: Dataset, batch_size: int = 256,
                   shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def split_dataset(filepath: str, train_ratio: float = 0.9,
                  seed: int = 42, seq_mode: bool = False):
    """Split into train/test and return two Dataset objects."""
    cls = DatesDatasetSeq if seq_mode else DatesDataset
    full = cls(filepath)
    n = len(full)
    n_train = int(n * train_ratio)
    n_test  = n - n_train
    return torch.utils.data.random_split(
        full, [n_train, n_test],
        generator=torch.Generator().manual_seed(seed),
    )
