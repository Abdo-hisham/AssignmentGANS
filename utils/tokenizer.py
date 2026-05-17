"""
Custom tokenizer for the Dates Generator problem.
Converts conditions and dates into token sequences.
"""

from typing import List, Tuple, Dict, Optional
import re

# --- Vocabulary ---
DAYS   = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
LEAPS  = ["False", "True"]

# Decades: 180 to 219  (1800-2199)
DECADES = [str(d) for d in range(180, 220)]

# Date digits: day (2), month (2), year (4)  → 8 digit positions, each 0-9
DIGITS = [str(d) for d in range(10)]

# Special tokens
PAD   = "<PAD>"
SOS   = "<SOS>"
EOS   = "<EOS>"
UNK   = "<UNK>"
SEP   = "<SEP>"

# Build full vocab
CONDITION_TOKENS = (
    [f"[{d}]" for d in DAYS] +
    [f"[{m}]" for m in MONTHS] +
    ["[False]", "[True]"] +
    [f"[{dec}]" for dec in DECADES]
)

DATE_TOKENS = DIGITS + ["-"]

SPECIAL_TOKENS = [PAD, SOS, EOS, UNK, SEP]

ALL_TOKENS = SPECIAL_TOKENS + CONDITION_TOKENS + DATE_TOKENS

TOKEN2ID: Dict[str, int] = {tok: i for i, tok in enumerate(ALL_TOKENS)}
ID2TOKEN: Dict[int, str]  = {i: tok for tok, i in TOKEN2ID.items()}

VOCAB_SIZE = len(ALL_TOKENS)

PAD_ID = TOKEN2ID[PAD]
SOS_ID = TOKEN2ID[SOS]
EOS_ID = TOKEN2ID[EOS]
UNK_ID = TOKEN2ID[UNK]
SEP_ID = TOKEN2ID[SEP]


def tokenize_condition(line: str) -> List[str]:
    """
    Extract condition tokens from a line.
    Input:  '[MON] [DEC] [False] [196] 3-12-1962'
    Output: ['[MON]', '[DEC]', '[False]', '[196]']
    """
    tokens = re.findall(r'\[[^\]]+\]', line)
    return tokens[:4]  # exactly 4 condition tokens


def tokenize_date(date_str: str) -> List[str]:
    """
    Convert a date string 'dd-mm-yyyy' into character tokens.
    Output includes '-' separators: ['0','3','-','1','2','-','1','9','6','2']
    Zero-pad day and month to 2 digits.
    """
    parts = date_str.strip().split("-")
    day, month, year = parts[0].zfill(2), parts[1].zfill(2), parts[2].zfill(4)
    return list(day) + ["-"] + list(month) + ["-"] + list(year)


def encode_condition(tokens: List[str]) -> List[int]:
    return [TOKEN2ID.get(t, UNK_ID) for t in tokens]


def encode_date(tokens: List[str]) -> List[int]:
    return [TOKEN2ID.get(t, UNK_ID) for t in tokens]


def decode_date_ids(ids: List[int]) -> str:
    """Convert a list of token ids back to a date string."""
    chars = []
    for i in ids:
        tok = ID2TOKEN.get(i, UNK)
        if tok in (SOS, EOS, PAD, UNK, SEP):
            continue
        chars.append(tok)
    return "".join(chars)


def parse_line(line: str) -> Tuple[List[str], Optional[str]]:
    """
    Parse a full data line into (condition_tokens, date_str).
    If date is absent (input-only file), date_str is None.
    """
    line = line.strip()
    cond_tokens = tokenize_condition(line)

    # Try to find date at end
    m = re.search(r'(\d{1,2}-\d{1,2}-\d{4})\s*$', line)
    date_str = m.group(1) if m else None
    return cond_tokens, date_str


def build_sequence(cond_tokens: List[str], date_str: Optional[str] = None,
                   add_sos: bool = True, add_eos: bool = True) -> List[int]:
    """
    Build a full integer sequence:
      [SOS] cond_ids [SEP] date_ids [EOS]
    """
    seq: List[int] = []
    if add_sos:
        seq.append(SOS_ID)
    seq.extend(encode_condition(cond_tokens))
    seq.append(SEP_ID)
    if date_str:
        date_toks = tokenize_date(date_str)
        seq.extend(encode_date(date_toks))
    if add_eos:
        seq.append(EOS_ID)
    return seq


def format_output_line(cond_tokens: List[str], date_str: str) -> str:
    """Reconstruct output line matching data.txt format."""
    return " ".join(cond_tokens) + " " + date_str
