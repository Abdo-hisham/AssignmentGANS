"""
generate_data.py – Generates the data.txt file for the Dates Generator problem.
Produces dates in range 1-1-1800 to 31-12-2200.
Run: python data/generate_data.py
"""

import datetime
import random
from pathlib import Path

DAYS   = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

MIN_YEAR = 1800
MAX_YEAR = 2200
N        = 200_000
SEED     = 42

random.seed(SEED)


def is_leap(y: int) -> bool:
    return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)


def random_date() -> datetime.date:
    while True:
        year  = random.randint(MIN_YEAR, MAX_YEAR)
        month = random.randint(1, 12)
        try:
            # pick random day in month
            import calendar
            _, last = calendar.monthrange(year, month)
            day = random.randint(1, last)
            return datetime.date(year, month, day)
        except Exception:
            continue


def date_to_line(d: datetime.date) -> str:
    day_tok   = f"[{DAYS[d.weekday()]}]"
    month_tok = f"[{MONTHS[d.month - 1]}]"
    leap_tok  = f"[{is_leap(d.year)}]"
    dec_tok   = f"[{d.year // 10}]"
    date_str  = f"{d.day}-{d.month}-{d.year}"
    return f"{day_tok} {month_tok} {leap_tok} {dec_tok} {date_str}"


out_path = Path(__file__).parent / "data.txt"
example_path = Path(__file__).parent / "example_input.txt"

print(f"Generating {N} date samples …")
lines = []
for _ in range(N):
    lines.append(date_to_line(random_date()))

out_path.write_text("\n".join(lines) + "\n")
print(f"✓ Saved to {out_path}")

# Write example_input.txt (conditions only, first 100 lines)
example_lines = []
for line in lines[:100]:
    parts = line.rsplit(" ", 1)   # split off the date
    example_lines.append(parts[0])
example_path.write_text("\n".join(example_lines) + "\n")
print(f"✓ Example input saved to {example_path}")
