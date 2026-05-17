"""
Date validation and generation utilities.
All dates are constrained to 1-1-1800 → 31-12-2200.
"""

import datetime
from typing import Optional, Tuple

MIN_YEAR = 1800
MAX_YEAR = 2200

DAY_MAP = {
    "MON": 0, "TUE": 1, "WED": 2, "THU": 3,
    "FRI": 4, "SAT": 5, "SUN": 6,
}
MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def is_leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def parse_date_string(s: str) -> Optional[datetime.date]:
    """Parse 'dd-mm-yyyy' → date object, or None if invalid."""
    try:
        parts = s.strip().split("-")
        if len(parts) != 3:
            return None
        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
        if not (MIN_YEAR <= year <= MAX_YEAR):
            return None
        return datetime.date(year, month, day)
    except (ValueError, OverflowError):
        return None


def validate_date(date_str: str, cond_tokens: list) -> bool:
    """
    Check that a generated date string satisfies all conditions.
    cond_tokens = ['[MON]', '[DEC]', '[False]', '[196]']
    """
    d = parse_date_string(date_str)
    if d is None:
        return False

    day_tok   = cond_tokens[0].strip("[]")   # e.g. 'MON'
    month_tok = cond_tokens[1].strip("[]")   # e.g. 'DEC'
    leap_tok  = cond_tokens[2].strip("[]")   # 'True' or 'False'
    dec_tok   = cond_tokens[3].strip("[]")   # e.g. '196'

    # Day of week
    if d.weekday() != DAY_MAP[day_tok]:
        return False

    # Month
    if d.month != MONTH_MAP[month_tok]:
        return False

    # Leap year
    expected_leap = leap_tok == "True"
    if is_leap(d.year) != expected_leap:
        return False

    # Decade: dec_tok '196' → years 1960-1969
    decade_start = int(dec_tok) * 10
    if not (decade_start <= d.year < decade_start + 10):
        return False

    return True


def extract_conditions(cond_tokens: list) -> Tuple[str, int, bool, int]:
    """Return (day_name, month_num, is_leap, decade_start)."""
    day_tok   = cond_tokens[0].strip("[]")
    month_tok = cond_tokens[1].strip("[]")
    leap_tok  = cond_tokens[2].strip("[]")
    dec_tok   = cond_tokens[3].strip("[]")
    return (
        day_tok,
        MONTH_MAP[month_tok],
        leap_tok == "True",
        int(dec_tok) * 10,
    )


def find_valid_date(day_name: str, month_num: int,
                    want_leap: bool, decade_start: int) -> Optional[str]:
    """
    Brute-force find a valid date matching all conditions.
    Used as fallback when model output is invalid.
    """
    decade_end = decade_start + 10
    for year in range(decade_start, min(decade_end, MAX_YEAR + 1)):
        if is_leap(year) != want_leap:
            continue
        try:
            d = datetime.date(year, month_num, 1)
        except ValueError:
            continue
        # Walk days in month
        while d.month == month_num:
            if d.weekday() == DAY_MAP[day_name]:
                return f"{d.day}-{d.month}-{d.year}"
            d += datetime.timedelta(days=1)
    return None  # no valid date exists (impossible conditions)
