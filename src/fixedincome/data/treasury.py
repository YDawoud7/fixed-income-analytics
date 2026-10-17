"""Download the US Treasury daily par yield curve.

Source: U.S. Department of the Treasury, "Daily Treasury Par Yield Curve
Rates" (https://home.treasury.gov). Yields are CMT par yields in percent on
a semiannual bond-equivalent basis.
"""

from __future__ import annotations

import io
import re
import urllib.request
from pathlib import Path

import pandas as pd

URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
    "&field_tdr_date_value={year}&page&_format=csv"
)

_TENOR_RE = re.compile(r"^\s*([\d.]+)\s*(mo|month|yr|year)s?\s*$", re.IGNORECASE)


def tenor_in_years(label: str) -> float | None:
    """Convert a column label such as '3 Mo', '1.5 Month' or '10 Yr' to years."""
    m = _TENOR_RE.match(label)
    if not m:
        return None
    value, unit = float(m.group(1)), m.group(2).lower()
    return value / 12 if unit.startswith("mo") else value


def parse_par_curve_csv(text: str) -> pd.DataFrame:
    """Parse Treasury CSV text into a frame indexed by date, columns = tenor in years, values in decimals."""
    raw = pd.read_csv(io.StringIO(text))
    raw["Date"] = pd.to_datetime(raw["Date"], format="%m/%d/%Y")
    raw = raw.set_index("Date").sort_index()
    cols = {}
    for c in raw.columns:
        t = tenor_in_years(str(c))
        if t is not None:
            cols[c] = t
    out = raw[list(cols)].rename(columns=cols).astype(float) / 100.0
    return out.reindex(sorted(out.columns), axis=1)


def fetch_par_curve(year: int, cache_dir: str | Path | None = None, timeout: float = 30) -> pd.DataFrame:
    """Download one calendar year of daily par yields (optionally cached as CSV)."""
    cache = Path(cache_dir) / f"treasury_par_{year}.csv" if cache_dir else None
    if cache is not None and cache.exists():
        return parse_par_curve_csv(cache.read_text())
    req = urllib.request.Request(URL.format(year=year), headers={"User-Agent": "fixedincome/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text)
    return parse_par_curve_csv(text)
