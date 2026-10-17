"""Day count conventions.

Each convention maps a pair of dates to a year fraction. ACT/ACT ICMA also
needs the reference coupon period and the coupon frequency, because it
measures time as a fraction of that period.
"""

from __future__ import annotations

from datetime import date

ACT_360 = "ACT/360"
ACT_365F = "ACT/365F"
THIRTY_360 = "30/360"
ACT_ACT_ISDA = "ACT/ACT-ISDA"
ACT_ACT_ICMA = "ACT/ACT-ICMA"

CONVENTIONS = (ACT_360, ACT_365F, THIRTY_360, ACT_ACT_ISDA, ACT_ACT_ICMA)


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _thirty_360_days(start: date, end: date) -> int:
    """Day count under the 30/360 US bond basis."""
    d1, d2 = start.day, end.day
    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 >= 30:
        d2 = 30
    return 360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)


def _act_act_isda(start: date, end: date) -> float:
    if start.year == end.year:
        return (end - start).days / (366 if _is_leap(start.year) else 365)
    frac = (date(start.year + 1, 1, 1) - start).days / (366 if _is_leap(start.year) else 365)
    frac += end.year - start.year - 1
    frac += (end - date(end.year, 1, 1)).days / (366 if _is_leap(end.year) else 365)
    return frac


def year_fraction(
    start: date,
    end: date,
    convention: str = ACT_365F,
    ref_start: date | None = None,
    ref_end: date | None = None,
    frequency: int | None = None,
) -> float:
    """Year fraction between ``start`` and ``end`` under ``convention``.

    For ACT/ACT-ICMA, ``ref_start``/``ref_end`` must bound the coupon period
    that contains the accrual and ``frequency`` is coupons per year.
    """
    if end < start:
        return -year_fraction(end, start, convention, ref_start, ref_end, frequency)

    if convention == ACT_360:
        return (end - start).days / 360.0
    if convention == ACT_365F:
        return (end - start).days / 365.0
    if convention == THIRTY_360:
        return _thirty_360_days(start, end) / 360.0
    if convention == ACT_ACT_ISDA:
        return _act_act_isda(start, end)
    if convention == ACT_ACT_ICMA:
        if ref_start is None or ref_end is None or frequency is None:
            raise ValueError("ACT/ACT-ICMA needs ref_start, ref_end and frequency")
        return (end - start).days / (frequency * (ref_end - ref_start).days)
    raise ValueError(f"Unknown day count convention: {convention!r}")
