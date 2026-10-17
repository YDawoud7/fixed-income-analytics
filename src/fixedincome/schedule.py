"""Coupon schedule generation.

Coupon dates are generated backwards from maturity in steps of 12/frequency
months, the way US Treasury and most bullet bond schedules are built. If the
maturity falls on a month end, every coupon date is rolled to month end.
Dates are left unadjusted for holidays, which matches how Treasury accrual
is computed.
"""

from __future__ import annotations

import calendar
from datetime import date


def _month_end(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def is_month_end(d: date) -> bool:
    return d.day == _month_end(d.year, d.month)


def add_months(d: date, months: int, eom: bool = False) -> date:
    """Shift ``d`` by a whole number of months, clipping to the month length.

    With ``eom=True`` the result is always the last day of its month.
    """
    idx = d.year * 12 + (d.month - 1) + months
    year, month = divmod(idx, 12)
    month += 1
    last = _month_end(year, month)
    day = last if eom else min(d.day, last)
    return date(year, month, day)


def coupon_dates(maturity: date, frequency: int, after: date) -> list[date]:
    """All coupon dates strictly after ``after`` up to and including maturity."""
    if frequency not in (1, 2, 4, 12):
        raise ValueError("frequency must be 1, 2, 4 or 12")
    step = 12 // frequency
    eom = is_month_end(maturity)
    dates = []
    k = 0
    while True:
        d = add_months(maturity, -step * k, eom)
        if d <= after:
            break
        dates.append(d)
        k += 1
    return dates[::-1]


def previous_coupon(maturity: date, frequency: int, settlement: date) -> date:
    """Last coupon date on or before ``settlement`` (may precede issue)."""
    step = 12 // frequency
    eom = is_month_end(maturity)
    k = 0
    while True:
        d = add_months(maturity, -step * k, eom)
        if d <= settlement:
            return d
        k += 1
