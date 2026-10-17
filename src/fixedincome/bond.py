"""Fixed-rate bullet bonds priced on a yield basis.

Pricing follows the street convention used for US Treasuries: cash flows
are discounted at the periodic yield y/f, with the first period shortened by
the fraction of the current coupon period remaining until the next coupon.

    dirty = sum_k CF_k / (1 + y/f) ** (w + k),  k = 0..n-1

where w = days(settlement, next coupon) / days(previous, next coupon).
Prices are quoted per 100 of face value.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy.optimize import brentq

from .daycount import ACT_ACT_ICMA, year_fraction
from .schedule import coupon_dates, previous_coupon


@dataclass(frozen=True)
class FixedRateBond:
    coupon: float  # annual coupon rate as a decimal, e.g. 0.0425
    maturity: date
    frequency: int = 2
    face: float = 100.0
    day_count: str = ACT_ACT_ICMA

    @property
    def coupon_amount(self) -> float:
        return self.face * self.coupon / self.frequency

    def cashflows(self, settlement: date) -> list[tuple[date, float]]:
        """Remaining (date, amount) pairs strictly after settlement."""
        if settlement >= self.maturity:
            return []
        dates = coupon_dates(self.maturity, self.frequency, settlement)
        flows = [(d, self.coupon_amount) for d in dates]
        last_date, last_amt = flows[-1]
        flows[-1] = (last_date, last_amt + self.face)
        return flows

    def _period(self, settlement: date) -> tuple[date, date]:
        prev = previous_coupon(self.maturity, self.frequency, settlement)
        nxt = coupon_dates(self.maturity, self.frequency, settlement)[0]
        return prev, nxt

    def accrued_interest(self, settlement: date) -> float:
        if settlement >= self.maturity:
            return 0.0
        prev, nxt = self._period(settlement)
        frac = year_fraction(prev, settlement, self.day_count, prev, nxt, self.frequency)
        return self.face * self.coupon * frac

    def _times_and_flows(self, settlement: date) -> tuple[np.ndarray, np.ndarray]:
        """Discounting exponents (in coupon periods) and cash flow amounts."""
        flows = self.cashflows(settlement)
        prev, nxt = self._period(settlement)
        w = (nxt - settlement).days / (nxt - prev).days
        periods = w + np.arange(len(flows))
        amounts = np.array([a for _, a in flows])
        return periods, amounts

    def dirty_price(self, yld: float, settlement: date) -> float:
        periods, amounts = self._times_and_flows(settlement)
        disc = (1 + yld / self.frequency) ** -periods
        return float(np.sum(amounts * disc))

    def clean_price(self, yld: float, settlement: date) -> float:
        return self.dirty_price(yld, settlement) - self.accrued_interest(settlement)

    def yield_to_maturity(self, clean_price: float, settlement: date) -> float:
        """Solve for the yield that reproduces a quoted clean price."""
        target = clean_price + self.accrued_interest(settlement)

        def f(y: float) -> float:
            return self.dirty_price(y, settlement) - target

        return float(brentq(f, -0.99 * self.frequency, 10.0, xtol=1e-14, maxiter=500))
