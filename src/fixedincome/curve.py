"""Zero coupon yield curves.

A ``ZeroCurve`` stores continuously compounded zero rates at pillar times
(in years) and interpolates between them. Two schemes are supported:

* ``"log_linear"``: linear in log discount factors, which gives piecewise
  flat instantaneous forwards. This is the standard choice for a
  bootstrapped curve because it never creates spurious forward kinks.
* ``"linear"``: linear in zero rates.

Zero rates are held flat beyond the first and last pillars.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

ArrayLike = float | Sequence[float] | np.ndarray


class ZeroCurve:
    def __init__(self, times: Sequence[float], zero_rates: Sequence[float], interpolation: str = "log_linear"):
        t = np.asarray(times, dtype=float)
        z = np.asarray(zero_rates, dtype=float)
        if t.ndim != 1 or t.shape != z.shape or len(t) == 0:
            raise ValueError("times and zero_rates must be equal-length 1-D sequences")
        if np.any(t <= 0) or np.any(np.diff(t) <= 0):
            raise ValueError("times must be positive and strictly increasing")
        if interpolation not in ("log_linear", "linear"):
            raise ValueError("interpolation must be 'log_linear' or 'linear'")
        self.times = t
        self.zero_rates = z
        self.interpolation = interpolation

    @classmethod
    def from_discount_factors(cls, times, dfs, interpolation: str = "log_linear") -> "ZeroCurve":
        t = np.asarray(times, dtype=float)
        return cls(t, -np.log(np.asarray(dfs, dtype=float)) / t, interpolation)

    def zero_rate(self, t: ArrayLike) -> np.ndarray | float:
        """Continuously compounded zero rate at time ``t`` (years)."""
        tt = np.asarray(t, dtype=float)
        if self.interpolation == "linear":
            z = np.interp(tt, self.times, self.zero_rates)
        else:
            # Interpolate -ln(DF) = z * t linearly; flat zero rates outside.
            log_df = np.interp(tt, self.times, self.zero_rates * self.times)
            z = np.where(tt > 0, log_df / np.where(tt > 0, tt, 1.0), self.zero_rates[0])
            z = np.where(tt < self.times[0], self.zero_rates[0], z)
            z = np.where(tt > self.times[-1], self.zero_rates[-1], z)
        return float(z) if np.ndim(z) == 0 else z

    def discount(self, t: ArrayLike) -> np.ndarray | float:
        tt = np.asarray(t, dtype=float)
        df = np.exp(-np.asarray(self.zero_rate(tt)) * tt)
        return float(df) if np.ndim(df) == 0 else df

    def forward_rate(self, t1: ArrayLike, t2: ArrayLike) -> np.ndarray | float:
        """Continuously compounded forward rate between ``t1`` and ``t2``."""
        a, b = np.asarray(t1, dtype=float), np.asarray(t2, dtype=float)
        f = np.log(np.asarray(self.discount(a)) / np.asarray(self.discount(b))) / (b - a)
        return float(f) if np.ndim(f) == 0 else f

    def par_yield(self, maturity: float, frequency: int = 2) -> float:
        """Coupon rate that prices a bullet bond at par off this curve."""
        n = int(round(maturity * frequency))
        if n < 1 or abs(n / frequency - maturity) > 1e-9:
            raise ValueError("maturity must be a whole number of coupon periods")
        t = np.arange(1, n + 1) / frequency
        dfs = np.asarray(self.discount(t))
        return float(frequency * (1 - dfs[-1]) / dfs.sum())

    def shifted(self, bp: float) -> "ZeroCurve":
        """Parallel shift of the zero curve by ``bp`` basis points."""
        return ZeroCurve(self.times, self.zero_rates + bp * 1e-4, self.interpolation)

    def __repr__(self) -> str:
        return f"ZeroCurve({len(self.times)} pillars, {self.times[0]:.3g}y-{self.times[-1]:.3g}y, {self.interpolation})"
