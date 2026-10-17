"""Pricing and risk of bonds off a zero curve.

* ``price_from_curve``: dirty price from discounting each cash flow at the
  curve's zero rate plus an optional constant spread.
* ``z_spread``: the constant spread over the zero curve that reproduces a
  market price.
* ``key_rate_durations``: sensitivity to localised bumps of the zero curve.
  Each key rate gets a triangular bump that peaks at its tenor and fades to
  zero at the neighbouring key tenors, so the bumps sum to a parallel shift
  and the key rate durations sum to the curve duration.

Cash flow times are measured as ACT/365F year fractions from settlement.
"""

from __future__ import annotations

from datetime import date
from typing import Sequence

import numpy as np
from scipy.optimize import brentq

from .bond import FixedRateBond
from .curve import ZeroCurve
from .daycount import ACT_365F, year_fraction

DEFAULT_KEY_TENORS = (0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30)


def _times_amounts(bond: FixedRateBond, settlement: date) -> tuple[np.ndarray, np.ndarray]:
    flows = bond.cashflows(settlement)
    t = np.array([year_fraction(settlement, d, ACT_365F) for d, _ in flows])
    a = np.array([amt for _, amt in flows])
    return t, a


def price_from_curve(bond: FixedRateBond, curve: ZeroCurve, settlement: date, spread: float = 0.0) -> float:
    """Dirty price per 100 face. ``spread`` is continuously compounded, decimal."""
    t, a = _times_amounts(bond, settlement)
    z = np.asarray(curve.zero_rate(t))
    return float(np.sum(a * np.exp(-(z + spread) * t)))


def z_spread(bond: FixedRateBond, curve: ZeroCurve, settlement: date, clean_price: float) -> float:
    target = clean_price + bond.accrued_interest(settlement)
    return float(brentq(lambda s: price_from_curve(bond, curve, settlement, s) - target, -0.5, 0.5, xtol=1e-14))


def _triangle(t: np.ndarray, keys: np.ndarray, i: int) -> np.ndarray:
    """Weight of key tenor ``i`` at times ``t``; flat beyond the end keys."""
    w = np.zeros_like(t)
    k = keys[i]
    if i > 0:
        left = keys[i - 1]
        mask = (t > left) & (t <= k)
        w[mask] = (t[mask] - left) / (k - left)
    else:
        w[t <= k] = 1.0
    if i < len(keys) - 1:
        right = keys[i + 1]
        mask = (t > k) & (t < right)
        w[mask] = (right - t[mask]) / (right - k)
    else:
        w[t > k] = 1.0
    return w


def key_rate_durations(
    bond: FixedRateBond,
    curve: ZeroCurve,
    settlement: date,
    key_tenors: Sequence[float] = DEFAULT_KEY_TENORS,
    bump_bp: float = 1.0,
) -> dict[float, float]:
    """Key rate durations (years) from central differences on zero rate bumps."""
    keys = np.asarray(sorted(key_tenors), dtype=float)
    t, a = _times_amounts(bond, settlement)
    z = np.asarray(curve.zero_rate(t))
    p0 = float(np.sum(a * np.exp(-z * t)))
    h = bump_bp * 1e-4
    out = {}
    for i, k in enumerate(keys):
        dz = h * _triangle(t, keys, i)
        up = np.sum(a * np.exp(-(z + dz) * t))
        down = np.sum(a * np.exp(-(z - dz) * t))
        out[float(k)] = float((down - up) / (2 * h * p0))
    return out


def curve_duration(bond: FixedRateBond, curve: ZeroCurve, settlement: date, bump_bp: float = 1.0) -> float:
    """Effective duration under a parallel shift of the zero curve."""
    h = bump_bp * 1e-4
    p0 = price_from_curve(bond, curve, settlement)
    up = price_from_curve(bond, curve, settlement, h)
    down = price_from_curve(bond, curve, settlement, -h)
    return (down - up) / (2 * h * p0)
