from datetime import date

import numpy as np
import pytest

from fixedincome.bond import FixedRateBond
from fixedincome.curve import ZeroCurve
from fixedincome.curve_risk import (
    curve_duration,
    key_rate_durations,
    price_from_curve,
    z_spread,
)

SETTLE = date(2025, 10, 17)
CURVE = ZeroCurve([0.25, 1, 2, 5, 10, 30], [0.042, 0.039, 0.037, 0.038, 0.041, 0.046])


def test_flat_curve_price_matches_continuous_yield():
    bond = FixedRateBond(coupon=0.04, maturity=date(2030, 10, 17))
    flat = ZeroCurve([1, 30], [0.04, 0.04])
    t = np.arange(1, 11) / 2  # whole periods since settlement sits on a coupon date
    expected = np.sum(2.0 * np.exp(-0.04 * t)) + 100 * np.exp(-0.04 * 5)
    # ACT/365F times differ slightly from exact half years, so allow a small tolerance.
    assert price_from_curve(bond, flat, SETTLE) == pytest.approx(expected, rel=5e-4)


def test_z_spread_round_trip():
    bond = FixedRateBond(coupon=0.045, maturity=date(2034, 11, 15))
    spread = 0.0035
    dirty = price_from_curve(bond, CURVE, SETTLE, spread)
    clean = dirty - bond.accrued_interest(SETTLE)
    assert z_spread(bond, CURVE, SETTLE, clean) == pytest.approx(spread, abs=1e-12)


def test_key_rate_durations_sum_to_curve_duration():
    bond = FixedRateBond(coupon=0.0425, maturity=date(2045, 5, 15))
    krd = key_rate_durations(bond, CURVE, SETTLE)
    assert sum(krd.values()) == pytest.approx(curve_duration(bond, CURVE, SETTLE), rel=1e-6)


def test_key_rate_exposure_concentrates_near_maturity():
    bond = FixedRateBond(coupon=0.0, maturity=date(2030, 10, 17))  # ~5y zero
    krd = key_rate_durations(bond, CURVE, SETTLE)
    assert max(krd, key=krd.get) == 5
    assert krd[10] == pytest.approx(0.0)
