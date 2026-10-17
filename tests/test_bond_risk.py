from datetime import date

import pytest

from fixedincome.bond import FixedRateBond

SETTLE = date(2025, 10, 17)


def test_textbook_duration_of_par_bond():
    # 5-year 9% semiannual bond at a 9% yield: Macaulay 4.13y, modified 3.96y.
    bond = FixedRateBond(coupon=0.09, maturity=date(2030, 1, 15))
    settle = date(2025, 1, 15)
    assert bond.macaulay_duration(0.09, settle) == pytest.approx(4.13, abs=5e-3)
    assert bond.modified_duration(0.09, settle) == pytest.approx(3.96, abs=5e-3)


def test_zero_coupon_duration_equals_maturity():
    bond = FixedRateBond(coupon=0.0, maturity=date(2035, 1, 15))
    assert bond.macaulay_duration(0.04, date(2025, 1, 15)) == pytest.approx(10.0)


@pytest.mark.parametrize("coupon,yld", [(0.02, 0.045), (0.0475, 0.0475), (0.07, 0.03)])
def test_analytic_matches_finite_difference(coupon, yld):
    bond = FixedRateBond(coupon=coupon, maturity=date(2044, 2, 15))
    assert bond.modified_duration(yld, SETTLE) == pytest.approx(bond.effective_duration(yld, SETTLE), rel=1e-6)
    assert bond.convexity(yld, SETTLE) == pytest.approx(bond.effective_convexity(yld, SETTLE), rel=1e-4)


def test_dv01_predicts_one_basis_point_move():
    bond = FixedRateBond(coupon=0.04, maturity=date(2035, 8, 15))
    y = 0.042
    actual = bond.dirty_price(y - 1e-4, SETTLE) - bond.dirty_price(y, SETTLE)
    assert bond.dv01(y, SETTLE) == pytest.approx(actual, rel=1e-3)


def test_duration_convexity_approximation():
    bond = FixedRateBond(coupon=0.03, maturity=date(2054, 5, 15))
    y, dy = 0.045, 0.01
    p0 = bond.dirty_price(y, SETTLE)
    approx = p0 * (1 - bond.modified_duration(y, SETTLE) * dy + 0.5 * bond.convexity(y, SETTLE) * dy**2)
    assert approx == pytest.approx(bond.dirty_price(y + dy, SETTLE), rel=2e-3)
