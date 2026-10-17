from datetime import date

import pytest

from fixedincome.bond import FixedRateBond


def test_textbook_price_on_coupon_date():
    # Fabozzi: 20-year 10% semiannual bond at an 11% yield prices at $919.77 per $1,000 face.
    settle = date(2025, 1, 15)
    bond = FixedRateBond(coupon=0.10, maturity=date(2045, 1, 15))
    assert bond.clean_price(0.11, settle) == pytest.approx(91.977, abs=5e-4)


def test_par_bond_prices_at_par_on_coupon_date():
    settle = date(2025, 2, 15)
    bond = FixedRateBond(coupon=0.045, maturity=date(2035, 2, 15))
    assert bond.clean_price(0.045, settle) == pytest.approx(100.0, abs=1e-10)


def test_accrued_interest_act_act_icma():
    # 4.25% note, coupon period 2024-11-15 -> 2025-05-15 (181 days),
    # settlement on 2025-02-03 is 80 days into the period.
    bond = FixedRateBond(coupon=0.0425, maturity=date(2034, 11, 15))
    expected = 100 * 0.0425 / 2 * 80 / 181
    assert bond.accrued_interest(date(2025, 2, 3)) == pytest.approx(expected, rel=1e-12)


def test_yield_round_trip_between_coupons():
    bond = FixedRateBond(coupon=0.0375, maturity=date(2031, 8, 15))
    settle = date(2025, 10, 17)
    for y in (0.0, 0.021, 0.0437, 0.09):
        price = bond.clean_price(y, settle)
        assert bond.yield_to_maturity(price, settle) == pytest.approx(y, abs=1e-10)


def test_clean_price_continuous_across_coupon_date():
    # Dirty price drops by the coupon on the payment date; clean price should not jump.
    bond = FixedRateBond(coupon=0.05, maturity=date(2030, 6, 30))
    before = bond.clean_price(0.04, date(2025, 6, 29))
    after = bond.clean_price(0.04, date(2025, 7, 1))
    assert abs(before - after) < 0.01


def test_cashflows_include_principal():
    bond = FixedRateBond(coupon=0.04, maturity=date(2026, 5, 15))
    flows = bond.cashflows(date(2025, 6, 1))
    assert flows == [(date(2025, 11, 15), 2.0), (date(2026, 5, 15), 102.0)]
