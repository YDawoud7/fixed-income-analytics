from datetime import date

import pytest

from fixedincome.schedule import add_months, coupon_dates, previous_coupon


def test_add_months_clips_to_month_length():
    assert add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)
    assert add_months(date(2023, 1, 31), 1) == date(2023, 2, 28)
    assert add_months(date(2024, 8, 15), -6) == date(2024, 2, 15)


def test_add_months_end_of_month_rule():
    assert add_months(date(2024, 2, 29), 6, eom=True) == date(2024, 8, 31)


def test_semiannual_schedule():
    dates = coupon_dates(date(2026, 11, 15), 2, after=date(2024, 10, 1))
    assert dates == [
        date(2024, 11, 15),
        date(2025, 5, 15),
        date(2025, 11, 15),
        date(2026, 5, 15),
        date(2026, 11, 15),
    ]


def test_schedule_excludes_settlement_on_coupon_date():
    dates = coupon_dates(date(2026, 5, 15), 2, after=date(2025, 11, 15))
    assert dates == [date(2026, 5, 15)]


def test_month_end_maturity_rolls_all_dates():
    dates = coupon_dates(date(2027, 8, 31), 2, after=date(2026, 1, 1))
    assert dates == [date(2026, 2, 28), date(2026, 8, 31), date(2027, 2, 28), date(2027, 8, 31)]


def test_previous_coupon():
    assert previous_coupon(date(2030, 5, 15), 2, date(2025, 3, 3)) == date(2024, 11, 15)
    assert previous_coupon(date(2030, 5, 15), 2, date(2025, 5, 15)) == date(2025, 5, 15)


def test_invalid_frequency():
    with pytest.raises(ValueError):
        coupon_dates(date(2030, 1, 1), 3, date(2025, 1, 1))
