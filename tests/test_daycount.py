from datetime import date

import pytest

from fixedincome.daycount import (
    ACT_360,
    ACT_365F,
    ACT_ACT_ICMA,
    ACT_ACT_ISDA,
    THIRTY_360,
    year_fraction,
)


def test_act_360_and_act_365f():
    d1, d2 = date(2024, 1, 15), date(2024, 7, 15)  # 182 days
    assert year_fraction(d1, d2, ACT_360) == pytest.approx(182 / 360)
    assert year_fraction(d1, d2, ACT_365F) == pytest.approx(182 / 365)


@pytest.mark.parametrize(
    "start,end,days",
    [
        (date(2024, 1, 31), date(2024, 3, 31), 60),  # both 31sts roll to 30
        (date(2024, 1, 30), date(2024, 3, 31), 60),
        (date(2024, 1, 15), date(2024, 3, 31), 76),  # D2 kept at 31 when D1 < 30
        (date(2023, 2, 28), date(2023, 8, 28), 180),
    ],
)
def test_thirty_360(start, end, days):
    assert year_fraction(start, end, THIRTY_360) == pytest.approx(days / 360)


def test_act_act_isda_spans_leap_year():
    # 2023-11-01 -> 2024-03-01: 61 days in 2023 (365) and 60 days in 2024 (366)
    expected = 61 / 365 + 60 / 366
    assert year_fraction(date(2023, 11, 1), date(2024, 3, 1), ACT_ACT_ISDA) == pytest.approx(expected)


def test_act_act_isda_full_years():
    assert year_fraction(date(2020, 1, 1), date(2023, 1, 1), ACT_ACT_ISDA) == pytest.approx(3.0)


def test_act_act_icma_full_period_is_one_over_frequency():
    s, e = date(2024, 2, 15), date(2024, 8, 15)
    assert year_fraction(s, e, ACT_ACT_ICMA, s, e, 2) == pytest.approx(0.5)


def test_act_act_icma_requires_reference_period():
    with pytest.raises(ValueError):
        year_fraction(date(2024, 1, 1), date(2024, 2, 1), ACT_ACT_ICMA)


def test_reversed_dates_are_negative():
    assert year_fraction(date(2024, 7, 1), date(2024, 1, 1), ACT_365F) < 0
