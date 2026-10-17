import numpy as np
import pytest

from fixedincome.bootstrap import bootstrap_par_curve
from fixedincome.curve import ZeroCurve

TENORS = [1 / 12, 2 / 12, 3 / 12, 4 / 12, 6 / 12, 1, 2, 3, 5, 7, 10, 20, 30]
# An upward sloping curve with an inverted front end, in decimals.
PARS = [0.0430, 0.0428, 0.0425, 0.0420, 0.0410, 0.0395, 0.0380, 0.0378, 0.0385, 0.0398, 0.0415, 0.0465, 0.0470]


def test_flat_curve_discounting():
    c = ZeroCurve([1, 5, 10], [0.03, 0.03, 0.03])
    assert c.discount(7.5) == pytest.approx(np.exp(-0.03 * 7.5))
    assert c.forward_rate(2, 9) == pytest.approx(0.03)


def test_log_linear_gives_flat_forwards_between_pillars():
    c = ZeroCurve([1, 2], [0.02, 0.04])
    f1 = c.forward_rate(1.1, 1.3)
    f2 = c.forward_rate(1.6, 1.9)
    assert f1 == pytest.approx(f2)
    assert f1 == pytest.approx((0.04 * 2 - 0.02 * 1) / 1)


def test_flat_extrapolation():
    c = ZeroCurve([1, 10], [0.02, 0.05], interpolation="linear")
    assert c.zero_rate(0.25) == pytest.approx(0.02)
    assert c.zero_rate(40) == pytest.approx(0.05)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        ZeroCurve([1, 1], [0.01, 0.02])
    with pytest.raises(ValueError):
        ZeroCurve([1, 2], [0.01, 0.02], interpolation="cubic")


def test_flat_par_curve_bootstraps_to_flat_zeros():
    curve = bootstrap_par_curve([0.5, 1, 2, 5, 10, 30], [0.05] * 6)
    expected = 2 * np.log(1.025)
    assert np.allclose(curve.zero_rate([0.5, 1, 3, 7, 10, 25]), expected)


@pytest.mark.parametrize("tenor", [1, 2, 3, 5, 7, 10, 20, 30])
def test_bootstrap_reprices_input_par_yields(tenor):
    curve = bootstrap_par_curve(TENORS, PARS)
    expected = PARS[TENORS.index(tenor)]
    assert curve.par_yield(tenor) == pytest.approx(expected, abs=1e-12)


def test_bill_section_uses_simple_interest():
    curve = bootstrap_par_curve(TENORS, PARS)
    t, y = 3 / 12, 0.0425
    assert curve.discount(t) == pytest.approx(1 / (1 + y * t))


def test_upward_par_curve_has_zeros_above_pars_at_long_end():
    curve = bootstrap_par_curve(TENORS, PARS)
    z30 = 2 * (np.exp(curve.zero_rate(30) / 2) - 1)  # convert to semiannual
    assert z30 > PARS[-1]


def test_parallel_shift():
    c = ZeroCurve([1, 5], [0.03, 0.04])
    assert c.shifted(25).zero_rate(5) == pytest.approx(0.0425)
