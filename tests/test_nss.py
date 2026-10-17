import numpy as np
import pytest

from fixedincome.nss import fit_nelson_siegel, fit_svensson, nelson_siegel, svensson

T = np.array([1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])


def test_limits():
    # y(0) = b0 + b1 and y(inf) = b0
    assert nelson_siegel(1e-10, 0.04, -0.01, 0.02, 2.0) == pytest.approx(0.03, abs=1e-8)
    assert nelson_siegel(1e6, 0.04, -0.01, 0.02, 2.0) == pytest.approx(0.04, abs=1e-5)


def test_recovers_nelson_siegel_parameters():
    true = (0.045, -0.012, 0.018, 1.8)
    fit = fit_nelson_siegel(T, nelson_siegel(T, *true))
    assert fit.rmse < 1e-9
    assert (fit.b0, fit.b1, fit.b2, fit.tau) == pytest.approx(true, rel=1e-4)


def test_svensson_fits_double_hump_exactly():
    true = (0.042, -0.008, -0.02, 0.03, 1.0, 8.0)
    y = svensson(T, *true)
    fit = fit_svensson(T, y)
    assert fit.rmse < 1e-7
    assert np.allclose(fit(T), y, atol=1e-7)


def test_svensson_never_worse_than_nelson_siegel():
    rng = np.random.default_rng(0)
    y = svensson(T, 0.042, -0.008, -0.02, 0.03, 1.0, 8.0) + rng.normal(0, 2e-4, len(T))
    assert fit_svensson(T, y).rmse <= fit_nelson_siegel(T, y).rmse + 1e-12


def test_too_few_points():
    with pytest.raises(ValueError):
        fit_svensson(T[:4], np.full(4, 0.03))
