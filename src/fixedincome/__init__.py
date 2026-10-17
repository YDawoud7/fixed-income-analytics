"""Fixed income analytics: bond pricing, risk and yield curves."""

from .bond import FixedRateBond
from .bootstrap import bootstrap_par_curve
from .curve import ZeroCurve
from .curve_risk import curve_duration, key_rate_durations, price_from_curve, z_spread
from .nss import fit_nelson_siegel, fit_svensson, nelson_siegel, svensson

__version__ = "0.1.0"

__all__ = [
    "FixedRateBond",
    "ZeroCurve",
    "bootstrap_par_curve",
    "price_from_curve",
    "z_spread",
    "key_rate_durations",
    "curve_duration",
    "nelson_siegel",
    "svensson",
    "fit_nelson_siegel",
    "fit_svensson",
]
