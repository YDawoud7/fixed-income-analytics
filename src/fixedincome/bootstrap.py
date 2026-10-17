"""Bootstrap a zero curve from par yields.

Designed for the US Treasury par yield curve (CMT rates), where every tenor
is quoted as a semiannual bond-equivalent par yield. The method:

1. Tenors under six months are bills with no coupons before maturity, so the
   quoted yield converts directly to a discount factor using simple interest:
   DF(t) = 1 / (1 + y t).
2. Par yields are linearly interpolated onto a semiannual grid
   0.5, 1.0, ..., T_max.
3. Each grid point is solved in turn from the par bond condition

       1 = (c/2) * sum_{i<n} DF_i + (1 + c/2) * DF_n

   so the bootstrapped curve reprices every grid par bond exactly.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .curve import ZeroCurve


def bootstrap_par_curve(
    tenors: Sequence[float],
    par_yields: Sequence[float],
    frequency: int = 2,
    interpolation: str = "log_linear",
) -> ZeroCurve:
    """Build a ``ZeroCurve`` from par yields (decimals) at ``tenors`` (years)."""
    tenors = np.asarray(tenors, dtype=float)
    par_yields = np.asarray(par_yields, dtype=float)
    order = np.argsort(tenors)
    tenors, par_yields = tenors[order], par_yields[order]
    if len(tenors) < 2:
        raise ValueError("need at least two tenors")
    mask = ~np.isnan(par_yields)
    tenors, par_yields = tenors[mask], par_yields[mask]

    step = 1.0 / frequency
    times: list[float] = []
    dfs: list[float] = []

    # Money market section: tenors shorter than one coupon period.
    for t, y in zip(tenors, par_yields):
        if t < step - 1e-9:
            times.append(float(t))
            dfs.append(1.0 / (1.0 + y * t))

    t_max = tenors[-1]
    n_max = int(np.floor(t_max * frequency + 1e-9))
    grid = np.arange(1, n_max + 1) * step
    grid_par = np.interp(grid, tenors, par_yields)

    coupon_dfs: list[float] = []
    for t, c in zip(grid, grid_par):
        annuity = sum(coupon_dfs)
        df = (1.0 - c / frequency * annuity) / (1.0 + c / frequency)
        if df <= 0:
            raise ValueError(f"bootstrap failed at t={t:.2f}: non-positive discount factor")
        coupon_dfs.append(df)
        times.append(float(t))
        dfs.append(df)

    return ZeroCurve.from_discount_factors(times, dfs, interpolation)
