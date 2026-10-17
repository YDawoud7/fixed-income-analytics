"""Nelson-Siegel and Nelson-Siegel-Svensson yield curve models.

    NS:  y(t) = b0 + b1 * L1(t/tau) + b2 * L2(t/tau)
    NSS: y(t) = NS(t) + b3 * L2(t/tau2)

with loadings L1(x) = (1 - e^-x) / x and L2(x) = L1(x) - e^-x.
b0 is the long-run level, b1 the slope (b0 + b1 is the instantaneous short
rate) and b2/b3 control the humps.

For fixed decay parameters the model is linear in the betas, so fitting
first scans a grid of taus with ordinary least squares and then refines all
parameters jointly with bounded nonlinear least squares. This avoids the
local minima that a cold-started optimiser tends to fall into.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import least_squares

TAU_BOUNDS = (0.05, 30.0)


def _l1(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x, 1e-12)
    return (1 - np.exp(-x)) / x


def _l2(x: np.ndarray) -> np.ndarray:
    return _l1(x) - np.exp(-np.maximum(x, 1e-12))


def nelson_siegel(t, b0: float, b1: float, b2: float, tau: float) -> np.ndarray:
    x = np.asarray(t, dtype=float) / tau
    return b0 + b1 * _l1(x) + b2 * _l2(x)


def svensson(t, b0: float, b1: float, b2: float, b3: float, tau1: float, tau2: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return nelson_siegel(t, b0, b1, b2, tau1) + b3 * _l2(t / tau2)


@dataclass(frozen=True)
class NSParams:
    b0: float
    b1: float
    b2: float
    tau: float
    rmse: float

    def __call__(self, t) -> np.ndarray:
        return nelson_siegel(t, self.b0, self.b1, self.b2, self.tau)


@dataclass(frozen=True)
class NSSParams:
    b0: float
    b1: float
    b2: float
    b3: float
    tau1: float
    tau2: float
    rmse: float

    def __call__(self, t) -> np.ndarray:
        return svensson(t, self.b0, self.b1, self.b2, self.b3, self.tau1, self.tau2)


def _design(t: np.ndarray, taus: tuple[float, ...]) -> np.ndarray:
    cols = [np.ones_like(t), _l1(t / taus[0]), _l2(t / taus[0])]
    for tau in taus[1:]:
        cols.append(_l2(t / tau))
    return np.column_stack(cols)


def _grid_search(t, y, w, n_taus: int):
    grid = np.geomspace(0.2, 15.0, 25)
    best = None
    for taus in product(grid, repeat=n_taus):
        if n_taus == 2 and taus[1] <= taus[0] * 1.2:
            continue  # keep humps distinct so the design stays well conditioned
        X = _design(t, taus)
        betas, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
        sse = float(np.sum((w * (X @ betas - y)) ** 2))
        if best is None or sse < best[0]:
            best = (sse, betas, taus)
    return best[1], best[2]


def _fit(t, y, weights, n_taus: int):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones_like(t) if weights is None else np.sqrt(np.asarray(weights, dtype=float))
    if len(t) < 3 + n_taus:
        raise ValueError("not enough points to fit the model")

    betas, taus = _grid_search(t, y, w, n_taus)
    x0 = np.concatenate([betas, taus])
    n_b = len(betas)
    lo = np.r_[np.full(n_b, -np.inf), np.full(n_taus, TAU_BOUNDS[0])]
    hi = np.r_[np.full(n_b, np.inf), np.full(n_taus, TAU_BOUNDS[1])]

    def resid(p):
        X = _design(t, tuple(p[n_b:]))
        return w * (X @ p[:n_b] - y)

    sol = least_squares(resid, x0, bounds=(lo, hi), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)
    p = sol.x
    fitted = _design(t, tuple(p[n_b:])) @ p[:n_b]
    rmse = float(np.sqrt(np.mean((fitted - y) ** 2)))
    return p, rmse


def fit_nelson_siegel(t, y, weights=None) -> NSParams:
    p, rmse = _fit(t, y, weights, 1)
    return NSParams(*map(float, p), rmse=rmse)


def fit_svensson(t, y, weights=None) -> NSSParams:
    p, rmse = _fit(t, y, weights, 2)
    b0, b1, b2, b3, tau1, tau2 = map(float, p)
    return NSSParams(b0, b1, b2, b3, tau1, tau2, rmse=rmse)
