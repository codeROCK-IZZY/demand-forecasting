"""Make forecasts add up: the stores must sum to their region, the regions to the total.

bottom-up   forecast the bottom series only and add them up
MinT        combine every level's own forecast, weighting by how wrong each level has been
            (Wickramasuriya, Athanasopoulos & Hyndman, 2019), with the error covariance shrunk towards
            its diagonal (Schäfer & Strimmer, 2005) because there are more series than error samples
"""

from __future__ import annotations

import numpy as np


def bottom_up(s: np.ndarray, base: np.ndarray, n_bottom: int) -> np.ndarray:
    out: np.ndarray = s @ base[-n_bottom:]
    return out


def shrunk_covariance(errors: np.ndarray) -> np.ndarray:
    """errors: (n_series, n_samples). Shrinks correlations towards zero with the Schäfer-Strimmer lambda."""
    e = errors - errors.mean(axis=1, keepdims=True)
    n = e.shape[1]
    sd = e.std(axis=1) + 1e-9
    z = e / sd[:, None]
    r = z @ z.T / n
    w = np.einsum("in,jn->ijn", z, z)
    var_r = n / (n - 1) ** 3 * ((w - r[:, :, None]) ** 2).sum(axis=2)
    off = ~np.eye(len(r), dtype=bool)
    lam = float(np.clip(var_r[off].sum() / (r[off] ** 2).sum(), 0.0, 1.0))
    shrunk = r * (1 - lam)
    np.fill_diagonal(shrunk, 1.0)
    out: np.ndarray = shrunk * np.outer(sd, sd)
    return out


def mint(s: np.ndarray, base: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Reconcile base forecasts (n_series, h) with error covariance w (n_series, n_series)."""
    w_inv = np.linalg.pinv(w)
    g = np.linalg.solve(s.T @ w_inv @ s, s.T @ w_inv)
    out: np.ndarray = s @ (g @ base)
    return out


def nonnegative(s: np.ndarray, reconciled: np.ndarray, n_bottom: int) -> np.ndarray:
    """Clip negative bottom forecasts to zero and add up again, so the result stays coherent."""
    out: np.ndarray = s @ np.maximum(reconciled[-n_bottom:], 0.0)
    return out
