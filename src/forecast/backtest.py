"""Rolling-origin backtests over the whole hierarchy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import DAYS, Chain
from .models import Calendar, H

TEST_ORIGINS = [DAYS - 28 * k for k in range(8, 0, -1)]  # the last 8 four-week blocks
CAL_ORIGINS = [TEST_ORIGINS[0] - 28 * k for k in range(6, 0, -1)]  # the 6 blocks before, for calibration
FIRST_TRAIN_ORIGIN = 400  # needs a year of history for features


@dataclass(frozen=True)
class Hierarchy:
    s: np.ndarray  # summing matrix (n_all, n_bottom)
    labels: list[str]
    level: np.ndarray  # 0 total, 1 region, 2 store, 3 bottom
    y: np.ndarray  # (n_all, DAYS) every level, NaN where nothing was open yet
    cal: Calendar


def build(chain: Chain) -> Hierarchy:
    s, labels = chain.summing_matrix()
    level = np.array([0] + [1] * 3 + [2] * len(chain.region_of_store) + [3] * chain.n_bottom)
    bottom = chain.sales
    y = s @ np.nan_to_num(bottom)
    open_ = s @ (~np.isnan(bottom)).astype(float)
    y[open_ == 0] = np.nan
    y[-chain.n_bottom :] = bottom
    share = s / s.sum(axis=1, keepdims=True)
    category = np.full(len(s), -1)
    product = np.full(len(s), -1)
    category[-chain.n_bottom :] = chain.category_of_product[chain.product_of]
    product[-chain.n_bottom :] = chain.product_of
    cal = Calendar(share @ chain.promo, share @ chain.price, chain.holiday, level, category, product)
    return Hierarchy(s, labels, level, y, cal)


def train_origins(until: int) -> list[int]:
    return list(range(FIRST_TRAIN_ORIGIN, until - H + 1, 7))


def mase_scale(y: np.ndarray, until: int) -> np.ndarray:
    """In-sample mean absolute error of the seasonal-naive forecast, per series."""
    hist = y[:, :until]
    d = np.abs(hist[:, 7:] - hist[:, :-7])
    with np.errstate(invalid="ignore"):
        out: np.ndarray = np.nanmean(d, axis=1) + 1e-9
    return out
