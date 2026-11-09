"""Forecasters. Each takes the history before an origin and returns the next H days for every series.

seasonal naive    the same weekday last week, repeated
Holt-Winters      level + weekly seasonality, smoothing chosen per series from a small grid by
                  in-sample one-step error (vectorised across series)
global GBM        one gradient-boosted model for every series and level, trained on scaled targets
                  with lag, calendar, promotion and price features known at the origin; quantile loss
                  gives the 10th, 50th, 80th and 90th percentiles directly
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

H = 28
QUANTILES = (0.1, 0.5, 0.8, 0.9)


def seasonal_naive(y: np.ndarray, origin: int, h: int = H) -> np.ndarray:
    last_week = y[:, origin - 7 : origin]
    out: np.ndarray = np.nan_to_num(np.tile(last_week, (1, h // 7 + 1))[:, :h])
    return out


def holt_winters(y: np.ndarray, origin: int, h: int = H) -> np.ndarray:
    hist = np.nan_to_num(y[:, :origin])
    n, t = hist.shape
    best_err = np.full(n, np.inf)
    best_fc = np.zeros((n, h))
    for alpha in (0.05, 0.15, 0.35):
        for gamma in (0.05, 0.2):
            level = hist[:, :28].mean(axis=1)
            season = hist[:, :28].reshape(n, 4, 7).mean(axis=1) - level[:, None]
            err = np.zeros(n)
            for i in range(28, t):
                s = season[:, i % 7]
                pred = level + s
                e = hist[:, i] - pred
                if i >= t - 365:
                    err += e * e
                new_level = level + alpha * e
                season[:, i % 7] = s + gamma * (hist[:, i] - new_level - s)
                level = new_level
            fc = np.maximum(0.0, level[:, None] + season[:, [(t + k) % 7 for k in range(h)]])
            better = err < best_err
            best_err[better] = err[better]
            best_fc[better] = fc[better]
    return best_fc


@dataclass(frozen=True)
class Calendar:
    promo: np.ndarray  # (n_series, DAYS) share of children on promotion
    price: np.ndarray  # (n_series, DAYS) mean relative price
    holiday: np.ndarray  # (DAYS,)
    level: np.ndarray  # (n_series,) 0 total, 1 region, 2 store, 3 bottom
    category: np.ndarray  # (n_series,) product category for bottom series, -1 above
    product: np.ndarray  # (n_series,) product id for bottom series, -1 above


FEATURES = [
    "mean7",
    "mean28",
    "same_dow_last_week",
    "last_year",
    "last_year_week",
    "year_ratio",
    "h",
    "dow",
    "doy_sin",
    "doy_cos",
    "holiday",
    "pre_holiday",
    "promo",
    "price",
    "promo_now_vs_recent",
    "level",
    "category",
    "product",
]
CATEGORICAL = [FEATURES.index(c) for c in ("level", "category", "product")]


def _nanmean(a: np.ndarray, axis: int) -> np.ndarray:
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        return np.nanmean(a, axis=axis)


def features(y: np.ndarray, cal: Calendar, origin: int, h: int = H) -> tuple[np.ndarray, np.ndarray]:
    """Features for every series and horizon at one origin: (n_series * h, n_features), and the scale per row."""
    n = y.shape[0]
    scale = np.nan_to_num(_nanmean(y[:, origin - 56 : origin], 1), nan=0.0) + 0.5
    mean7 = _nanmean(y[:, origin - 7 : origin], 1) / scale
    mean28 = _nanmean(y[:, origin - 28 : origin], 1) / scale
    year_ago = _nanmean(y[:, origin - 364 : origin - 336], 1) / scale
    recent_promo = cal.promo[:, origin - 28 : origin].mean(axis=1)
    cols = []
    for k in range(h):
        d = origin + k
        doy = 2 * np.pi * (d % 364) / 364
        pre = float(cal.holiday[d + 1 : d + 4].any()) if d + 1 < len(cal.holiday) else 0.0
        cols.append(
            np.column_stack(
                [
                    mean7,
                    mean28,
                    y[:, origin - 7 + k % 7] / scale,
                    y[:, d - 364] / scale,
                    _nanmean(y[:, d - 367 : d - 360], 1) / scale,
                    mean28 / np.where(year_ago > 0, year_ago, np.nan),
                    np.full(n, k + 1),
                    np.full(n, d % 7),
                    np.full(n, np.sin(doy)),
                    np.full(n, np.cos(doy)),
                    np.full(n, cal.holiday[d]),
                    np.full(n, pre),
                    cal.promo[:, d],
                    cal.price[:, d],
                    cal.promo[:, d] - recent_promo,
                    cal.level,
                    cal.category,
                    cal.product,
                ]
            )
        )
    x = np.stack(cols, axis=1).reshape(n * h, len(FEATURES))
    return x, np.repeat(scale, h)


@dataclass
class GlobalGBM:
    max_iter: int = 300
    learning_rate: float = 0.06
    seed: int = 0
    models: dict[float, HistGradientBoostingRegressor] = field(default_factory=dict)

    def fit(self, y: np.ndarray, cal: Calendar, origins: list[int], h: int = H) -> GlobalGBM:
        xs, ts = [], []
        for o in origins:
            x, s = features(y, cal, o, h)
            target = y[:, o : o + h].reshape(-1) / s
            ok = ~np.isnan(target)
            xs.append(x[ok])
            ts.append(target[ok])
        x_all, t_all = np.vstack(xs), np.concatenate(ts)
        for q in QUANTILES:
            m = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                max_iter=self.max_iter,
                learning_rate=self.learning_rate,
                max_leaf_nodes=31,
                categorical_features=CATEGORICAL,
                random_state=self.seed,
            )
            self.models[q] = m.fit(x_all, t_all)
        return self

    def predict(self, y: np.ndarray, cal: Calendar, origin: int, h: int = H) -> dict[float, np.ndarray]:
        x, s = features(y, cal, origin, h)
        out = {q: np.maximum(0.0, m.predict(x) * s).reshape(-1, h) for q, m in self.models.items()}
        # quantile models are trained separately; keep them in order
        stacked = np.sort(np.stack([out[q] for q in QUANTILES]), axis=0)
        return {q: stacked[i] for i, q in enumerate(QUANTILES)}
