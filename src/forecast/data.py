"""Three years of daily unit sales for a small retail chain, with the hierarchy and calendar a planner has.

Hierarchy: total -> 3 regions -> 12 stores -> 120 store x product series (10 products per store).

What makes it realistic:
  - weekly and yearly seasonality that differ by product category (garden peaks in spring, heating in winter)
  - promotions planned in advance (known for the forecast horizon), with lifts that differ by product
  - public holidays (stores sell more the days before, little on the day)
  - price changes with elasticity
  - slow movers: low, intermittent demand with many zero days
  - a store that opens a year late (short history)
  - a structural break: one product in one region loses 40% of its demand after a competitor opens
  - noise that grows with the level (negative binomial counts)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DAYS = 3 * 364
REGIONS = 3
STORES_PER_REGION = 4
PRODUCTS = 10
CATEGORIES = ["garden", "heating", "grocery", "toys", "outdoor"]


@dataclass(frozen=True)
class Chain:
    sales: np.ndarray  # (n_bottom, DAYS) units; NaN before a store opened
    promo: np.ndarray  # (n_bottom, DAYS) 0/1, planned in advance
    price: np.ndarray  # (n_bottom, DAYS) relative price (1.0 = list price)
    holiday: np.ndarray  # (DAYS,) 0/1
    store_of: np.ndarray  # (n_bottom,) store id
    region_of_store: np.ndarray  # (n_stores,) region id
    product_of: np.ndarray  # (n_bottom,) product id
    category_of_product: np.ndarray  # (PRODUCTS,) category id
    break_series: np.ndarray  # bottom series hit by the structural break
    break_day: int

    @property
    def n_bottom(self) -> int:
        return int(self.sales.shape[0])

    def summing_matrix(self) -> tuple[np.ndarray, list[str]]:
        """S such that all_levels = S @ bottom, with row labels (total, regions, stores, bottom)."""
        n_stores = len(self.region_of_store)
        rows, labels = [np.ones(self.n_bottom)], ["total"]
        for r in range(REGIONS):
            rows.append((self.region_of_store[self.store_of] == r).astype(float))
            labels.append(f"region {r + 1}")
        for s in range(n_stores):
            rows.append((self.store_of == s).astype(float))
            labels.append(f"store {s + 1}")
        rows += list(np.eye(self.n_bottom))
        labels += [f"store {self.store_of[i] + 1} / product {self.product_of[i] + 1}" for i in range(self.n_bottom)]
        return np.array(rows), labels


def generate(seed: int = 11) -> Chain:
    rng = np.random.default_rng(seed)
    n_stores = REGIONS * STORES_PER_REGION
    region_of_store = np.repeat(np.arange(REGIONS), STORES_PER_REGION)
    store_of = np.repeat(np.arange(n_stores), PRODUCTS)
    product_of = np.tile(np.arange(PRODUCTS), n_stores)
    n = len(store_of)
    t = np.arange(DAYS)
    dow = t % 7
    doy = t % 364

    category_of_product = rng.integers(0, len(CATEGORIES), PRODUCTS)
    peak = {0: 100, 1: 340, 2: 0, 3: 330, 4: 180}  # day of year each category peaks
    amp = {0: 0.8, 1: 0.9, 2: 0.08, 3: 0.6, 4: 0.5}
    weekly = np.array([0.85, 0.8, 0.85, 0.95, 1.15, 1.45, 1.0])

    holiday = np.zeros(DAYS)
    for y in range(3):
        for d in (0, 89, 120, 145, 236, 358, 359):
            holiday[min(DAYS - 1, y * 364 + d)] = 1
    pre_holiday = np.zeros(DAYS)
    for hd in np.flatnonzero(holiday):
        pre_holiday[max(0, int(hd) - 3) : int(hd)] = 1

    base = rng.lognormal(1.6, 0.9, n)  # mean daily units per series; some are slow movers
    store_size = rng.lognormal(0.0, 0.3, n_stores)
    base *= store_size[store_of]
    elasticity = rng.uniform(1.0, 2.5, n)
    promo_lift = rng.uniform(1.3, 2.6, PRODUCTS)

    promo = np.zeros((n, DAYS))
    for p in range(PRODUCTS):  # promotions run chain-wide for a product, a week at a time
        for start in rng.choice(np.arange(0, DAYS - 7, 7), 14, replace=False):
            promo[product_of == p, start : start + 7] = 1
    price = np.ones((n, DAYS))
    for i in range(n):
        for c in rng.integers(30, DAYS - 30, 2):
            price[i, c:] *= rng.uniform(0.9, 1.12)
    price *= np.where(promo > 0, 0.8, 1.0)

    cat = category_of_product[product_of]
    season = np.stack([1 + amp[int(c)] * np.cos(2 * np.pi * (doy - peak[int(c)]) / 364) for c in cat])
    trend = 1 + rng.normal(0.0, 0.12, n)[:, None] * t[None, :] / DAYS
    mu = (
        base[:, None]
        * weekly[dow][None, :]
        * season
        * trend
        * np.where(promo > 0, promo_lift[product_of][:, None], 1.0)
        * price ** (-elasticity[:, None])
        * (1 + 0.5 * pre_holiday)[None, :]
        * (1 - 0.7 * holiday)[None, :]
    )

    break_day = DAYS - 200
    hit = np.flatnonzero((region_of_store[store_of] == 1) & (product_of == 3))
    mu[hit, break_day:] *= 0.6

    k = 6.0  # negative binomial dispersion
    sales = rng.negative_binomial(k, k / (k + mu)).astype(float)
    late = 5  # store 6 opens a year in
    sales[store_of == late, :364] = np.nan
    return Chain(
        sales, promo, price, holiday, store_of, region_of_store, product_of, category_of_product, hit, break_day
    )
