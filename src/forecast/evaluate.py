"""The evaluation: calibrate on six four-week blocks, then backtest eight more, at every level of the hierarchy."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import backtest as bt
from . import data, models, reconcile

LEVELS = ["total", "region", "store", "store x product"]
TARGET = 0.8  # nominal coverage of the 10-90 interval
SHORTAGE, LEFTOVER = 4.0, 1.0  # cost of a unit short (lost margin) and a unit left over (waste)


@dataclass
class Result:
    series: int
    days: int
    test_origins: list[int]
    zero_share: float
    accuracy: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)  # method -> level -> metric
    coverage: dict[str, dict[str, float]] = field(default_factory=dict)  # raw/conformal -> level
    width: dict[str, dict[str, float]] = field(default_factory=dict)
    coherence: dict[str, float] = field(default_factory=dict)
    special: dict[str, dict[str, float]] = field(default_factory=dict)
    stock: dict[str, dict[str, float]] = field(default_factory=dict)
    seconds: float = 0.0


def scale_at(y: np.ndarray, origin: int) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        out: np.ndarray = np.nan_to_num(np.nanmean(y[:, origin - 56 : origin], axis=1)) + 0.5
    return out


def conformal_level(scores: np.ndarray, level: float) -> float:
    s = np.sort(scores[~np.isnan(scores)])
    k = min(len(s) - 1, int(np.ceil((len(s) + 1) * level)) - 1)
    return float(s[k])


def run(seed: int = 11, step: int = 14) -> Result:
    t0 = time.perf_counter()
    chain = data.generate(seed)
    hz = bt.build(chain)
    n_bottom = chain.n_bottom
    r = Result(len(hz.y), data.DAYS, bt.TEST_ORIGINS, float(np.nanmean(chain.sales == 0)))

    # ---- calibration: a model trained before the calibration blocks, scored on them
    cal_model = models.GlobalGBM().fit(hz.y, hz.cal, bt.train_origins(bt.CAL_ORIGINS[0])[::-1][:: step // 7][::-1])
    cqr: dict[int, list[np.ndarray]] = {k: [] for k in range(4)}
    upper: dict[int, list[np.ndarray]] = {k: [] for k in range(4)}
    resid = []
    for o in bt.CAL_ORIGINS:
        q = cal_model.predict(hz.y, hz.cal, o)
        truth = hz.y[:, o : o + models.H]
        s = scale_at(hz.y, o)[:, None]
        e = np.maximum(q[0.1] - truth, truth - q[0.9]) / s
        u = (truth - q[0.8]) / s
        for k in range(4):
            cqr[k].append(e[hz.level == k].ravel())
            upper[k].append(u[hz.level == k].ravel())
        resid.append(np.nan_to_num(truth - q[0.5]))
    c_int = {k: conformal_level(np.concatenate(cqr[k]), TARGET) for k in range(4)}
    c_up = {k: conformal_level(np.concatenate(upper[k]), 0.8) for k in range(4)}
    w = reconcile.shrunk_covariance(np.concatenate(resid, axis=1))

    # ---- test: refit on everything before the first test block, then roll through eight blocks
    model = models.GlobalGBM().fit(hz.y, hz.cal, bt.train_origins(bt.TEST_ORIGINS[0])[::-1][:: step // 7][::-1])
    scale_mase = bt.mase_scale(hz.y, bt.TEST_ORIGINS[0])
    methods = ["seasonal naive", "Holt-Winters", "GBM (base)", "GBM bottom-up", "GBM + WLS", "GBM + MinT"]
    abs_err = {m: dict.fromkeys(range(4), 0.0) for m in methods}
    actual = {m: dict.fromkeys(range(4), 0.0) for m in methods}
    signed = {m: dict.fromkeys(range(4), 0.0) for m in methods}
    mase: dict[str, dict[int, list[float]]] = {m: {k: [] for k in range(4)} for m in methods}
    cov: dict[str, dict[int, list[np.ndarray]]] = {n: {k: [] for k in range(4)} for n in ("raw quantiles", "conformal")}
    wid: dict[str, dict[int, list[np.ndarray]]] = {n: {k: [] for k in range(4)} for n in ("raw quantiles", "conformal")}
    incoherence = {"GBM (base)": 0.0, "GBM bottom-up": 0.0, "GBM + MinT": 0.0}
    brk = {m: [0.0, 0.0] for m in ("seasonal naive", "Holt-Winters", "GBM + MinT")}
    late = np.zeros(len(hz.y), bool)
    late[-n_bottom:] = chain.store_of == 5
    late_err = {m: [0.0, 0.0] for m in ("seasonal naive", "Holt-Winters", "GBM + MinT")}
    policies = ["seasonal naive (point)", "GBM median", "GBM 80th percentile", "GBM 80th percentile, conformal"]
    stock = {p: [0.0, 0.0, 0.0, 0.0] for p in policies}  # short, leftover, demand, served

    for o in bt.TEST_ORIGINS:
        truth = hz.y[:, o : o + models.H]
        ok = ~np.isnan(truth)
        y0 = np.nan_to_num(truth)
        q = model.predict(hz.y, hz.cal, o)
        s = scale_at(hz.y, o)[:, None]
        fc = {
            "seasonal naive": models.seasonal_naive(hz.y, o),
            "Holt-Winters": models.holt_winters(hz.y, o),
            "GBM (base)": q[0.5],
            "GBM bottom-up": reconcile.bottom_up(hz.s, q[0.5], n_bottom),
            "GBM + MinT": reconcile.nonnegative(hz.s, reconcile.mint(hz.s, q[0.5], w), n_bottom),
            "GBM + WLS": reconcile.nonnegative(hz.s, reconcile.mint(hz.s, q[0.5], np.diag(np.diag(w))), n_bottom),
        }
        for m, f in fc.items():
            for k in range(4):
                sel = (hz.level[:, None] == k) & ok
                abs_err[m][k] += float(np.abs(f - y0)[sel].sum())
                actual[m][k] += float(y0[sel].sum())
                signed[m][k] += float((f - y0)[sel].sum())
                rows = hz.level == k
                mase[m][k].extend(
                    list(np.nanmean(np.where(ok[rows], np.abs(f - y0)[rows], np.nan), axis=1) / scale_mase[rows])
                )
        for name in incoherence:
            f = fc[name]
            incoherence[name] += float(np.abs(hz.s @ f[-n_bottom:] - f).sum() / max(np.abs(f).sum(), 1e-9))
        lvl = hz.level[:, None]
        c_lo = np.vectorize(lambda k: c_int[int(k)])(lvl) * s
        bounds = {"raw quantiles": (q[0.1], q[0.9]), "conformal": (q[0.1] - c_lo, q[0.9] + c_lo)}
        for name, (lo, hi) in bounds.items():
            inside = (truth >= lo) & (truth <= hi)
            for k in range(4):
                sel = (hz.level[:, None] == k) & ok
                cov[name][k].append(inside[sel])
                wid[name][k].append(((hi - np.maximum(lo, 0)) / s)[sel])
        if o + models.H > chain.break_day:
            rows = len(hz.y) - n_bottom + chain.break_series
            cols = slice(max(0, chain.break_day - o), models.H)
            for m in brk:
                brk[m][0] += float(np.abs(fc[m][rows, cols] - y0[rows, cols]).sum())
                brk[m][1] += float(y0[rows, cols].sum())
        for m in late_err:
            late_err[m][0] += float(np.abs(fc[m][late] - y0[late]).sum())
            late_err[m][1] += float(y0[late].sum())
        # stock decisions for each store x product: stock this much each day (perishable, so leftovers are waste)
        b = slice(len(hz.y) - n_bottom, None)
        orders = {
            policies[0]: fc["seasonal naive"][b],
            policies[1]: q[0.5][b],
            policies[2]: q[0.8][b],
            policies[3]: q[0.8][b] + c_up[3] * s[b],
        }
        d = y0[b]
        m_ok = ok[b]
        for p, qty in orders.items():
            qty = np.round(np.maximum(qty, 0))
            stock[p][0] += float(np.maximum(d - qty, 0)[m_ok].sum())
            stock[p][1] += float(np.maximum(qty - d, 0)[m_ok].sum())
            stock[p][2] += float(d[m_ok].sum())
            stock[p][3] += float(np.minimum(d, qty)[m_ok].sum())

    for m in methods:
        r.accuracy[m] = {
            LEVELS[k]: {
                "wape": abs_err[m][k] / actual[m][k],
                "bias": signed[m][k] / actual[m][k],
                "mase": float(np.nanmean(mase[m][k])),
            }
            for k in range(4)
        }
    for name in cov:
        r.coverage[name] = {LEVELS[k]: float(np.concatenate(v).mean()) for k, v in cov[name].items()}
        r.width[name] = {LEVELS[k]: float(np.concatenate(v).mean()) for k, v in wid[name].items()}
    r.coherence = {m: v / len(bt.TEST_ORIGINS) for m, v in incoherence.items()}
    r.special = {
        "after the structural break": {m: a / b for m, (a, b) in brk.items()},
        "store opened a year late": {m: a / b for m, (a, b) in late_err.items()},
    }
    for p, (short, left, dem, served) in stock.items():
        r.stock[p] = {
            "fill_rate": served / dem,
            "short_units": short,
            "leftover_units": left,
            "cost": SHORTAGE * short + LEFTOVER * left,
        }
    r.seconds = time.perf_counter() - t0
    return r


def check(r: Result) -> list[str]:
    p = []
    sn, hw = r.accuracy["seasonal naive"], r.accuracy["Holt-Winters"]
    for m in ("GBM bottom-up", "GBM + WLS", "GBM + MinT"):
        for lv in LEVELS:
            if not r.accuracy[m][lv]["wape"] < min(sn[lv]["wape"], hw[lv]["wape"]):
                p.append(f"{lv}: {m} does not beat both baselines")
    if r.coherence["GBM + MinT"] > 1e-6:
        p.append(f"MinT forecasts do not add up ({r.coherence['GBM + MinT']:.2e})")
    c = r.coverage["conformal"]["store x product"]
    if abs(c - TARGET) > 0.03:
        p.append(f"conformal coverage {c:.3f} not within 3 points of {TARGET}")
    cost = {k: v["cost"] for k, v in r.stock.items()}
    if not cost["GBM 80th percentile, conformal"] < cost["GBM median"] < cost["seasonal naive (point)"]:
        p.append(f"stock costs out of order: {cost}")
    return p


def to_json(r: Result) -> str:
    return json.dumps(asdict(r), indent=2) + "\n"


def to_markdown(r: Result) -> str:
    L = [
        "# Evaluation",
        "",
        f"{r.series} series (1 total, 3 regions, 12 stores, 120 store x product) over {r.days} days; {r.zero_share:.0%} of "
        f"store x product days sell nothing. Calibrated on 6 four-week blocks, then backtested on {len(r.test_origins)} "
        "more, each forecast made from data before its block only. Generated by `forecast evaluate`.",
        "",
        "## Accuracy (WAPE, lower is better)",
        "",
        "| method | " + " | ".join(LEVELS) + " | MASE, store x product | bias, store x product |",
        "|---|" + "---:|" * (len(LEVELS) + 2),
    ]
    for m, lv in r.accuracy.items():
        L.append(
            f"| {m} | "
            + " | ".join(f"{lv[x]['wape']:.3f}" for x in LEVELS)
            + f" | {lv['store x product']['mase']:.3f} | {lv['store x product']['bias']:+.1%} |"
        )
    L += [
        "",
        f"Forecasts that don't add up (share of volume): GBM base {r.coherence['GBM (base)']:.2%}, "
        f"bottom-up {r.coherence['GBM bottom-up']:.0e}, MinT {r.coherence['GBM + MinT']:.0e}.",
        "",
        "## 80% intervals",
        "",
        "| | " + " | ".join(LEVELS) + " |",
        "|---|" + "---:|" * len(LEVELS),
    ]
    for name in r.coverage:
        L.append(f"| coverage, {name} | " + " | ".join(f"{r.coverage[name][x]:.1%}" for x in LEVELS) + " |")
        L.append(f"| width / scale, {name} | " + " | ".join(f"{r.width[name][x]:.2f}" for x in LEVELS) + " |")
    L += [
        "",
        "## Hard cases (WAPE)",
        "",
        "| case | " + " | ".join(next(iter(r.special.values()))) + " |",
        "|---|---:|---:|---:|",
    ]
    for case, v in r.special.items():
        L.append(f"| {case} | " + " | ".join(f"{x:.3f}" for x in v.values()) + " |")
    L += [
        "",
        f"## Stock decisions (each unit short costs {SHORTAGE:g}, each unit left over {LEFTOVER:g})",
        "",
        "| policy | fill rate | units short | units left over | cost | vs seasonal naive |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    base = r.stock["seasonal naive (point)"]["cost"]
    for p, v in r.stock.items():
        L.append(
            f"| {p} | {v['fill_rate']:.1%} | {v['short_units']:,.0f} | {v['leftover_units']:,.0f} | {v['cost']:,.0f} | "
            f"{v['cost'] / base - 1:+.1%} |"
        )
    L.append("")
    return "\n".join(L)
