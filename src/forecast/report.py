"""A self-contained HTML report of the evaluation."""

from __future__ import annotations

import html

from .evaluate import LEVELS, TARGET, Result

CSS = """
:root { --bg:#0b0e13; --panel:#121720; --line:#232c3a; --ink:#e8ecf2; --ink2:#9aa6b8; --s1:#3987e5; --s2:#d95926;
  --good:#3fcf8e; --bad:#ff6b6b; font-family: Inter, system-ui, sans-serif; }
* { box-sizing: border-box; } body { margin:0; background:var(--bg); color:var(--ink); line-height:1.5; }
main { max-width: 1180px; margin: 0 auto; padding: 40px 28px 64px; }
h1 { font-size: 30px; letter-spacing:-0.02em; margin: 0 0 6px; } h2 { font-size: 19px; margin: 34px 0 14px; }
.sub, .muted { color: var(--ink2); } .sub { margin: 0 0 24px; }
.tiles { display:grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.tile { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px 18px; }
.tile b { display:block; font-size: 28px; letter-spacing:-0.02em; } .tile span { color: var(--ink2); font-size: 13px; }
table { width:100%; border-collapse: collapse; background:var(--panel); border:1px solid var(--line);
  border-radius:12px; overflow:hidden; font-size: 13.5px; }
th, td { padding: 8px 12px; border-bottom:1px solid var(--line); text-align:right; }
th:first-child, td:first-child { text-align:left; } th { color: var(--ink2); font-weight:500; font-size:12px; }
td.best { color: var(--good); font-weight: 600; }
.cols { display:grid; grid-template-columns: 1fr 1fr; gap: 18px; }
"""


def render(r: Result) -> str:
    best = {lv: min(r.accuracy.items(), key=lambda kv, lv=lv: kv[1][lv]["wape"])[0] for lv in LEVELS}  # type: ignore[misc]
    sn = r.accuracy["seasonal naive"]["store x product"]["wape"]
    bu = r.accuracy["GBM bottom-up"]["store x product"]["wape"]
    stock = r.stock
    cost_gain = 1 - stock["GBM 80th percentile, conformal"]["cost"] / stock["seasonal naive (point)"]["cost"]
    tiles = [
        (f"{1 - bu / sn:.0%}", "less error than seasonal naive on store x product (WAPE)"),
        (
            f"{r.coverage['conformal']['store x product']:.1%}",
            f"of actuals inside the 80% interval after conformal calibration (raw: {r.coverage['raw quantiles']['store x product']:.1%})",
        ),
        (f"{cost_gain:.0%}", "lower stock cost ordering at the 80th percentile instead of seasonal naive"),
        (
            f"{r.coherence['GBM (base)']:.1%}",
            "of volume where unreconciled forecasts don't add up (0 after reconciliation)",
        ),
    ]
    t = "".join(f'<div class="tile"><b>{html.escape(a)}</b><span>{html.escape(b)}</span></div>' for a, b in tiles)
    acc = "".join(
        f"<tr><td>{html.escape(m)}</td>"
        + "".join(f'<td class="{"best" if best[lv] == m else ""}">{v[lv]["wape"]:.3f}</td>' for lv in LEVELS)
        + f"<td>{v['store x product']['bias']:+.1%}</td></tr>"
        for m, v in r.accuracy.items()
    )
    covr = "".join(
        f"<tr><td>{n}</td>" + "".join(f"<td>{r.coverage[n][lv]:.1%}</td>" for lv in LEVELS) + "</tr>"
        for n in r.coverage
    )
    first = next(iter(r.special.values()))
    hard = "".join(
        f"<tr><td>{c}</td>" + "".join(f"<td>{x:.3f}</td>" for x in v.values()) + "</tr>" for c, v in r.special.items()
    )
    base = stock["seasonal naive (point)"]["cost"]
    st = "".join(
        f"<tr><td>{html.escape(p)}</td><td>{v['fill_rate']:.1%}</td><td>{v['short_units']:,.0f}</td>"
        f"<td>{v['leftover_units']:,.0f}</td><td>{v['cost'] / base - 1:+.1%}</td></tr>"
        for p, v in stock.items()
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Demand Forecasting: evaluation</title><style>{CSS}</style></head><body><main>
<h1>Demand forecasts, backtested</h1>
<p class="sub">{r.series} series over {r.days} days (total, 3 regions, 12 stores, 120 store x product). Eight four-week
blocks, each forecast from data before it only; intervals calibrated on the six blocks before those.</p>
<div class="tiles">{t}</div>
<h2>Accuracy by level (WAPE; best in green)</h2>
<table><tr><th>method</th>{"".join(f"<th>{lv}</th>" for lv in LEVELS)}<th>bias, store x product</th></tr>{acc}</table>
<div class="cols"><div><h2>{TARGET:.0%} interval coverage</h2>
<table><tr><th></th>{"".join(f"<th>{lv}</th>" for lv in LEVELS)}</tr>{covr}</table></div>
<div><h2>Hard cases (WAPE)</h2><table><tr><th>case</th>{"".join(f"<th>{html.escape(m)}</th>" for m in first)}</tr>{hard}</table></div></div>
<h2>Stock decisions (a unit short costs 4, a unit left over 1)</h2>
<table><tr><th>policy</th><th>fill rate</th><th>units short</th><th>units left over</th><th>cost vs seasonal naive</th></tr>{st}</table>
<p class="muted">Median forecasts of skewed demand sit below the mean, hence the negative bias; that is what minimises
absolute error, and why stock is set from a higher quantile instead.</p>
</main></body></html>
"""
