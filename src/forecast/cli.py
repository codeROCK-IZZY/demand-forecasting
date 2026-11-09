"""`forecast`: run the backtest evaluation, or forecast the next four weeks for one store."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import check, run, to_json, to_markdown
    from .report import render

    r = run(seed=args.seed, step=args.step)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(to_json(r), newline="\n", encoding="utf-8")
    (out / "evaluation.md").write_text(to_markdown(r), newline="\n", encoding="utf-8")
    (out / "report.html").write_text(render(r), newline="\n", encoding="utf-8")
    print(f"{len(r.test_origins)} backtest blocks ({r.seconds:.0f}s) -> {out}")
    if args.gate:
        problems = check(r)
        for p in problems:
            print(f"GATE FAILED: {p}", file=sys.stderr)
        return 1 if problems else 0
    return 0


def cmd_store(args: argparse.Namespace) -> int:
    import numpy as np

    from . import backtest as bt
    from . import data, models, reconcile

    chain = data.generate(args.seed)
    hz = bt.build(chain)
    origin = data.DAYS - models.H  # pretend today is four weeks before the data ends
    m = models.GlobalGBM(max_iter=150).fit(hz.y, hz.cal, bt.train_origins(origin)[::-1][::4][::-1])
    q = m.predict(hz.y, hz.cal, origin)
    med = reconcile.bottom_up(hz.s, q[0.5], chain.n_bottom)
    row = hz.labels.index(f"store {args.store}")
    actual = hz.y[row, origin : origin + models.H]
    print(f"store {args.store}, next {models.H} days from day {origin}")
    print("week   forecast   daily p10-p90, summed   actual")
    for w in range(4):
        sl = slice(7 * w, 7 * w + 7)
        print(
            f"{w + 1:>4} {med[row, sl].sum():>10.0f}   {q[0.1][row, sl].sum():>7.0f} - {q[0.9][row, sl].sum():<7.0f}"
            f"           {np.nansum(actual[sl]):>8.0f}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="forecast", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate", help="calibrate, backtest eight four-week blocks, cost the stock decisions")
    e.add_argument("--seed", type=int, default=11)
    e.add_argument("--step", type=int, default=14, help="days between training origins")
    e.add_argument("--out", default=str(ROOT / "eval" / "reports"))
    e.add_argument("--gate", action="store_true")
    e.set_defaults(fn=cmd_evaluate)
    s = sub.add_parser("store", help="four weekly forecasts with intervals for one store")
    s.add_argument("store", type=int, choices=range(1, 13))
    s.add_argument("--seed", type=int, default=11)
    s.set_defaults(fn=cmd_store)
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
