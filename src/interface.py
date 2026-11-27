"""Command-line interface for the Austrian Lotto statistical analysis bot."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import database as db
from models import MODEL_REGISTRY, get_model

REPO_ROOT = Path(__file__).resolve().parent.parent
DISCLAIMER = ("DISCLAIMER: Lotto 6 aus 45 draws are designed to be random. No statistical model can reliably "
              "predict lottery numbers; every combination has the same 1 in 8,145,060 jackpot chance. "
              "This is a research/analysis tool - do not use it as a reason to spend more money.")


def _print_md(name: str) -> int:
    path = REPO_ROOT / name
    if not path.exists():
        print(f"{name} not found")
        return 1
    print(path.read_text(encoding="utf-8"))
    return 0


def choose_model(conn) -> tuple[str, str]:
    """Best model from the latest development backtest only if its Brier CI lies entirely below baseline."""
    rows = [r for r in db.get_latest_backtest(conn) if r["status"] == "ok" and r["model_name"] != "uniform_baseline"]
    winners = sorted((r for r in rows if r["delta_ci_high"] is not None and r["delta_ci_high"] < 0),
                     key=lambda r: r["mean_brier"])
    if winners:
        return winners[0]["model_name"], "beat uniform baseline in latest backtest (95% CI)"
    return "uniform_baseline", "no model beat the uniform baseline with confidence in backtesting"


def cmd_status(conn, args) -> int:
    latest = db.latest_draw(conn)
    print(f"Draws stored: {db.count_draws(conn)}")
    if latest:
        print(f"Latest draw:  {latest.draw_date}  {list(latest.numbers)}  Zz {latest.additional_number}")
    upd = db.last_update(conn)
    if upd:
        print(f"Last update:  {upd['started_at']} status={upd['status']} new_draws={upd['new_draws']}")
    pending = db.get_forecasts(conn, scored=False)
    if pending:
        print(f"Pending forecasts: {len(pending)} (target {pending[0]['target_draw_date']})")
    return 0


def cmd_suggest(conn, args) -> int:
    draws = db.get_draws(conn)
    name, reason = (args.model, "selected by user") if args.model else choose_model(conn)
    model = get_model(name).fit(draws)
    nums = model.suggest_numbers()
    print(f"Suggestion: {' '.join(f'{n:2d}' for n in nums)}")
    print(f"Model: {name} ({reason}); trained on {len(draws)} draws")
    return 0


def cmd_backtest(conn, args) -> int:
    split = "holdout" if args.holdout else "development"
    if args.run:
        from backtester import run_backtest
        run_backtest(conn, split=split, refit_every=args.refit_every, max_forecasts=args.max_forecasts)
    rows = db.get_latest_backtest(conn, split)
    if not rows:
        print(f"No {split} backtest results yet. Run: python src/interface.py backtest --run")
        return 1
    print(f"Backtest ({split}) run {rows[0]['run_id']} at {rows[0]['created_at']}")
    print(f"{'model':<20}{'N':>6}{'Brier':>11}{'delta vs uniform [95% CI]':>36}{'matches [95% CI]':>26}")
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['model_name']:<20} FAILED: {r['error']}")
            continue
        print(f"{r['model_name']:<20}{r['n_forecasts']:>6}{r['mean_brier']:>11.6f}"
              f"{r['delta_brier']:>+12.2e} [{r['delta_ci_low']:+.2e},{r['delta_ci_high']:+.2e}]"
              f"{r['mean_matches']:>8.3f} [{r['matches_ci_low']:.3f},{r['matches_ci_high']:.3f}]")
    print("Fair-lottery expectation: 0.800 matches per line; negative delta = better than uniform.")
    return 0


def cmd_update(conn, args) -> int:
    from scheduler import run_cycle
    print(json.dumps(run_cycle(conn, full=args.full), indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Argument parser with all sub-commands."""
    ap = argparse.ArgumentParser(prog="lotto-agent", description=__doc__, epilog=DISCLAIMER)
    ap.add_argument("--db", type=Path, default=None, help="SQLite path (default data/lotto.db)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="create/migrate the database")
    sub.add_parser("status", help="latest draw and data update status")
    s = sub.add_parser("suggest", help="generate a 6-number suggestion")
    s.add_argument("--model", choices=sorted(MODEL_REGISTRY))
    b = sub.add_parser("backtest", help="show (or run) backtest results vs random baseline")
    b.add_argument("--run", action="store_true")
    b.add_argument("--holdout", action="store_true")
    b.add_argument("--refit-every", type=int, default=5)
    b.add_argument("--max-forecasts", type=int, default=None)
    sub.add_parser("strategy", help="show strategy.md")
    sub.add_parser("experiments", help="show experiments.md")
    u = sub.add_parser("update", help="manual data update and model refresh")
    u.add_argument("--full", action="store_true", help="download the complete history")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    print(DISCLAIMER + "\n")
    if args.command == "strategy":
        return _print_md("strategy.md")
    if args.command == "experiments":
        return _print_md("experiments.md")
    conn = db.connect(args.db)
    try:
        if args.command == "init":
            print(f"Database ready at {args.db or db.DEFAULT_DB_PATH}")
            return 0
        return {"status": cmd_status, "suggest": cmd_suggest, "backtest": cmd_backtest,
                "update": cmd_update}[args.command](conn, args)
    except Exception as exc:
        logging.getLogger(__name__).debug("error", exc_info=True)
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
