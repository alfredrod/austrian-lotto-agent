"""Periodic update cycle: fetch new draws, score saved forecasts, refit models, forecast the next draw.

Austrian Lotto draws take place on Wednesday and Sunday evenings (Europe/Vienna). The cycle is idempotent:
draws are inserted once, forecasts are unique per (target date, model) and immutable, and scoring happens
exactly once. A file lock prevents concurrent runs. Under-filled forecasts for draws that have not happened
yet are topped up to COMBINATIONS_PER_MODEL lines at the end of every cycle.
"""
from __future__ import annotations

import argparse
import fcntl
import logging
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytz
import requests

import backfill
import data_collector
import database as db
from models import default_models

logger = logging.getLogger(__name__)
VIENNA = pytz.timezone("Europe/Vienna")
DRAW_WEEKDAYS = (2, 6)  # Wednesday, Sunday
DRAW_RESULTS_HOUR = 20  # results reliably available after ~20:00 Vienna time
STALE_AFTER_DAYS = 2
REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_MD = REPO_ROOT / "strategy.md"
LOCK_PATH = REPO_ROOT / "data" / ".scheduler.lock"
COMBINATIONS_PER_MODEL: int = int(os.environ.get("LOTTO_COMBINATIONS_PER_MODEL", "10"))
BACKFILL_ENABLED: bool = os.environ.get("LOTTO_BACKFILL", "1").strip().lower() not in ("0", "false", "no", "off")
T = TypeVar("T")


def now_vienna() -> datetime:
    """Current time in Europe/Vienna."""
    return datetime.now(VIENNA)


def next_draw_date(after: date) -> date:
    """First Wednesday/Sunday strictly after `after`."""
    d = after + timedelta(days=1)
    while d.weekday() not in DRAW_WEEKDAYS:
        d += timedelta(days=1)
    return d


def expected_latest_draw(now: datetime) -> date:
    """Most recent draw date whose results should already be published at `now` (Vienna)."""
    d = now.date()
    if d.weekday() in DRAW_WEEKDAYS and now.hour < DRAW_RESULTS_HOUR:
        d -= timedelta(days=1)
    while d.weekday() not in DRAW_WEEKDAYS:
        d -= timedelta(days=1)
    return d


def with_retries(fn: Callable[[], T], attempts: int = 3, base_delay: float = 30.0) -> T:
    """Call fn, retrying transient network errors with exponential backoff."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            if attempt == attempts:
                raise
            delay = base_delay * 2 ** (attempt - 1)
            logger.warning("Transient error (attempt %d/%d): %s - retrying in %.0fs", attempt, attempts, exc, delay)
            time.sleep(delay)
    raise RuntimeError("unreachable")


def score_pending_forecasts(conn) -> int:
    """Score every unscored forecast whose target draw is now known. Returns number scored."""
    import json
    scored = 0
    for f in db.get_unscored_forecasts(conn):
        draw = db.get_draw_by_date(conn, date.fromisoformat(f["target_draw_date"]))
        if draw is None:
            continue
        nums = set(json.loads(f["numbers"]))
        probs = np.array(json.loads(f["probabilities"]))
        y = np.zeros(45)
        y[np.array(draw.numbers) - 1] = 1
        db.score_forecast(conn, f["id"], len(nums & set(draw.numbers)), draw.additional_number in nums,
                          float(((probs - y) ** 2).mean()))
        for c in db.get_forecast_combinations(conn, f["id"]):
            cnums = set(json.loads(c["numbers"]))
            db.score_forecast_combination(conn, c["id"], len(cnums & set(draw.numbers)),
                                          draw.additional_number in cnums)
        scored += 1
        logger.info("Scored forecast %d (%s, %s): %d matches", f["id"], f["model_name"], draw.draw_date,
                    len(nums & set(draw.numbers)))
    return scored


def make_forecasts(conn, target: date) -> int:
    """Fit all models on all stored draws and save immutable forecasts for `target`. Returns number created."""
    draws = db.get_draws(conn, before=target)
    if not draws:
        logger.error("No draws stored; cannot forecast")
        return 0
    created = 0
    rng = np.random.default_rng()
    for model in default_models():
        try:
            model.fit(draws)
            probs = model.predict_probabilities()
            mv = db.insert_model_version(conn, model.name, model.params, len(draws), draws[-1].draw_date, probs)
            combos = model.suggest_combinations(COMBINATIONS_PER_MODEL, rng)
            if db.insert_forecast(conn, target, model.name, combos[0], probs,
                                  draws[-1].draw_date, mv, combinations=combos) is not None:
                created += 1
        except Exception:
            logger.exception("Forecast failed for model %s", model.name)
    return created


def backfill_pending_combinations(conn, today: Optional[date] = None) -> dict[str, Any]:
    """Top up under-filled forecasts for draws that have not happened yet.

    Forecasts saved before the multi-line feature hold a single combination. The repair is
    INSERT-only (see ``backfill``), so the immutability triggers stay intact, and it is idempotent:
    a forecast that already holds COMBINATIONS_PER_MODEL lines is skipped. Targets that are already
    drawn are left alone, because combinations added afterwards would never be scored and would
    blur the record of what was actually played.
    """
    today = today or now_vienna().date()
    targets = sorted({date.fromisoformat(f["target_draw_date"]) for f in db.get_forecasts(conn)
                      if date.fromisoformat(f["target_draw_date"]) >= today})
    report: dict[str, Any] = {"targets": [t.isoformat() for t in targets], "forecasts_processed": 0,
                              "rows_inserted": 0}
    for target in targets:
        rep = backfill.backfill_combinations(conn, target=target)
        report["forecasts_processed"] += rep["forecasts_processed"]
        report["rows_inserted"] += rep["rows_inserted"]
    if report["rows_inserted"]:
        logger.info("Backfilled %d combination(s) across %d forecast(s) for %s",
                    report["rows_inserted"], report["forecasts_processed"], ", ".join(report["targets"]))
    return report


def update_strategy_data_state(conn, path: Path = STRATEGY_MD) -> None:
    """Rewrite the auto-generated data-state block in strategy.md."""
    if not path.exists():
        return
    latest = db.latest_draw(conn)
    scored = db.get_forecasts(conn, scored=True)
    tally: dict[str, list[int]] = {}
    for f in scored:
        tally.setdefault(f["model_name"], []).append(f["matched_count"])
    lines = [f"- Draws stored: {db.count_draws(conn)}",
             f"- Latest draw: {latest.draw_date} {list(latest.numbers)} + Zz {latest.additional_number}" if latest
             else "- Latest draw: none",
             f"- Holdout start date (frozen): {db.get_meta(conn, 'holdout_start_date') or 'not yet set'}",
             f"- Live forecasts scored: {len(scored)}"]
    for name, m in sorted(tally.items()):
        lines.append(f"  - {name}: {len(m)} forecasts, mean matches {np.mean(m):.3f} (fair expectation 0.800)")
    block = "<!-- DATA_STATE_START -->\n" + "\n".join(lines) + "\n<!-- DATA_STATE_END -->"
    text = path.read_text(encoding="utf-8")
    new = re.sub(r"<!-- DATA_STATE_START -->.*?<!-- DATA_STATE_END -->", block, text, flags=re.S)
    if new != text:
        path.write_text(new, encoding="utf-8")


def run_cycle(conn, session: Optional[requests.Session] = None, now: Optional[datetime] = None,
              full: bool = False, backfill_enabled: Optional[bool] = None) -> dict[str, Any]:
    """One full update cycle. Returns a summary dict; errors are logged to update_log."""
    now = now or now_vienna()
    if backfill_enabled is None:
        backfill_enabled = BACKFILL_ENABLED
    log_id = db.log_update_start(conn)
    summary: dict[str, Any] = {"new_draws": 0, "scored": 0, "forecasts": 0, "backfilled": 0,
                               "stale": False, "status": "ok"}
    try:
        try:
            rep = with_retries(lambda: data_collector.collect(conn, session, recent_only=not full and
                                                              db.count_draws(conn) > 0, today=now.date()))
            summary["new_draws"] = rep.inserted
            if rep.errors:
                summary["status"] = "partial"
                summary["errors"] = rep.errors
        except Exception as exc:
            logger.error("Data collection failed: %s", exc)
            summary["status"] = "collection_failed"
            summary["errors"] = [str(exc)]
        latest = db.latest_draw(conn)
        expected = expected_latest_draw(now)
        if latest is None or (expected - latest.draw_date).days >= STALE_AFTER_DAYS:
            summary["stale"] = True
            logger.warning("STALE DATA: latest stored draw %s, expected %s", latest.draw_date if latest else None,
                           expected)
        summary["scored"] = score_pending_forecasts(conn)
        if latest is not None:
            target = next_draw_date(max(latest.draw_date, expected))
            summary["target_draw"] = target.isoformat()
            summary["forecasts"] = make_forecasts(conn, target)
        if backfill_enabled:
            try:
                summary["backfilled"] = backfill_pending_combinations(conn, today=now.date())["rows_inserted"]
            except Exception as exc:
                logger.exception("Combination backfill failed")
                summary["backfill_error"] = str(exc)
        update_strategy_data_state(conn)
    except Exception as exc:
        logger.exception("Update cycle failed")
        summary["status"] = "failed"
        summary["errors"] = summary.get("errors", []) + [str(exc)]
    db.log_update_finish(conn, log_id, summary["status"], summary["new_draws"],
                         f"{summary} ")
    return summary


def run_locked(db_path: Optional[Path] = None, full: bool = False,
               backfill_enabled: Optional[bool] = None) -> dict[str, Any]:
    """Run a cycle under an exclusive file lock (skips if another run is active)."""
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_PATH, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.warning("Another scheduler run is in progress; skipping")
            return {"status": "skipped"}
        conn = db.connect(db_path)
        try:
            return run_cycle(conn, full=full, backfill_enabled=backfill_enabled)
        finally:
            conn.close()


def main(argv: Optional[list[str]] = None) -> int:
    """CLI: --check-new (single cycle, for cron/GitHub Actions) or --daemon (loop)."""
    ap = argparse.ArgumentParser(description="Austrian Lotto update scheduler")
    ap.add_argument("--check-new", action="store_true", help="run one update cycle and exit")
    ap.add_argument("--full", action="store_true", help="re-download the full history")
    ap.add_argument("--no-backfill", action="store_true",
                    help="do not top up under-filled forecasts (also: LOTTO_BACKFILL=0)")
    ap.add_argument("--daemon", action="store_true", help="run forever, checking every --interval minutes")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--db", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    backfill_enabled = not args.no_backfill
    if args.daemon:
        import schedule
        schedule.every(args.interval).minutes.do(run_locked, args.db, args.full, backfill_enabled)
        run_locked(args.db, args.full, backfill_enabled)
        while True:
            schedule.run_pending()
            time.sleep(30)
    summary = run_locked(args.db, args.full, backfill_enabled)
    print(summary)
    return 0 if summary.get("status") in ("ok", "skipped", "partial") else 1


if __name__ == "__main__":
    sys.exit(main())
