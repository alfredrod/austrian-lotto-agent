"""Backfill combinations for forecasts saved before the multi-line feature.

Forecasts written before ``forecast_combinations`` existed carry ``n_combinations = 1`` and no
combination rows at all, so only one line per model is playable. The ``forecasts`` table is
immutable and its rows cannot be deleted, but two facts make a repair possible without rewriting
history:

* ``n_combinations`` is not listed in the ``forecasts_immutable`` trigger, so the counter can be
  updated.
* ``forecast_combinations`` only blocks UPDATE and DELETE; INSERT is allowed.

The stored ``numbers`` column is the line the model actually played, so it becomes ``combo_index`` 0
and is never regenerated. The remaining lines are sampled from the model rebuilt with the
hyper-parameters recorded in ``model_versions`` and refitted on exactly the draws that were
available when the forecast was made (``draw_date <= data_cutoff_date``), so no draw from after the
cutoff leaks into the backfill. The operation is idempotent: a forecast that already holds the
requested number of combinations is skipped and existing ``combo_index`` values are never
overwritten.

Usage::

    python src/backfill.py --dry-run             # report what would be added
    python src/backfill.py                       # add the missing lines
    python src/backfill.py --target 2026-11-29   # only one target draw
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import database as db
from models import get_model

logger = logging.getLogger(__name__)

COMBINATIONS_PER_MODEL: int = int(os.environ.get("LOTTO_COMBINATIONS_PER_MODEL", "10"))
PROBABILITY_DRIFT_TOLERANCE = 1e-4


def add_forecast_combinations(conn, forecast_id: int, combinations: Sequence[Sequence[int]],
                              start_index: int = 0) -> int:
    """Append combinations to an existing forecast and sync ``forecasts.n_combinations``.

    Only INSERTs into ``forecast_combinations`` and updates the counter column, both of which the
    immutability triggers permit. Existing rows are never touched, so repeated calls are safe.
    Returns the number of rows inserted.
    """
    if not combinations:
        return 0
    taken = {r["combo_index"] for r in conn.execute(
        "SELECT combo_index FROM forecast_combinations WHERE forecast_id = ?", (forecast_id,))}
    rows = []
    for offset, combo in enumerate(combinations):
        index = start_index + offset
        if index in taken:
            continue
        rows.append((forecast_id, index, json.dumps(sorted(int(n) for n in combo))))
    if not rows:
        return 0
    try:
        conn.executemany(
            "INSERT INTO forecast_combinations (forecast_id,combo_index,numbers) VALUES (?,?,?)", rows)
        total = conn.execute("SELECT COUNT(*) FROM forecast_combinations WHERE forecast_id = ?",
                             (forecast_id,)).fetchone()[0]
        conn.execute("UPDATE forecasts SET n_combinations = ? WHERE id = ?", (total, forecast_id))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return len(rows)


def _model_for_forecast(conn, forecast):
    """Rebuild the model with the hyper-parameters recorded for this forecast."""
    params: dict[str, Any] = {}
    if forecast["model_version_id"] is not None:
        row = conn.execute("SELECT params FROM model_versions WHERE id = ?",
                           (forecast["model_version_id"],)).fetchone()
        if row is not None:
            try:
                params = json.loads(row["params"] or "{}")
            except (TypeError, ValueError):
                params = {}
    try:
        return get_model(forecast["model_name"], **params)
    except (KeyError, TypeError) as exc:
        logger.warning("Cannot rebuild %s with params %s (%s); falling back to defaults",
                       forecast["model_name"], params, exc)
        return get_model(forecast["model_name"])


def _training_draws(conn, forecast) -> list:
    """Draws that were available when ``forecast`` was made (never any later ones)."""
    cutoff = date.fromisoformat(forecast["data_cutoff_date"]) if forecast["data_cutoff_date"] else None
    horizon = (cutoff + timedelta(days=1)) if cutoff else date.fromisoformat(forecast["target_draw_date"])
    return db.get_draws(conn, before=horizon)


def backfill_combinations(conn, target: Optional[date] = None, dry_run: bool = False,
                          seed: Optional[int] = None,
                          count: Optional[int] = None) -> dict[str, Any]:
    """Give every under-filled forecast its missing combinations. Returns a report dict."""
    wanted = count or COMBINATIONS_PER_MODEL
    if wanted < 1:
        return {"status": "error", "message": "combination count must be >= 1"}
    rng = np.random.default_rng(seed)
    report: dict[str, Any] = {"status": "ok", "target_count": wanted, "dry_run": dry_run,
                              "forecasts_processed": 0, "forecasts_skipped": 0, "rows_inserted": 0,
                              "forecasts": []}
    for f in db.get_forecasts(conn):
        if target is not None and f["target_draw_date"] != target.isoformat():
            continue
        existing = db.get_forecast_combinations(conn, f["id"])
        if len(existing) >= wanted:
            report["forecasts_skipped"] += 1
            continue
        try:
            model = _model_for_forecast(conn, f)
        except KeyError:
            report["forecasts"].append({"forecast_id": f["id"], "model": f["model_name"],
                                        "status": "unknown_model"})
            continue
        draws = _training_draws(conn, f)
        if not draws:
            report["forecasts"].append({"forecast_id": f["id"], "model": f["model_name"],
                                        "status": "no_training_draws"})
            continue
        model.fit(draws)
        probs = model.predict_probabilities()
        stored = np.asarray(json.loads(f["probabilities"]), dtype=float)
        drift = float(np.abs(probs - stored).max()) if probs.shape == stored.shape else float("inf")
        if existing:
            known = [json.loads(r["numbers"]) for r in existing]
            prefix: list[list[int]] = []
        else:
            known = [json.loads(f["numbers"])]
            prefix = [sorted(int(n) for n in known[0])]
        seen = {tuple(sorted(int(n) for n in c)) for c in known}
        extra: list[list[int]] = []
        need = wanted - len(known)
        for _ in range(100):
            if len(extra) >= need:
                break
            for cand in model.suggest_combinations(need - len(extra), rng):
                key = tuple(sorted(int(n) for n in cand))
                if key in seen:
                    continue
                seen.add(key)
                extra.append(list(key))
                if len(extra) >= need:
                    break
        if drift > PROBABILITY_DRIFT_TOLERANCE:
            logger.warning("Model %s no longer reproduces the probabilities stored for %s "
                           "(max abs drift %.2e); the added lines come from the current model code",
                           f["model_name"], f["target_draw_date"], drift)
        entry: dict[str, Any] = {"forecast_id": f["id"], "model": f["model_name"],
                                 "target_draw_date": f["target_draw_date"],
                                 "existing_combinations": len(known), "adding": len(extra),
                                 "probability_drift": round(drift, 8)}
        if len(extra) < need:
            entry["shortfall"] = need - len(extra)
        if dry_run:
            entry["status"] = "would_backfill"
            entry["new_combinations"] = prefix + extra
        else:
            entry["inserted"] = add_forecast_combinations(conn, f["id"], prefix + extra,
                                                          start_index=len(existing))
            entry["status"] = "backfilled"
            report["rows_inserted"] += entry["inserted"]
        report["forecasts_processed"] += 1
        report["forecasts"].append(entry)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Add missing combinations to already-saved forecasts")
    ap.add_argument("--db", type=Path, default=None, help="SQLite path (default data/lotto.db)")
    ap.add_argument("--target", type=date.fromisoformat, default=None,
                    help="only this target draw (YYYY-MM-DD)")
    ap.add_argument("--count", type=int, default=None,
                    help=f"combinations per model (default {COMBINATIONS_PER_MODEL})")
    ap.add_argument("--dry-run", action="store_true", help="report without writing anything")
    ap.add_argument("--seed", type=int, default=None, help="seed for reproducible sampling")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    conn = db.connect(args.db)
    try:
        report = backfill_combinations(conn, target=args.target, dry_run=args.dry_run,
                                       seed=args.seed, count=args.count)
    finally:
        conn.close()
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
