"""Tests for the combination backfill utility."""
import json
import sqlite3
from datetime import date, timedelta

import numpy as np
import pytest

import backfill
import database as db
import scheduler


def _make_draws(n: int = 120, start: date = date(2024, 1, 3)) -> list:
    rng = np.random.default_rng(7)
    draws = []
    for i in range(n):
        numbers = tuple(sorted(int(x) for x in rng.choice(np.arange(1, 46), 6, replace=False)))
        draws.append(db.Draw(draw_date=start + timedelta(days=7 * i), numbers=numbers,
                             additional_number=int(rng.integers(1, 46)), source="test"))
    return draws


@pytest.fixture()
def conn():
    c = db.connect(":memory:")
    db.insert_draws(c, _make_draws())
    yield c
    c.close()


def _legacy_forecast(conn, target, model_name="uniform_baseline", numbers=(1, 2, 3, 4, 5, 6),
                     probabilities=None, cutoff=None):
    """Insert a pre-v2 style forecast: n_combinations = 1 and no combination rows."""
    probs = probabilities if probabilities is not None else [6 / 45] * 45
    cur = conn.execute(
        "INSERT INTO forecasts (created_at,target_draw_date,model_name,model_version_id,numbers,"
        "probabilities,data_cutoff_date,n_combinations) VALUES (?,?,?,?,?,?,?,1)",
        (db.utcnow(), target.isoformat(), model_name, None, json.dumps(sorted(numbers)),
         json.dumps([round(float(p), 6) for p in probs]),
         cutoff.isoformat() if cutoff else None),
    )
    conn.commit()
    return int(cur.lastrowid)


def test_backfill_adds_missing_combinations(conn):
    fid = _legacy_forecast(conn, date(2024, 6, 5), cutoff=date(2024, 5, 29))
    assert db.get_forecast_combinations(conn, fid) == []

    report = backfill.backfill_combinations(conn, count=10, seed=1)

    assert report["status"] == "ok"
    assert report["rows_inserted"] == 10
    rows = db.get_forecast_combinations(conn, fid)
    assert [r["combo_index"] for r in rows] == list(range(10))
    assert json.loads(rows[0]["numbers"]) == [1, 2, 3, 4, 5, 6]
    assert conn.execute("SELECT n_combinations FROM forecasts WHERE id = ?", (fid,)).fetchone()[0] == 10
    assert len({r["numbers"] for r in rows}) == 10
    for r in rows:
        nums = json.loads(r["numbers"])
        assert len(nums) == 6 and len(set(nums)) == 6 and all(1 <= n <= 45 for n in nums)


def test_backfill_is_idempotent(conn):
    fid = _legacy_forecast(conn, date(2024, 6, 5), cutoff=date(2024, 5, 29))
    backfill.backfill_combinations(conn, count=10, seed=1)
    before = [r["numbers"] for r in db.get_forecast_combinations(conn, fid)]

    report = backfill.backfill_combinations(conn, count=10, seed=2)

    assert report["rows_inserted"] == 0
    assert report["forecasts_skipped"] == 1
    assert [r["numbers"] for r in db.get_forecast_combinations(conn, fid)] == before


def test_backfill_respects_data_cutoff(conn):
    cutoff = date(2024, 6, 5)
    fid = _legacy_forecast(conn, date(2024, 6, 12), cutoff=cutoff)
    forecast = conn.execute("SELECT * FROM forecasts WHERE id = ?", (fid,)).fetchone()
    draws = backfill._training_draws(conn, forecast)
    assert draws and all(d.draw_date <= cutoff for d in draws)
    assert draws[-1].draw_date == cutoff


def test_backfill_dry_run_writes_nothing(conn):
    fid = _legacy_forecast(conn, date(2024, 6, 5), cutoff=date(2024, 5, 29))
    report = backfill.backfill_combinations(conn, count=10, dry_run=True, seed=1)
    assert report["forecasts"][0]["status"] == "would_backfill"
    assert len(report["forecasts"][0]["new_combinations"]) == 10
    assert db.get_forecast_combinations(conn, fid) == []
    assert conn.execute("SELECT n_combinations FROM forecasts WHERE id = ?", (fid,)).fetchone()[0] == 1


def test_backfill_target_filter(conn):
    keep = _legacy_forecast(conn, date(2024, 6, 5), cutoff=date(2024, 5, 29))
    other = _legacy_forecast(conn, date(2024, 6, 12), model_name="bayesian_frequency",
                             cutoff=date(2024, 6, 5))
    report = backfill.backfill_combinations(conn, target=date(2024, 6, 5), count=10, seed=1)
    assert report["forecasts_processed"] == 1
    assert len(db.get_forecast_combinations(conn, keep)) == 10
    assert db.get_forecast_combinations(conn, other) == []


def test_forecast_numbers_still_immutable(conn):
    fid = _legacy_forecast(conn, date(2024, 6, 5), cutoff=date(2024, 5, 29))
    backfill.backfill_combinations(conn, count=10, seed=1)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE forecasts SET numbers = ? WHERE id = ?",
                     (json.dumps([7, 8, 9, 10, 11, 12]), fid))
    conn.rollback()


def test_scheduler_backfills_upcoming_targets_only(conn):
    past = _legacy_forecast(conn, date(2024, 6, 5), cutoff=date(2024, 5, 29))
    upcoming = _legacy_forecast(conn, date(2024, 6, 12), model_name="bayesian_frequency",
                                cutoff=date(2024, 6, 5))

    report = scheduler.backfill_pending_combinations(conn, today=date(2024, 6, 10))

    assert report["targets"] == ["2024-06-12"]
    assert report["rows_inserted"] == 10
    assert len(db.get_forecast_combinations(conn, upcoming)) == 10
    assert db.get_forecast_combinations(conn, past) == []
    assert conn.execute("SELECT n_combinations FROM forecasts WHERE id = ?", (past,)).fetchone()[0] == 1


def test_scheduler_backfill_is_idempotent(conn):
    fid = _legacy_forecast(conn, date(2024, 6, 12), cutoff=date(2024, 6, 5))

    first = scheduler.backfill_pending_combinations(conn, today=date(2024, 6, 10))
    before = [r["numbers"] for r in db.get_forecast_combinations(conn, fid)]
    second = scheduler.backfill_pending_combinations(conn, today=date(2024, 6, 10))

    assert first["rows_inserted"] == 10
    assert second["rows_inserted"] == 0
    assert [r["numbers"] for r in db.get_forecast_combinations(conn, fid)] == before


def test_scheduler_backfill_noop_without_upcoming_targets(conn):
    _legacy_forecast(conn, date(2024, 6, 5), cutoff=date(2024, 5, 29))

    report = scheduler.backfill_pending_combinations(conn, today=date(2024, 6, 10))

    assert report == {"targets": [], "forecasts_processed": 0, "rows_inserted": 0}
