"""SQLite storage layer: schema, migrations and CRUD helpers."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = Path(os.environ.get("LOTTO_DB_PATH", REPO_ROOT / "data" / "lotto.db"))


@dataclass(frozen=True)
class Draw:
    """A single validated Lotto 6 aus 45 draw."""

    draw_date: date
    numbers: tuple[int, ...]
    additional_number: int
    source: str = ""

    @property
    def sorted_numbers(self) -> tuple[int, ...]:
        """Main numbers in ascending order."""
        return tuple(sorted(self.numbers))


MIGRATIONS: list[str] = [
    # v1: initial schema
    """
    CREATE TABLE IF NOT EXISTS draws (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        draw_date TEXT NOT NULL UNIQUE,
        n1 INTEGER NOT NULL, n2 INTEGER NOT NULL, n3 INTEGER NOT NULL,
        n4 INTEGER NOT NULL, n5 INTEGER NOT NULL, n6 INTEGER NOT NULL,
        additional_number INTEGER NOT NULL,
        source TEXT,
        retrieved_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS model_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        params TEXT NOT NULL,
        trained_at TEXT NOT NULL,
        n_draws INTEGER NOT NULL,
        last_draw_date TEXT,
        probabilities TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS forecasts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        target_draw_date TEXT NOT NULL,
        model_name TEXT NOT NULL,
        model_version_id INTEGER REFERENCES model_versions(id),
        numbers TEXT NOT NULL,
        probabilities TEXT NOT NULL,
        data_cutoff_date TEXT,
        scored_at TEXT,
        matched_count INTEGER,
        additional_matched INTEGER,
        brier_score REAL,
        UNIQUE(target_draw_date, model_name)
    );
    CREATE TRIGGER IF NOT EXISTS forecasts_immutable
    BEFORE UPDATE OF created_at, target_draw_date, model_name, model_version_id,
                     numbers, probabilities, data_cutoff_date ON forecasts
    BEGIN SELECT RAISE(ABORT, 'forecasts are immutable'); END;
    CREATE TRIGGER IF NOT EXISTS forecasts_score_once
    BEFORE UPDATE OF scored_at, matched_count, brier_score ON forecasts
    WHEN OLD.scored_at IS NOT NULL
    BEGIN SELECT RAISE(ABORT, 'forecast already scored'); END;
    CREATE TRIGGER IF NOT EXISTS forecasts_no_delete
    BEFORE DELETE ON forecasts
    BEGIN SELECT RAISE(ABORT, 'forecasts cannot be deleted'); END;
    CREATE TABLE IF NOT EXISTS backtest_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        model_name TEXT NOT NULL,
        params TEXT,
        split TEXT NOT NULL,
        status TEXT NOT NULL,
        n_forecasts INTEGER,
        mean_brier REAL, brier_ci_low REAL, brier_ci_high REAL,
        mean_log_loss REAL,
        mean_matches REAL, matches_ci_low REAL, matches_ci_high REAL,
        baseline_brier REAL,
        delta_brier REAL, delta_ci_low REAL, delta_ci_high REAL,
        calibration TEXT,
        error TEXT
    );
    CREATE TABLE IF NOT EXISTS update_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        status TEXT NOT NULL,
        new_draws INTEGER DEFAULT 0,
        message TEXT
    );
    CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """,
    # v2: multiple combinations per forecast
    """
    ALTER TABLE forecasts ADD COLUMN n_combinations INTEGER NOT NULL DEFAULT 1;
    CREATE TABLE IF NOT EXISTS forecast_combinations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        forecast_id INTEGER NOT NULL REFERENCES forecasts(id),
        combo_index INTEGER NOT NULL,
        numbers TEXT NOT NULL,
        matched_count INTEGER,
        additional_matched INTEGER,
        UNIQUE(forecast_id, combo_index)
    );
    CREATE TRIGGER IF NOT EXISTS forecast_combinations_immutable
    BEFORE UPDATE OF forecast_id, combo_index, numbers ON forecast_combinations
    BEGIN SELECT RAISE(ABORT, 'forecast combinations are immutable'); END;
    CREATE TRIGGER IF NOT EXISTS forecast_combinations_score_once
    BEFORE UPDATE OF matched_count, additional_matched ON forecast_combinations
    WHEN OLD.matched_count IS NOT NULL
    BEGIN SELECT RAISE(ABORT, 'combination already scored'); END;
    CREATE TRIGGER IF NOT EXISTS forecast_combinations_no_delete
    BEFORE DELETE ON forecast_combinations
    BEGIN SELECT RAISE(ABORT, 'forecast combinations cannot be deleted'); END;
    """,
]


def utcnow() -> str:
    """Current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Optional[Path | str] = None) -> sqlite3.Connection:
    """Open (and migrate) the database. Use ':memory:' for tests."""
    path = str(db_path or DEFAULT_DB_PATH)
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    migrate(conn)
    return conn


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations using PRAGMA user_version. Returns the schema version."""
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for i, script in enumerate(MIGRATIONS[version:], start=version + 1):
        conn.executescript(script)
        conn.execute(f"PRAGMA user_version = {i}")
        logger.info("Applied DB migration v%d", i)
    conn.commit()
    return len(MIGRATIONS)


# ---------------------------------------------------------------- draws
def _row_to_draw(row: sqlite3.Row) -> Draw:
    return Draw(
        draw_date=date.fromisoformat(row["draw_date"]),
        numbers=tuple(row[f"n{i}"] for i in range(1, 7)),
        additional_number=row["additional_number"],
        source=row["source"] or "",
    )


def insert_draws(conn: sqlite3.Connection, draws: Iterable[Draw]) -> dict[str, int]:
    """Insert draws; existing identical dates are skipped, conflicting ones are logged and not overwritten."""
    stats = {"inserted": 0, "existing": 0, "conflicts": 0}
    now = utcnow()
    for d in draws:
        row = conn.execute("SELECT * FROM draws WHERE draw_date = ?", (d.draw_date.isoformat(),)).fetchone()
        if row is not None:
            existing = _row_to_draw(row)
            if existing.sorted_numbers == d.sorted_numbers and existing.additional_number == d.additional_number:
                stats["existing"] += 1
            else:
                stats["conflicts"] += 1
                logger.error("Conflicting record for %s: stored %s+%s vs new %s+%s (%s); keeping stored",
                             d.draw_date, existing.sorted_numbers, existing.additional_number,
                             d.sorted_numbers, d.additional_number, d.source)
            continue
        n = d.sorted_numbers
        conn.execute(
            "INSERT INTO draws (draw_date,n1,n2,n3,n4,n5,n6,additional_number,source,retrieved_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (d.draw_date.isoformat(), *(int(x) for x in n), int(d.additional_number), d.source, now),
        )
        stats["inserted"] += 1
    conn.commit()
    return stats


def get_draws(conn: sqlite3.Connection, before: Optional[date] = None) -> list[Draw]:
    """All draws in chronological order, optionally strictly before a date."""
    if before is None:
        rows = conn.execute("SELECT * FROM draws ORDER BY draw_date").fetchall()
    else:
        rows = conn.execute("SELECT * FROM draws WHERE draw_date < ? ORDER BY draw_date",
                            (before.isoformat(),)).fetchall()
    return [_row_to_draw(r) for r in rows]


def get_draw_by_date(conn: sqlite3.Connection, d: date) -> Optional[Draw]:
    """Draw for a given date, if stored."""
    row = conn.execute("SELECT * FROM draws WHERE draw_date = ?", (d.isoformat(),)).fetchone()
    return _row_to_draw(row) if row else None


def latest_draw(conn: sqlite3.Connection) -> Optional[Draw]:
    """Most recent stored draw."""
    row = conn.execute("SELECT * FROM draws ORDER BY draw_date DESC LIMIT 1").fetchone()
    return _row_to_draw(row) if row else None


def count_draws(conn: sqlite3.Connection) -> int:
    """Number of stored draws."""
    return conn.execute("SELECT COUNT(*) FROM draws").fetchone()[0]


# ---------------------------------------------------------------- meta
def get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    """Read a metadata value."""
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Write a metadata value."""
    conn.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                 (key, value))
    conn.commit()


# ---------------------------------------------------------------- models & forecasts
def insert_model_version(conn: sqlite3.Connection, model_name: str, params: dict[str, Any], n_draws: int,
                         last_draw_date: Optional[date], probabilities: Sequence[float]) -> int:
    """Record a fitted model version and return its id."""
    cur = conn.execute(
        "INSERT INTO model_versions (model_name,params,trained_at,n_draws,last_draw_date,probabilities)"
        " VALUES (?,?,?,?,?,?)",
        (model_name, json.dumps(params), utcnow(), n_draws,
         last_draw_date.isoformat() if last_draw_date else None, json.dumps([round(float(p), 6) for p in probabilities])),
    )
    conn.commit()
    return int(cur.lastrowid)


def insert_forecast(conn: sqlite3.Connection, target_draw_date: date, model_name: str, numbers: Sequence[int],
                    probabilities: Sequence[float], data_cutoff_date: Optional[date],
                    model_version_id: Optional[int] = None,
                    combinations: Optional[Sequence[Sequence[int]]] = None) -> Optional[int]:
    """Save an immutable forecast with all its combinations. Returns None if one already exists."""
    try:
        combos = [sorted(int(n) for n in c) for c in (combinations or [numbers])]
        cur = conn.execute(
            "INSERT INTO forecasts (created_at,target_draw_date,model_name,model_version_id,numbers,probabilities,"
            "data_cutoff_date,n_combinations) VALUES (?,?,?,?,?,?,?,?)",
            (utcnow(), target_draw_date.isoformat(), model_name, model_version_id,
             json.dumps(sorted(int(n) for n in numbers)), json.dumps([round(float(p), 6) for p in probabilities]),
             data_cutoff_date.isoformat() if data_cutoff_date else None, len(combos)),
        )
        fid = int(cur.lastrowid)
        conn.executemany(
            "INSERT INTO forecast_combinations (forecast_id,combo_index,numbers) VALUES (?,?,?)",
            [(fid, i, json.dumps(c)) for i, c in enumerate(combos)],
        )
        conn.commit()
        return fid
    except sqlite3.IntegrityError:
        conn.rollback()
        logger.info("Forecast for %s by %s already exists; not overwriting", target_draw_date, model_name)
        return None


def get_unscored_forecasts(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Forecasts not yet scored."""
    return conn.execute("SELECT * FROM forecasts WHERE scored_at IS NULL ORDER BY target_draw_date").fetchall()


def score_forecast(conn: sqlite3.Connection, forecast_id: int, matched: int, additional_matched: bool,
                   brier: float) -> None:
    """Attach the outcome to a forecast (allowed exactly once)."""
    conn.execute("UPDATE forecasts SET scored_at=?, matched_count=?, additional_matched=?, brier_score=? WHERE id=?",
                 (utcnow(), matched, int(additional_matched), brier, forecast_id))
    conn.commit()


def get_forecast_combinations(conn: sqlite3.Connection,
                              forecast_id: Optional[int] = None) -> list[sqlite3.Row]:
    """Return combinations of one forecast, or all forecasts when forecast_id is None."""
    q, params = "SELECT * FROM forecast_combinations", ()
    if forecast_id is not None:
        q += " WHERE forecast_id = ?"
        params = (forecast_id,)
    return conn.execute(q + " ORDER BY forecast_id, combo_index", params).fetchall()


def score_forecast_combination(conn: sqlite3.Connection, combo_id: int, matched: int,
                               additional_matched: bool) -> None:
    """Attach draw outcome to a single combination row (allowed exactly once)."""
    conn.execute(
        "UPDATE forecast_combinations SET matched_count=?, additional_matched=? WHERE id=?",
        (matched, int(additional_matched), combo_id),
    )
    conn.commit()


def get_forecasts(conn: sqlite3.Connection, scored: Optional[bool] = None) -> list[sqlite3.Row]:
    """List forecasts, optionally filtered by scored state."""
    q = "SELECT * FROM forecasts"
    if scored is True:
        q += " WHERE scored_at IS NOT NULL"
    elif scored is False:
        q += " WHERE scored_at IS NULL"
    return conn.execute(q + " ORDER BY target_draw_date, model_name").fetchall()


# ---------------------------------------------------------------- backtests & update log
def insert_backtest_result(conn: sqlite3.Connection, result: dict[str, Any]) -> int:
    """Store one backtest result row (successful or failed)."""
    cols = ["run_id", "created_at", "model_name", "params", "split", "status", "n_forecasts", "mean_brier",
            "brier_ci_low", "brier_ci_high", "mean_log_loss", "mean_matches", "matches_ci_low", "matches_ci_high",
            "baseline_brier", "delta_brier", "delta_ci_low", "delta_ci_high", "calibration", "error"]
    row = {c: result.get(c) for c in cols}
    row["created_at"] = row["created_at"] or utcnow()
    for c in ("params", "calibration"):
        if row[c] is not None and not isinstance(row[c], str):
            row[c] = json.dumps(row[c])
    cur = conn.execute(f"INSERT INTO backtest_results ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                       [row[c] for c in cols])
    conn.commit()
    return int(cur.lastrowid)


def get_latest_backtest(conn: sqlite3.Connection, split: str = "development") -> list[sqlite3.Row]:
    """Rows of the most recent backtest run for a split."""
    row = conn.execute("SELECT run_id FROM backtest_results WHERE split=? ORDER BY id DESC LIMIT 1", (split,)).fetchone()
    if not row:
        return []
    return conn.execute("SELECT * FROM backtest_results WHERE run_id=? ORDER BY id", (row["run_id"],)).fetchall()


def log_update_start(conn: sqlite3.Connection) -> int:
    """Open an update_log entry."""
    cur = conn.execute("INSERT INTO update_log (started_at,status) VALUES (?, 'running')", (utcnow(),))
    conn.commit()
    return int(cur.lastrowid)


def log_update_finish(conn: sqlite3.Connection, log_id: int, status: str, new_draws: int, message: str) -> None:
    """Close an update_log entry."""
    conn.execute("UPDATE update_log SET finished_at=?, status=?, new_draws=?, message=? WHERE id=?",
                 (utcnow(), status, new_draws, message[:2000], log_id))
    conn.commit()


def last_update(conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
    """Most recent update_log entry."""
    return conn.execute("SELECT * FROM update_log ORDER BY id DESC LIMIT 1").fetchone()
