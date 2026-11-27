"""Tests for walk-forward backtesting (no data leakage)."""
from datetime import date, timedelta

import numpy as np

import database as db
from backtester import UNIFORM_BRIER, run_backtest, score, split_holdout, walk_forward
from models import BaseModel, UniformRandomBaseline


class SpyModel(BaseModel):
    """Records the training data it sees."""

    name = "spy"
    seen: list[int] = []

    def _fit_weights(self, X):
        SpyModel.seen.append(X.shape[0])
        return np.ones(45)


def make_draws(n, seed=0):
    rng = np.random.default_rng(seed)
    start = date(2000, 1, 2)
    return [db.Draw(start + timedelta(days=3 * i), tuple(sorted(rng.choice(np.arange(1, 46), 6, replace=False))),
                    1 if i % 2 else 2) for i in range(n)]


def test_no_leakage():
    draws = make_draws(60)
    SpyModel.seen = []
    res = walk_forward(SpyModel, draws, 10, 60, refit_every=1)
    assert SpyModel.seen == list(range(10, 60))
    assert all(size <= idx for size, idx in zip(res.train_sizes, res.indices))
    assert all((o == res.outcomes[k]).all() for k, o in enumerate(res.outcomes))


def test_refit_every_never_uses_future():
    draws = make_draws(50)
    res = walk_forward(SpyModel, draws, 10, 50, refit_every=7)
    assert all(size <= idx for size, idx in zip(res.train_sizes, res.indices))


def test_uniform_scores_baseline():
    draws = make_draws(80)
    s = score(walk_forward(UniformRandomBaseline, draws, 20, 80))
    assert np.isclose(s["mean_brier"], UNIFORM_BRIER) and abs(s["delta_brier"]) < 1e-12


def test_split_holdout():
    assert split_holdout(list(range(100))) == 80


def test_run_backtest_records_results(tmp_path):
    conn = db.connect(":memory:")
    db.insert_draws(conn, make_draws(300))
    rows = run_backtest(conn, min_train=100, refit_every=10, log_markdown=False)
    assert {r["status"] for r in rows} == {"ok"}
    assert db.get_meta(conn, "holdout_start_date") is not None
    assert len(db.get_latest_backtest(conn)) == len(rows)
