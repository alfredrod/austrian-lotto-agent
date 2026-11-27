"""Walk-forward backtesting of all models against the uniform baseline.

Protocol: draws are processed chronologically; to forecast draw i a fresh model is fitted only on draws
[0, i) (refitting every `refit_every` draws, never using draw i or later). The last 20% of draws (boundary
frozen in the `meta` table at first run) is a holdout that is evaluated only with --holdout and never used to
choose hyper-parameters.
"""
from __future__ import annotations

import argparse
import logging
import sys
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import database as db
from models import K_PICK, N_NUMBERS, UNIFORM_P, BaseModel, default_models, to_indicator_matrix

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_MD = REPO_ROOT / "experiments.md"
HOLDOUT_FRACTION = 0.2
UNIFORM_BRIER = (K_PICK * (1 - UNIFORM_P) ** 2 + (N_NUMBERS - K_PICK) * UNIFORM_P ** 2) / N_NUMBERS


@dataclass
class WalkForwardResult:
    """Per-draw outputs of a walk-forward run."""

    model_name: str
    indices: list[int] = field(default_factory=list)
    probabilities: list[np.ndarray] = field(default_factory=list)
    outcomes: list[np.ndarray] = field(default_factory=list)
    matches: list[int] = field(default_factory=list)
    train_sizes: list[int] = field(default_factory=list)


def split_holdout(draws: Sequence[Any], fraction: float = HOLDOUT_FRACTION) -> int:
    """Index where the holdout begins (last `fraction` of draws)."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    return int(round(len(draws) * (1 - fraction)))


def holdout_start_date(conn, draws: Sequence[db.Draw]) -> date:
    """Frozen holdout boundary date; computed once and stored so it never moves as data grows."""
    stored = db.get_meta(conn, "holdout_start_date")
    if stored:
        return date.fromisoformat(stored)
    d = draws[split_holdout(draws)].draw_date
    db.set_meta(conn, "holdout_start_date", d.isoformat())
    return d


def walk_forward(model_factory: Callable[[], BaseModel], draws: Sequence[Any], start: int, end: int,
                 refit_every: int = 1, seed: int = 0) -> WalkForwardResult:
    """Forecast draws[start:end], each using a model fitted only on strictly earlier draws."""
    if start < 1 or end > len(draws) or start >= end:
        raise ValueError(f"invalid range start={start} end={end} n={len(draws)}")
    X = to_indicator_matrix(draws)
    rng = np.random.default_rng(seed)
    model: Optional[BaseModel] = None
    res = WalkForwardResult(model_name=model_factory().name)
    for i in range(start, end):
        if model is None or (i - start) % refit_every == 0:
            model = model_factory().fit(draws[:i])
            fitted_on = i
        assert fitted_on <= i, "data leakage: model fitted on target draw"
        p = model.predict_probabilities()
        suggestion = model.suggest_numbers(rng)
        res.indices.append(i)
        res.probabilities.append(p)
        res.outcomes.append(X[i])
        res.matches.append(int(X[i][np.asarray(suggestion) - 1].sum()))
        res.train_sizes.append(model.n_train)
    return res


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, level: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def calibration_table(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    """Quantile-binned reliability table: mean predicted vs observed inclusion frequency."""
    p, y = probs.ravel(), outcomes.ravel()
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return [{"p_mean": float(p.mean()), "obs": float(y.mean()), "n": int(len(p))}]
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    return [{"p_mean": float(p[bins == b].mean()), "obs": float(y[bins == b].mean()), "n": int((bins == b).sum())}
            for b in range(len(edges) - 1) if (bins == b).any()]


def score(res: WalkForwardResult) -> dict[str, Any]:
    """Brier, log loss, matches and paired comparison vs uniform baseline, with bootstrap CIs."""
    P, Y = np.array(res.probabilities), np.array(res.outcomes)
    brier = ((P - Y) ** 2).mean(axis=1)
    Pc = np.clip(P, 1e-9, 1 - 1e-9)
    logloss = -(Y * np.log(Pc) + (1 - Y) * np.log(1 - Pc)).mean(axis=1)
    matches = np.array(res.matches, dtype=float)
    delta = brier - UNIFORM_BRIER  # negative = better than baseline
    bl, bh = bootstrap_ci(brier)
    ml, mh = bootstrap_ci(matches, seed=1)
    dl, dh = bootstrap_ci(delta, seed=2)
    return {"n_forecasts": len(brier), "mean_brier": float(brier.mean()), "brier_ci_low": bl, "brier_ci_high": bh,
            "mean_log_loss": float(logloss.mean()), "mean_matches": float(matches.mean()),
            "matches_ci_low": ml, "matches_ci_high": mh, "baseline_brier": UNIFORM_BRIER,
            "delta_brier": float(delta.mean()), "delta_ci_low": dl, "delta_ci_high": dh,
            "calibration": calibration_table(P, Y)}


def run_backtest(conn, models: Optional[list[BaseModel]] = None, split: str = "development",
                 min_train: int = 200, refit_every: int = 5, log_markdown: bool = True,
                 max_forecasts: Optional[int] = None) -> list[dict[str, Any]]:
    """Run walk-forward evaluation for each model, persist all results (including failures)."""
    draws = db.get_draws(conn)
    if len(draws) < min_train + 20:
        raise RuntimeError(f"need at least {min_train + 20} draws, have {len(draws)}")
    h_date = holdout_start_date(conn, draws)
    h_idx = next(i for i, d in enumerate(draws) if d.draw_date >= h_date)
    if split == "development":
        start, end = min_train, h_idx
    elif split == "holdout":
        start, end = h_idx, len(draws)
    else:
        raise ValueError("split must be 'development' or 'holdout'")
    if max_forecasts:
        start = max(start, end - max_forecasts)
    run_id = uuid.uuid4().hex[:12]
    results = []
    for m in models or default_models():
        row: dict[str, Any] = {"run_id": run_id, "model_name": m.name, "params": m.params, "split": split}
        try:
            factory = lambda m=m: type(m)(**m.params)  # noqa: E731
            res = walk_forward(factory, draws, start, end, refit_every=refit_every)
            row.update(score(res), status="ok")
            logger.info("%s: brier=%.6f delta=%.2e [%.2e, %.2e] matches=%.3f", m.name, row["mean_brier"],
                        row["delta_brier"], row["delta_ci_low"], row["delta_ci_high"], row["mean_matches"])
        except Exception as exc:  # record failed experiments too
            logger.exception("Backtest failed for %s", m.name)
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        db.insert_backtest_result(conn, row)
        results.append(row)
    if log_markdown:
        append_experiment_log(results, split, draws[start].draw_date, draws[end - 1].draw_date, refit_every)
    return results


def append_experiment_log(results: list[dict[str, Any]], split: str, first: date, last: date,
                          refit_every: int, path: Path = EXPERIMENTS_MD) -> None:
    """Append a dated entry describing a backtest run to experiments.md."""
    lines = [f"\n### {date.today().isoformat()} - Walk-forward backtest ({split}) run `{results[0]['run_id']}`\n",
             f"- Forecast window: {first} to {last}; refit every {refit_every} draws; training = all prior draws",
             "- Brier delta < 0 means better than uniform; CI = 95% bootstrap\n",
             "| Model | Status | N | Mean Brier | Delta vs uniform [95% CI] | Mean matches [95% CI] |",
             "|---|---|---|---|---|---|"]
    for r in results:
        if r["status"] == "ok":
            lines.append(f"| {r['model_name']} | ok | {r['n_forecasts']} | {r['mean_brier']:.6f} | "
                         f"{r['delta_brier']:+.2e} [{r['delta_ci_low']:+.2e}, {r['delta_ci_high']:+.2e}] | "
                         f"{r['mean_matches']:.3f} [{r['matches_ci_low']:.3f}, {r['matches_ci_high']:.3f}] |")
        else:
            lines.append(f"| {r['model_name']} | FAILED | - | - | {r.get('error', '')} | - |")
    beats = [r["model_name"] for r in results if r["status"] == "ok" and r["model_name"] != "uniform_baseline"
             and r["delta_ci_high"] < 0]
    lines.append(f"\n- Conclusion: {'models with CI entirely below baseline: ' + ', '.join(beats) if beats else 'no model beat the uniform baseline with confidence.'}\n")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv: Optional[list[str]] = None) -> int:
    """CLI: python src/backtester.py [--holdout] [--refit-every N] [--max-forecasts N]."""
    ap = argparse.ArgumentParser(description="Walk-forward backtest")
    ap.add_argument("--holdout", action="store_true", help="evaluate on the frozen holdout set")
    ap.add_argument("--refit-every", type=int, default=5)
    ap.add_argument("--max-forecasts", type=int, default=None)
    ap.add_argument("--db", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_backtest(db.connect(args.db), split="holdout" if args.holdout else "development",
                 refit_every=args.refit_every, max_forecasts=args.max_forecasts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
