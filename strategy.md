# Strategy

> **Lottery numbers cannot be reliably predicted.** Lotto 6 aus 45 is designed so every combination has the
> same chance (1 in 8,145,060 for the jackpot). This document tracks an honest statistical evaluation.

## Active approach
- Default suggestion model: **`uniform_baseline`** (truly random 6 of 45).
- A non-uniform model is used for `suggest` only if its walk-forward Brier score beats the uniform baseline
  with a 95% bootstrap CI entirely below zero on the development split. As of the last review, none does.

## Current data state
<!-- DATA_STATE_START -->
- Draws stored: 3708
- Latest draw: 2026-11-27 [12, 14, 19, 23, 31, 34] + Zz 1
- Holdout start date (frozen): 2020-08-09
- Live forecasts scored: 0
<!-- DATA_STATE_END -->

(The block above is rewritten automatically by `src/scheduler.py`.)

## Evaluation protocol
1. Walk-forward, chronological: the forecast for draw *i* uses a model fitted only on draws before *i*
   (refit every 5 draws in backtests, every draw live). No information from the target draw is used.
2. Metrics: per-number Brier score of inclusion probabilities, binary log loss, calibration table,
   matches per suggested line (fair expectation 6*6/45 = 0.8).
3. Each model is compared to the uniform baseline by the paired per-draw Brier difference with a
   95% bootstrap CI.
4. Holdout: the last 20% of draws (from 2020-08-09, frozen in the database) is never used for model
   development; it is evaluated once a change is final (`backtest --run --holdout`).
5. Live forecasts are timestamped, immutable (enforced by SQLite triggers) and scored exactly once.

## Evidence summary
Development backtest (2026-11-27, ~2,700 forecasts, see experiments.md):
- bayesian_frequency: Brier delta +2.7e-05 [CI +5.3e-06, +4.7e-05] -> worse than uniform
- recency_weighted: Brier delta +1.1e-04 [CI +7.1e-05, +1.4e-04] -> worse than uniform
- ridge_historical: Brier delta +2.6e-06 [CI -1.2e-06, +6.4e-06] -> indistinguishable from uniform
- Mean matches per line for all models ~0.80, the fair-lottery expectation.

## Limitations
- The draws are physically randomised; any apparent pattern in history is expected noise.
- No model can increase the expected number of matches above 0.8 per line for a fair draw.
- At best, a non-uniform suggestion might pick less popular combinations (affects prize *sharing*, not
  winning odds) - this is not modelled.
- Data quality depends on win2day CSVs; parse issues are logged, never silently fixed.

## Next planned review
2027-02-27 (quarterly), or earlier if the data pipeline reports anomalies.
