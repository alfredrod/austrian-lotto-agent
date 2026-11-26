# Experiment Log

Every experiment is recorded here, **including failed ones**. Backtest runs append entries automatically
(`python src/interface.py backtest --run`); all numbers are also stored in the `backtest_results` table.

## Entry template

```
### YYYY-MM-DD - <title>
- Hypothesis:
- Data window / split (development or holdout):
- Models & hyper-parameters:
- Result (Brier delta vs uniform with 95% CI, mean matches with 95% CI):
- Conclusion / decision:
```

## Entries

### 2026-11-27 - Initial baseline experiment
- Hypothesis: none of the frequency-based models will beat a fair uniform lottery (null hypothesis H0: draws are i.i.d. uniform).
- Data: 3,707 validated draws (1986-09-07 to 2026-11-25) from win2day CSVs; 105 exact cross-file duplicates removed, 0 conflicts.
- Split: development = draws 200 .. before 2020-08-09; holdout (last 20%, from 2020-08-09) untouched.
- Models: uniform_baseline; bayesian_frequency (alpha=100); recency_weighted (half_life=100, prior=20); ridge_historical (ridge_alpha=1000, 1000 draws window). Hyper-parameters fixed a priori, not tuned.
- Result: see automatically generated table below.
- Conclusion: no model improves on the uniform baseline; the Bayesian and recency models are measurably *worse* (their CI lies above 0), consistent with random draws. Suggestions default to the uniform baseline.

### 2026-11-26 - Walk-forward backtest (development) run `bf8b3dfe93a9`

- Forecast window: 1990-07-15 to 2020-08-05; refit every 5 draws; training = all prior draws
- Brier delta < 0 means better than uniform; CI = 95% bootstrap

| Model | Status | N | Mean Brier | Delta vs uniform [95% CI] | Mean matches [95% CI] |
|---|---|---|---|---|---|
| uniform_baseline | ok | 2766 | 0.115556 | +1.11e-17 [+1.08e-17, +1.15e-17] | 0.814 [0.785, 0.843] |
| bayesian_frequency | ok | 2766 | 0.115583 | +2.73e-05 [+5.29e-06, +4.74e-05] | 0.799 [0.770, 0.827] |
| recency_weighted | ok | 2766 | 0.115663 | +1.07e-04 [+7.06e-05, +1.41e-04] | 0.797 [0.769, 0.827] |
| ridge_historical | ok | 2766 | 0.115558 | +2.55e-06 [-1.16e-06, +6.42e-06] | 0.795 [0.766, 0.824] |

- Conclusion: no model beat the uniform baseline with confidence.

