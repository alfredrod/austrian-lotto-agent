# Austrian Lotto Agent (6 aus 45)

> ⚠️ **Disclaimer:** Lotto draws are random. **Statistical models cannot reliably predict lottery outcomes.**
> Every combination has the same 1 in 8,145,060 jackpot chance. This project is a research/analysis tool for
> honest statistical evaluation - not a way to win. Please gamble responsibly.

A statistical analysis bot for Österreichische Lotterien **Lotto 6 aus 45**. It downloads the complete draw
history from [win2day](https://www.win2day.at/lotterie/lotto/lotto-statistik-zahlen-ergebnisse-download),
validates and stores it in SQLite, fits several probability models, evaluates them with leakage-free
walk-forward backtests against a uniform random baseline, and keeps immutable live forecasts.

## Components
| File | Purpose |
|---|---|
| `src/data_collector.py` | Discovers CSV links on win2day, parses all layouts (1986-2010 & 2010-2017 archives, yearly files), validates, deduplicates |
| `src/database.py` | SQLite schema/migrations (`draws`, `forecasts`, `model_versions`, `backtest_results`, `update_log`, `meta`) and CRUD helpers |
| `src/models.py` | `UniformRandomBaseline`, `BayesianFrequencyModel` (Dirichlet shrinkage), `RecencyWeightedModel` (exponential decay), `RegularizedHistoricalModel` (ridge) |
| `src/backtester.py` | Walk-forward evaluation, Brier/log-loss/calibration/matches, bootstrap CIs, frozen 20% holdout |
| `src/scheduler.py` | Update cycle (Europe/Vienna): fetch new draws, score forecasts, refit, forecast next draw |
| `src/interface.py` | CLI |
| `src/backfill.py` | Adds the missing combinations to forecasts saved before the multi-line feature (INSERT-only, so the immutability triggers stay intact) |
| `strategy.md` / `experiments.md` | Current strategy & evidence / dated experiment log |

## Setup
```bash
git clone https://github.com/alfredrod/austrian-lotto-agent
cd austrian-lotto-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or: pip install -e .[dev]
python src/interface.py init             # create data/lotto.db
python src/interface.py update --full    # download the complete history (1986-today)
```
The repository already ships a populated `data/lotto.db`; `update` only adds new draws.

## Usage
```bash
python src/interface.py status                 # latest draw, last update, pending forecasts
python src/interface.py suggest                # 6 numbers + which model produced them
python src/interface.py suggest --model bayesian_frequency
python src/interface.py suggest --combinations 10
python src/interface.py backtest               # latest results vs uniform baseline with 95% CIs
python src/interface.py backtest --run         # re-run development backtest (logs to experiments.md)
python src/interface.py backtest --run --holdout   # evaluate frozen holdout (use sparingly)
python src/interface.py strategy               # show strategy.md
python src/interface.py experiments            # show experiments.md
python src/interface.py update                 # fetch new draws, score & refresh forecasts
python src/interface.py backfill --dry-run     # preview the combinations old forecasts are missing
python src/interface.py backfill               # add them (idempotent, INSERT-only)
```
Scheduler: `python src/scheduler.py --check-new` (one cycle, cron-friendly) or `--daemon --interval 60`.
Example cron (server in Vienna time): `15 20 * * 0,3 cd /path/to/repo && .venv/bin/python src/scheduler.py --check-new`.

A GitHub Actions workflow (`deploy/github-workflows/update.yml`, see "GitHub Actions" below) runs the cycle daily at 20:00 Vienna time and commits
the updated database and markdown files.

## Results so far
On ~2,700 walk-forward development forecasts no model beat the uniform baseline; see `strategy.md`.

## Tests
```bash
pip install pytest && python -m pytest -q
```

## License
MIT

## GitHub Actions
The workflow files live in `deploy/github-workflows/` because the automation app lacked the `workflows` permission. Activate them with:
```bash
mkdir -p .github/workflows && git mv deploy/github-workflows/*.yml .github/workflows/ && git commit -m "Enable workflows" && git push
```
Then in **Settings -> Actions -> General** set *Workflow permissions* to **Read and write** so the update job can commit.
