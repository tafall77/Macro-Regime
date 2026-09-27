# Macro Regime Engine

A Gaussian hidden-Markov-model regime detector over four FRED macro series, with
scenario overrides ("what if PMI prints 47?") scored against the same fitted model,
and a Dash dashboard showing **actual vs. scenario** regime confidence side by side.

```
FRED ──► data/fred_fetcher.py ──► Parquet cache ──► model/hmm_engine.fit()   (monthly, actual data only)
                 │                                          │
                 └──► scoring/live_score.py ◄── data/overrides.py            (cheap, on every refresh/override)
                                 │
                                 └──► dashboard/app.py
```

## Layout

| Path | What it does |
|---|---|
| `data/transforms.py` | Indicator registry: FRED series ids, headline transform, stationary feature transform. |
| `data/fred_fetcher.py` | Pulls the series from the FRED API, builds the monthly headline panel, caches to Parquet, exposes `get_actual(indicator, date)`, `refresh()` and `training_data()`. |
| `data/overrides.py` | `{indicator: value}` store (in-memory, optional JSON session file): `set_override`, `clear_override`, `clear_all`, `get_effective(indicator, date)`. |
| `model/training_data.py` | The `TrainingData` wrapper — the only thing `fit()` accepts. |
| `model/hmm_engine.py` | `HMMEngine.fit()`, `.score()` (forward filter from the last fitted posterior), `.persistence(state)`. |
| `model/labeler.py` | Human-in-the-loop state → `bullish / transition / bearish` mapping, keyed by `fit_id`. CLI included. |
| `scoring/live_score.py` | Orchestrates one cheap scoring pass: actual + scenario probabilities, persistence, labels. |
| `dashboard/app.py` | Dash app: 4 indicator panels, two gauges, delta, persistence, distribution, legend. |
| `scheduler.py` | Monthly refit job (`--once`, `--daemon`, or `--cron` for a crontab line). Separate from the live path. |
| `tests/` | 63 tests against a fake FRED API with a synthetic 3-regime history. |

## Indicators and transforms

| Indicator | FRED series | Headline (what you read / override) | Feature (what the HMM sees) |
|---|---|---|---|
| Yield curve | `DGS10`, `DGS2` | 10Y − 2Y spread, pp (month-end) | spread |
| Core PCE | `PCEPILFE` | YoY % change | YoY % change |
| Fed Funds | `FEDFUNDS` | effective rate, % | 3-month change, pp (`MACRO_REGIME_FF_ROC_MONTHS`) |
| ISM PMI | `NAPM` (see below) | PMI level | level − 50 |

Overrides are entered in **headline** units. The feature transform is re-applied on
top of the actual history, so an overridden Fed Funds rate becomes the right
3-month change and the scenario can never be internally inconsistent.

**ISM PMI on FRED.** ISM stopped licensing the PMI to FRED in 2016, so `NAPM` ends
there. Either point `MACRO_REGIME_PMI_SERIES` at a proxy series, or keep the real
prints yourself in `cache/manual/NAPM.csv` (`date,value`, first-of-month dates).
Manual rows are merged on top of the FRED pull and win on overlapping dates; if
FRED rejects the series id altogether the manual file is used alone. Any
indicator more than `MACRO_REGIME_STALE_LIMIT_MONTHS` (default 3) behind the
latest month raises a clear `StaleDataError` instead of silently scoring old data.

## Setup

```bash
pip install -r requirements.txt
export FRED_API_KEY=...            # free key: https://fred.stlouisfed.org/docs/api/api_key.html

python scheduler.py --once         # 1. pull FRED, fit the HMM, save to cache/model/
python -m model.labeler show       # 2. look at the state means …
python -m model.labeler accept     #    … and confirm the labels (or: set 0=bearish 1=transition 2=bullish)
python -m dashboard.app            # 3. http://127.0.0.1:8050
```

Monthly refit: `python scheduler.py --cron` prints a crontab line (2nd of the month,
06:00 UTC by default), or run `python scheduler.py --daemon`. The dashboard picks up
a new model file automatically; labels for a new `fit_id` show as *provisional*
until you confirm them again.

Useful environment variables: `MACRO_REGIME_CACHE_DIR`, `MACRO_REGIME_N_STATES`
(default 3), `MACRO_REGIME_TRAINING_START` (default 1990-01-01),
`MACRO_REGIME_TRAINING_WINDOW_YEARS` (default: expanding window),
`MACRO_REGIME_HOST` / `MACRO_REGIME_PORT`.

## How scoring works

* `fit()` standardises the feature matrix, runs EM (`hmmlearn`, full covariance,
  several random restarts), then re-orders the states by a growth score (bullish-signed
  sum of standardised state means) so state 0 is the most bearish-looking and the
  last state the most bullish-looking. It also stores the filtered posterior for
  every training month.
* `score(rows)` runs the forward recursion from the posterior at the month before the
  first row (or from the last fitted posterior for undated input) through the given
  rows and returns the state distribution after the last one. The actual run and the
  scenario run step through the same rows; only the latest month differs.
* `persistence(state)` is `1 / (1 − p_ii)` months from the transition-matrix diagonal.

## The override can never reach the refit

The risk isn't a logic error, it's silently contaminating the walk-forward refit with
a hypothetical value. Three layers guard against it:

1. **Type barrier.** `HMMEngine.fit()` only accepts a `TrainingData` carrying an
   allowed provenance. Bare DataFrames raise `ProvenanceError`. The only production
   factory is `FredFetcher.training_data()`, which reads the Parquet cache and has no
   access to the override store; `data/overrides.py` cannot produce a frame at all.
2. **Runtime assertion.** `scheduler.assert_actual_only()` re-derives the feature
   matrix from the cache and requires the training frame to match it exactly. Given
   the override store, it additionally checks that no overridden indicator's
   hypothetical value appears in the final training row.
3. **Tests.** `tests/test_no_override_leak.py` sets wild overrides, runs the refit with
   a spy on the store, forges a scenario `TrainingData` and checks it is rejected.

## Known simplifications

* Features are aligned by reference month, not release date, so the training
  history has the usual small look-ahead for lagging series; live scoring uses the
  last known value of each indicator (carried forward within the stale limit).
* The dashboard is single-user: overrides live in one server-side store persisted to
  `cache/overrides.json`.
