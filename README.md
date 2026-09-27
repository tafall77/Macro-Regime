# Macro Regime Engine

A hidden-Markov-model regime detector for US macro conditions, built for one question:
**"Given today's data, which regime are we in, and what happens to that read if one
number comes in differently?"**

It pulls macro series from FRED (rates and credit, inflation, labour market, activity and
sentiment; see [docs/indicators.md](docs/indicators.md) for the full catalogue), turns each into
a stationary feature, fits a Gaussian HMM with 2–3 hidden states, and scores the latest month
twice: once on the **actual** data and once with your **scenario overrides** layered on top.
Both reads come from the same fitted model, so the difference between them is purely the
effect of your hypothetical. It ships with a notebook façade that renders an HTML report inline,
a Dash dashboard, a monthly refit job, a walk-forward backtest, an indicator-selection tool,
and a demo mode that needs no API key.

```
FRED ──► data/fred_fetcher.py ──► Parquet cache ──► model/hmm_engine.fit()      monthly, ACTUAL data only
                 │                                        │
                 └──► scoring/live_score.py ◄── data/overrides.py                 cheap, on every refresh / override
                                 │
                        ┌────────┴─────────┐
                 reporting/report.py   dashboard/app.py
                 (HTML, notebook)      (Dash, browser)
```

---

## Quick start

### In a Jupyter notebook (recommended first run)

```python
from regime import MacroRegime

mr = MacroRegime(demo=True)        # synthetic 3-regime history, no FRED key needed
mr.refit()                         # pull → fit → save   (real mode: pulls from FRED)
mr.states()                        # per-state feature means + suggested labels
mr.label()                         # accept the suggestion (or mr.label({0: "bearish", 1: "transition", 2: "bullish"}))
mr.override(ism_pmi=47, unemployment=4.8)
mr.summary()                       # actual vs scenario probabilities as a DataFrame
mr.evaluate_indicators()           # which other catalogue indicators would add signal (real data)
mr.report()                        # full HTML report, rendered inline in the notebook
mr.report().save("regime.html")    # standalone file (inline_js=True embeds plotly.js for offline viewing)
```

`notebooks/macro_regime_walkthrough.ipynb` walks through all of this, including the backtest.

### Real data

```bash
pip install -r requirements.txt
export FRED_API_KEY=...            # free: https://fred.stlouisfed.org/docs/api/api_key.html

python scheduler.py --once         # 1. pull FRED, fit, save to cache/model/
python -m model.labeler show       # 2. inspect the state means …
python -m model.labeler accept     #    … and confirm labels (or: set 0=bearish 1=transition 2=bullish)
python -m dashboard.app            # 3. http://127.0.0.1:8050
python scheduler.py --cron         # 4. crontab line for the monthly refit
```

In a notebook the same thing is `MacroRegime()` (reads `FRED_API_KEY`) or `MacroRegime(api_key="...")`.

---

## What you get

### The report (`mr.report()`)

One self-contained HTML block, interactive Plotly figures, renders inline in Jupyter
(classic notebook and JupyterLab) and saves as a standalone file:

* **Indicator cards** — one per active indicator: current FRED headline value and date, the
  model input it becomes, and a "⚠ Using override" badge when a scenario value is in effect.
* **Twin gauges** — Actual vs Scenario confidence in the current regime, with the delta in
  points highlighted between them.
* **Persistence** — expected remaining months in the top regime, for both runs.
* **Full state distribution** — all states side by side.
* **Regime history** — filtered state probabilities over the last N years (stacked area),
  including months after the last refit.
* **Indicator history** — small multiples of the active headline series.
* **State legend** — state index, label, feature means, persistence, and both probabilities.

### The dashboard (`python -m dashboard.app`)

Same content, live: an indicator panel per active indicator with override inputs, "Refresh from FRED"
(re-pulls actuals, keeps overrides) and "Clear overrides", a regime selector for the gauges,
and the history chart. Overrides persist in `cache/overrides.json` across restarts. The
dashboard reloads the model automatically when the scheduler writes a new one.

### The backtest (`mr.backtest()`)

`model/backtest.py` does an honest walk-forward: refit on data strictly before each cut,
filter forward through the next `refit_every` months, repeat. Returns out-of-sample posteriors,
regime shares, empirical run lengths, and (in demo mode, where the truth is known) accuracy.

### Demo mode

`MacroRegime(demo=True)` generates a synthetic history from a known 3-state Markov chain
(hikes/inverted curve/hot inflation/PMI 45 ↔ cuts/steep curve/cool inflation/PMI 56), covering every
catalogue series, and serves it through a fake FRED session into a separate `cache/demo/` directory
(the not-on-FRED indicators are written there as manual CSVs). Everything else is the real
code path. `mr.true_states` exposes the generating regimes so you can check recovery.

---

## Indicators

Fifteen indicators are in the catalogue; seven are active by default. Every one has a
**headline** (the number you read and override) and a **feature** (the stationary form the HMM
sees). Full descriptions, transforms and FRED ids are in [docs/indicators.md](docs/indicators.md).

| Category | Default set | Also available |
|---|---|---|
| Interest rates / credit | yield curve (10Y−2Y), Fed Funds (3m change), Baa credit spread | |
| Inflation | Core PCE YoY | CPI YoY, Core CPI YoY, PPI MoM (3m avg) |
| Labour market | unemployment rate (12m change), initial jobless claims (YoY) | nonfarm payrolls (3m avg change) |
| Activity / sentiment | ISM Manufacturing PMI (vs 50) | U. Michigan sentiment, ISM Services PMI\*, Chicago PMI\*, Conference Board confidence\* |

\* not published on FRED; supplied from `cache/manual/<SERIES>.csv`.

Pick the active set with `MACRO_REGIME_INDICATORS` (comma-separated keys) or
`MacroRegime(indicators=[...])`. Overrides only ever apply to the latest month, and an override
on a level (say the unemployment rate) is pushed through the feature transform on top of the
actual history, so a scenario can never be internally inconsistent.

**Choosing indicators.** The default set was chosen for structure (one or two per category,
low collinearity, long history), not by measurement on real data. `mr.evaluate_indicators()`
runs a walk-forward ablation of every candidate against the active set on *your* data and
reports relevance, redundancy, out-of-sample confidence, switch rate and a `keep` flag. Run it
once with a real key before widening the set; keep the count under about eight.

**ISM PMI on FRED.** ISM stopped licensing the PMI to FRED in 2016, so `NAPM` ends there.
Either point `MACRO_REGIME_PMI_SERIES` at a proxy series or keep the real prints in
`cache/manual/NAPM.csv` (`date,value`). Manual rows win on overlapping dates; if FRED rejects
the id the manual file is used alone. Any indicator more than `MACRO_REGIME_STALE_LIMIT_MONTHS`
(default 3) behind the latest month raises a clear `StaleDataError` rather than silently
scoring old data.

---

## How the model works

* **Fit** (`HMMEngine.fit`) standardises the feature matrix, runs EM (`hmmlearn`, full
  covariance, several random restarts, best log-likelihood kept), then re-orders the states by
  a *growth score* — the bullish-signed sum of standardised state means — so state 0 is the most
  bearish-looking and the last state the most bullish-looking. That keeps indices stable across
  refits most of the time; the labeler still asks a human to confirm. The engine also stores the
  filtered posterior for every training month.
* **Score** (`HMMEngine.score`) runs the forward (filtering) recursion from the posterior at the
  month before the first row it is given — or from the last fitted posterior for undated input —
  and returns the state distribution after the last row. Re-scoring the final training month with
  an override therefore does not double-count it. The actual and scenario runs step through the
  same rows; only the latest month differs. No EM, no retraining: it's cheap enough to run on
  every keystroke.
* **Persistence** is `1 / (1 − p_ii)` months from the transition-matrix diagonal.
* **Labels** live in `model/labeler.py`, keyed by `fit_id`. Until you confirm labels for a new fit
  the report shows the heuristic suggestion marked *provisional*.

---

## The override can never reach the refit

The risk isn't a logic error; it's silently contaminating the walk-forward refit with a
hypothetical value. Three layers guard against it:

1. **Type barrier.** `HMMEngine.fit()` only accepts a `TrainingData` with an allowed provenance;
   a bare DataFrame raises `ProvenanceError`. The only production factory is
   `FredFetcher.training_data()`, which reads the Parquet cache and has no access to the override
   store. `data/overrides.py` cannot produce a feature frame at all.
2. **Runtime assertion.** `scheduler.assert_actual_only()` re-derives the feature matrix from the
   cache and requires the training frame to match it exactly; given the override store it also
   checks that no overridden indicator's hypothetical value appears in the last training row.
3. **Tests.** `tests/test_no_override_leak.py` sets wild overrides, runs the refit with a spy on
   the store, forges a scenario `TrainingData` and checks it is rejected.

---

## Layout

| Path | What it does |
|---|---|
| `regime.py` | `MacroRegime` — the notebook façade (`refresh / refit / label / override / summary / report / backtest`). |
| `data/transforms.py` | Indicator catalogue (15 entries): FRED ids, headline and feature transforms, category, bullish sign; active set from config. |
| `data/fred_fetcher.py` | FRED pull with pagination, Parquet cache keyed by date, manual CSV supplements, `get_actual`, `refresh`, `training_data`. |
| `data/overrides.py` | Override store (in-memory or JSON session file): `set_override`, `clear_override`, `clear_all`, `get_effective`. |
| `data/synthetic.py` | Synthetic 3-regime history and a fake FRED session (demo mode + tests). |
| `model/training_data.py` | The `TrainingData` wrapper `fit()` requires. |
| `model/hmm_engine.py` | `HMMEngine`: `fit`, `score`, `score_path`, `persistence`, `describe`, `save/load`. |
| `model/labeler.py` | State labels keyed by `fit_id`; `python -m model.labeler show / accept / set`. |
| `model/backtest.py` | Walk-forward out-of-sample evaluation. |
| `model/selection.py` | Indicator ablation: what each candidate adds to the active set, out of sample. |
| `scoring/live_score.py` | One cheap call: actual + scenario probabilities, persistence, labels, per-indicator snapshots. |
| `reporting/figures.py` | Plotly figure builders and the palette shared by report and dashboard. |
| `reporting/report.py` | Self-contained HTML report (`_repr_html_` for Jupyter, `.save()` for a file). |
| `dashboard/app.py` | Dash app. |
| `scheduler.py` | Monthly refit (`--once`, `--daemon`, `--cron`), separate from the live path. |
| `notebooks/macro_regime_walkthrough.ipynb` | End-to-end notebook. |
| `docs/indicators.md` | Every indicator: what it is, why it matters, what the model sees. |
| `tests/` | 78 tests on the fake FRED API. |

Local state lives under `cache/` (git-ignored): `fred/` (raw + panel Parquet), `model/`
(pickled engine, metadata JSON, labels), `overrides.json`, `manual/`, and `demo/` for demo mode.

## Configuration

All via environment variables; see `config.py` for the full list.

| Variable | Default | Meaning |
|---|---|---|
| `FRED_API_KEY` | — | required for real data |
| `MACRO_REGIME_CACHE_DIR` | `./cache` | where everything local lives |
| `MACRO_REGIME_INDICATORS` | 7 default keys | active indicator set, comma-separated catalogue keys |
| `MACRO_REGIME_N_STATES` | `3` | hidden states (2 or 3 recommended) |
| `MACRO_REGIME_TRAINING_START` | `1990-01-01` | first training month |
| `MACRO_REGIME_TRAINING_WINDOW_YEARS` | unset (expanding) | rolling window for the refit |
| `MACRO_REGIME_PMI_SERIES` | `NAPM` | FRED id for the PMI |
| `MACRO_REGIME_FF_ROC_MONTHS` | `3` | Fed Funds rate-of-change horizon |
| `MACRO_REGIME_STALE_LIMIT_MONTHS` | `3` | how far a lagging series may be carried forward |
| `MACRO_REGIME_REFIT_DAY` / `_REFIT_HOUR_UTC` | `2` / `6` | scheduler slot |
| `MACRO_REGIME_HOST` / `MACRO_REGIME_PORT` | `127.0.0.1` / `8050` | dashboard |

## Tests

```bash
python -m pytest -q        # ~10 s; no network, no FRED key
```

## Known simplifications

* Features are aligned by reference month, not release date, so the training history has the
  usual small look-ahead for lagging series. Live scoring uses the last known value of each
  indicator, carried forward within the stale limit.
* Gaps in the training history (missing months) are filtered as if consecutive.
* Probabilities on the synthetic demo data are sharper than you will see on real data, whose
  regimes overlap far more.
* The dashboard is single-user: one server-side override store, persisted to `cache/overrides.json`.
