# Indicators

Every indicator the engine knows about, grouped as in the catalogue
(`data/transforms.py::CATALOGUE`), with what it is, why it matters, and exactly what the
model sees. The **active set** is chosen with `MACRO_REGIME_INDICATORS` (see README);
the default is marked ✔ below.

Two layers sit between the raw series and the HMM:

| Layer | Meaning | Example |
|---|---|---|
| **Headline** | the number a human reads and overrides | "Unemployment = 4.1%" |
| **Feature** | the stationary form the HMM is fitted on | 12-month change in the unemployment rate, pp |

Overrides are typed in headline units; the feature is recomputed against the actual history.

---

## Interest rates / credit

The higher interest rates are, the more they slow the economy. When bond yields rise,
money rotates from equities toward fixed income. Credit spreads price the market's fear
of default.

| Key | Indicator | FRED | Headline | Feature | Sign | Default |
|---|---|---|---|---|---|---|
| `yield_curve` | Yield curve | `DGS10`, `DGS2` | 10Y − 2Y spread, pp (month-end) | spread | + | ✔ |
| `fed_funds` | Fed Funds | `FEDFUNDS` | effective rate, % | 3-month change, pp | − | ✔ |
| `credit_spread` | Credit spread | `BAA10Y` | Moody's Baa − 10Y, pp | level | − | ✔ |

*Sign* is the direction that reads pro-growth: a steeper curve is bullish (+), hiking and
wider credit spreads are bearish (−). The sign only orders the states and drives the label
suggestion; the HMM itself is unsupervised.

## Inflation

| Key | Indicator | FRED | Headline | Feature | Sign | Default |
|---|---|---|---|---|---|---|
| `core_pce` | Core PCE | `PCEPILFE` | YoY % | YoY % | − | ✔ |
| `cpi` | CPI | `CPIAUCSL` | YoY % | YoY % | − | |
| `core_cpi` | Core CPI | `CPILFESL` | YoY % | YoY % | − | |
| `ppi` | PPI | `PPIACO` | MoM % | 3-month average MoM % | − | |

**CPI (year over year)** measures the change in prices consumers pay for a basket of goods and
services versus the same month a year earlier. High CPI growth erodes purchasing power and
usually leads to Fed hikes.

**Core CPI** excludes food and energy and is the better read on underlying inflation; the Fed
weighs it heavily. It is highly collinear with Core PCE, which is why only one of the two is
in the default set.

**PPI (month over month)** measures prices received by producers. It leads consumer inflation:
producers pass costs on. Month-over-month PPI is noisy, so the feature is a 3-month average.

## Labour market

| Key | Indicator | FRED | Headline | Feature | Sign | Default |
|---|---|---|---|---|---|---|
| `unemployment` | Unemployment rate | `UNRATE` | rate, % | 12-month change, pp | − | ✔ |
| `jobless_claims` | Initial jobless claims | `ICSA` (weekly) | monthly average, thousands | YoY % change | − | ✔ |
| `payrolls` | Nonfarm payrolls | `PAYEMS` | 3-month avg monthly change, thousands | same | + | |

**Unemployment rate**: the share of the labour force that is jobless and looking. The level
is slow and trending, so the model uses its 12-month change, which is what recession rules
(Sahm) key off.

**Initial jobless claims**: weekly first-time filings for unemployment benefits. A leading
indicator that turns before the monthly unemployment data. Weeks are averaged per month and
compared year over year.

**Nonfarm payrolls**: the monthly change in paid workers excluding farms, government,
households and non-profits. One of the most watched prints. Averaged over 3 months to
strip out revisions and noise. Left out of the default set because it is largely redundant
with claims and the unemployment change; test it with the selection tool.

## Activity / sentiment

| Key | Indicator | Source | Headline | Feature | Sign | Default |
|---|---|---|---|---|---|---|
| `ism_pmi` | ISM Manufacturing PMI | `NAPM` (FRED, to 2016) or manual CSV | level | level − 50 | + | ✔ |
| `consumer_sentiment` | Consumer sentiment | `UMCSENT` (U. Michigan) | level | 12-month change | + | |
| `ism_services_pmi` | ISM Services PMI | **manual CSV only** | level | level − 50 | + | |
| `chicago_pmi` | Chicago PMI | **manual CSV only** | level | level − 50 | + | |
| `consumer_confidence` | Consumer confidence | **manual CSV only** (Conference Board) | level | 12-month change | + | |

**ISM Manufacturing PMI**: survey of manufacturing purchasing managers on new orders,
production, employment, deliveries and inventories. Above 50 is expansion, below 50 is
contraction. Manufacturing turns before the wider economy. ISM stopped licensing it to FRED
in 2016; keep current prints in `cache/manual/NAPM.csv`.

**ISM Services (Non-Manufacturing) PMI**: the same survey for services, which are roughly
80% of US GDP, so it often moves markets more than the manufacturing print. Not on FRED.

**Chicago PMI**: the MNI Chicago Business Barometer, released before the national ISM data, an
early read on manufacturing conditions. Not on FRED.

**Consumer confidence**: how optimistic consumers are about the economy and their own
finances; consumer spending is about 70% of GDP. The Conference Board index is not on FRED;
the University of Michigan sentiment index (`UMCSENT`) is, and is the FRED-available proxy.

### Manual CSV supplements

Any series id can be supplemented from `cache/manual/<SERIES_ID>.csv` with two columns,
`date,value`, one row per month (first-of-month dates are fine). Manual rows are merged over
the FRED pull and win on overlapping dates. For the three **manual only** indicators the CSV is
the sole source; activating one without its CSV fails at refresh with a message naming the
file it needs.

---

## Which ones to use

The default set is seven indicators: the four the model was designed around plus the credit
spread, the unemployment change and jobless claims. They were chosen for structure, one or
two per category, low collinearity, long FRED history, not by measurement on real data,
which this repository's author could not fetch when the set was chosen.

Before widening (or narrowing) the set, run the ablation on your real data:

```python
mr = MacroRegime()
mr.evaluate_indicators()                                   # every catalogue entry vs the active set
mr.evaluate_indicators(candidates=["payrolls", "cpi"])      # or a shortlist
```

It reports, per candidate and out of sample: how strongly the candidate differs across the
regimes the base model already finds (`relevance_f`), how much of it the base features
already explain (`redundancy_r2`), whether the reads get sharper (`confidence`) or flip more
often (`switches_per_year`), and a `keep` recommendation. Then set

```bash
export MACRO_REGIME_INDICATORS=yield_curve,core_pce,fed_funds,ism_pmi,credit_spread,unemployment
```

and refit. Keep the count modest: with full covariances and about 400 monthly rows, more than
eight or so features starts to overfit.
