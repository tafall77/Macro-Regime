# Sector & Commodity Seasonality

A Jupyter notebook for researching and backtesting quarterly seasonality in S&P 500 sectors and
commodities.

It answers three questions:

1. **What moved recently?** Trailing returns for the 11 S&P 500 sectors and 12 commodities, as sorted
   horizontal bar charts.
2. **What has the coming quarter looked like before?** For the next calendar quarter, each name's average
   return in that same quarter over the last 5 years. In late September that is the average of the last
   five Q4s.
3. **Is that worth trading?** A walk-forward backtest that ranks names on that 5-year seasonal average at
   the start of every quarter and checks whether the top-ranked names beat the bottom-ranked ones.

Prices come from Yahoo Finance through `yfinance`, so no account or API key is needed.

## Run it

1. Download the repository (green **Code** button, then **Download ZIP**) and extract it.
2. Open a terminal in the extracted folder and install the requirements. Python 3.10 or newer is needed.

   ```bash
   pip install -r requirements.txt
   ```

3. Start Jupyter and open the notebook.

   ```bash
   jupyter notebook Seasonality.ipynb
   ```

4. Run all cells (**Run → Run All Cells**). The first cell holds the settings.

The notebook downloads prices each time it runs and saves nothing to disk. The last section can write
one HTML copy of every chart if you set `SAVE_HTML = True`. It overwrites the same file each time.

## What the notebook shows

| Section | Chart | What each bar is |
|---|---|---|
| 1. Recent returns | S&P Sector Returns | trailing return over `PERIOD` (default 1 week) |
| 1. Recent returns | Commodity Returns | same, for commodities |
| 2. Seasonality | S&P Sector Returns · Qn Seasonality | average return in the upcoming quarter over the last 5 years |
| 2. Seasonality | Commodity Returns · Qn Seasonality | same, for commodities |
| 3. Backtest | Seasonal Backtest | growth of $1 for the top 3, bottom 3 and equal-weight portfolios |
| 3. Backtest | Rank IC by Quarter | how well the ranking predicted each calendar quarter |

Hover any bar for the details behind it: the dates of the return window, or each year's return
behind a seasonal average. Every chart has a table under it with the same numbers.

## Settings

All in the notebook's first cell.

| Setting | Default | Meaning |
|---|---|---|
| `PERIOD` | `"1W"` | recent-returns window: `1D`, `1W`, `2W`, `1M`, `3M`, `6M`, `1Y`, `MTD`, `QTD`, `YTD` |
| `LOOKBACK_YEARS` | `5` | years averaged for the seasonal charts and the backtest signal |
| `TOP_N` | `3` | names held in the top and bottom backtest portfolios |
| `START` | `"2000-01-01"` | first date of price history to download |
| `TARGET_QUARTER` | `None` | `None` for the next quarter, or `1` to `4` to study any quarter |
| `SECTORS` | SPDR sector ETFs | name to Yahoo ticker; add or remove freely |
| `COMMODITIES` | front-month futures | name to Yahoo ticker; `dict(sz.COMMODITY_ETFS)` switches to ETFs |

## How the numbers are calculated

- **Recent return.** Latest close divided by the close on or before the start of the window, minus one.
  A name whose data is more than 7 days stale shows as blank instead of an old number.
- **Quarterly return.** Close on the last trading day of the quarter over the close on the last trading
  day of the previous quarter. The current quarter only counts once its last trading day is in the data.
- **Seasonal average.** The simple mean of the last 5 completed returns for that calendar quarter. The
  table also shows the median, the best and worst year, and how many of the 5 were positive.
- **Backtest.** At the start of each quarter, every name gets the seasonal average computed only from
  earlier years, so there is no look-ahead. The top `TOP_N` and bottom `TOP_N` names are held for the
  quarter, equally weighted, and compared with an equal-weight portfolio of all ranked names. Results
  include annualised return, volatility, Sharpe ratio, win rate, maximum drawdown and the t-statistic
  of the average quarterly return.
- **Rank IC.** Each quarter, the rank correlation between the seasonal averages and the realised returns.
  Around 0 means no predictive power. A mean of about 0.05 or more with a t-statistic above 2 is a
  meaningful effect.
- **Sign hit rate.** How often the seasonal average had the same sign as the realised return, shown next
  to the base rate of positive quarters so you can tell skill from a rising market.

## Caveats

- **Sectors are ETFs.** The SPDR sector ETFs track the S&P 500 sector indices closely but include fees.
  Returns include dividends. Real Estate (XLRE) starts in 2015 and Communication Services (XLC) in 2018,
  so they join the backtest once they have 5 years of history.
- **Commodity futures jump at contract rolls.** Yahoo's continuous front-month series splice contracts
  without adjusting, which adds noise, most of all for natural gas and crude oil. The ETF universe
  includes the cost of rolling and is closer to what an investor could hold.
- **Small samples.** Each seasonal average rests on five numbers, and the backtest has roughly 85
  quarters. One unusual year can dominate an average.
- **No costs.** The backtest ignores commissions, spreads, slippage and taxes.

## Files

| File | Purpose |
|---|---|
| `Seasonality.ipynb` | the notebook |
| `seasonality.py` | data loading, return calculations, backtest and chart functions used by the notebook |
| `tests/test_seasonality.py` | tests on synthetic prices with known answers; run with `python -m pytest` |
| `requirements.txt` | Python packages |

The earlier macro regime engine is preserved in the git tag `archive/macro-regime-engine`.
