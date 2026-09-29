"""Sector and commodity returns, next-quarter seasonality, and a seasonal backtest.

Everything the notebook uses lives here:

* ``load_prices``        daily closes from Yahoo Finance (via ``yfinance``), one column per name
* ``period_returns``     trailing returns (1W, 1M, QTD, YTD, ...) for the bar charts
* ``quarterly_returns``  completed calendar-quarter returns
* ``next_quarter``, ``seasonal_table``  average return of a quarter over the last N years
* ``backtest``           walk-forward test of that seasonal signal, with no look-ahead
* ``bar_chart``, ``growth_chart``, ``save_html``  Plotly figures

Nothing is written to disk unless you call ``save_html``.
"""
from __future__ import annotations

import html
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# --------------------------------------------------------------------------- universes

#: S&P 500 sectors, via the SPDR Select Sector ETFs.
SECTORS: dict[str, str] = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

#: Commodities, via Yahoo's continuous front-month futures.
COMMODITIES: dict[str, str] = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Platinum": "PL=F",
    "Copper": "HG=F",
    "WTI Crude Oil": "CL=F",
    "Brent Crude Oil": "BZ=F",
    "Natural Gas": "NG=F",
    "Corn": "ZC=F",
    "Wheat": "ZW=F",
    "Soybeans": "ZS=F",
    "Coffee": "KC=F",
    "Sugar": "SB=F",
}

#: Tradable alternative to ``COMMODITIES``: ETFs, which include roll costs.
COMMODITY_ETFS: dict[str, str] = {
    "Gold": "GLD",
    "Silver": "SLV",
    "Platinum": "PPLT",
    "Copper": "CPER",
    "WTI Crude Oil": "USO",
    "Brent Crude Oil": "BNO",
    "Natural Gas": "UNG",
    "Corn": "CORN",
    "Wheat": "WEAT",
    "Soybeans": "SOYB",
    "Sugar": "CANE",
    "Agriculture (broad)": "DBA",
}

# --------------------------------------------------------------------------- data


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Pull the close column per ticker out of a ``yfinance.download`` frame."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=tickers, dtype="float64")
    if isinstance(raw.columns, pd.MultiIndex):
        top = raw.columns.get_level_values(0)
        if "Close" in top or "Adj Close" in top:  # group_by="column": (field, ticker)
            field = "Close" if "Close" in top else "Adj Close"
            close = raw[field]
        else:  # group_by="ticker": (ticker, field)
            field = "Close" if "Close" in raw.columns.get_level_values(1) else "Adj Close"
            close = raw.xs(field, axis=1, level=1)
    else:
        field = "Close" if "Close" in raw.columns else "Adj Close"
        close = raw[[field]].rename(columns={field: tickers[0]})
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close = close.reindex(columns=tickers).astype("float64")
    idx = pd.to_datetime(close.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    close.index = idx.normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close.index.name = "date"
    return close


def load_prices(
    universe: Mapping[str, str],
    start: str = "2000-01-01",
    end: str | None = None,
    total_return: bool = True,
) -> pd.DataFrame:
    """Daily closing prices from Yahoo Finance, one column per name in ``universe``.

    ``total_return=True`` uses dividend- and split-adjusted closes, so ETF returns
    include dividends. Futures have no dividends, so it makes no difference there.
    Tickers that return no data are dropped with a warning.
    """
    import yfinance as yf  # imported here so the rest of the module works without it

    tickers = list(dict.fromkeys(universe.values()))
    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=total_return,
        progress=False, group_by="column", threads=True,
    )
    close = _extract_close(raw, tickers)
    by_ticker = {t: n for n, t in universe.items()}
    close = close.rename(columns=by_ticker)
    names = [n for n in universe if n in close.columns]
    close = close[names]
    empty = [c for c in close.columns if close[c].dropna().empty]
    if len(empty) == len(names):
        raise RuntimeError(
            "Yahoo Finance returned no data for any ticker. Check your internet connection, "
            "wait a minute if you were rate-limited, or upgrade yfinance (pip install -U yfinance)."
        )
    if empty:
        warnings.warn(f"no data for {empty}; dropped", stacklevel=2)
        close = close.drop(columns=empty)
    return close.dropna(how="all")


# --------------------------------------------------------------------------- returns

PERIOD_LABELS = {
    "1D": "1 day", "1W": "1 week", "2W": "2 weeks", "1M": "1 month", "3M": "3 months",
    "6M": "6 months", "1Y": "1 year", "3Y": "3 years", "5Y": "5 years",
    "MTD": "Month to date", "QTD": "Quarter to date", "YTD": "Year to date",
}


def anchor_date(end, period: str, index: pd.DatetimeIndex | None = None) -> pd.Timestamp:
    """The date whose close a trailing ``period`` return is measured from."""
    end = pd.Timestamp(end).normalize()
    p = period.upper()
    if p == "1D":
        if index is None:
            raise ValueError("period '1D' needs the trading-day index")
        prior = index[index < end]
        if len(prior) == 0:
            raise ValueError("no trading day before the end date")
        return pd.Timestamp(prior[-1])
    if p == "MTD":
        return pd.Timestamp(end.year, end.month, 1) - pd.Timedelta(days=1)
    if p == "QTD":
        return pd.Timestamp(end.year, 3 * ((end.month - 1) // 3) + 1, 1) - pd.Timedelta(days=1)
    if p == "YTD":
        return pd.Timestamp(end.year, 1, 1) - pd.Timedelta(days=1)
    try:
        n, unit = int(p[:-1]), p[-1]
        offset = {"D": pd.DateOffset(days=n), "W": pd.DateOffset(weeks=n),
                  "M": pd.DateOffset(months=n), "Y": pd.DateOffset(years=n)}[unit]
    except (ValueError, KeyError):
        raise ValueError(f"unknown period {period!r}; use e.g. 1W, 1M, 3M, 1Y, MTD, QTD, YTD") from None
    return end - offset


def period_returns(prices: pd.DataFrame, period: str = "1W", end=None, max_stale_days: int = 7) -> pd.DataFrame:
    """Trailing return of each column over ``period`` ending at ``end`` (default: latest date).

    Returns one row per name with the return and the two closes it was measured
    between. A name whose latest close is more than ``max_stale_days`` older than
    ``end`` (or whose starting close is that far before the anchor) gets NaN.
    """
    px = prices.sort_index()
    end = pd.Timestamp(end).normalize() if end is not None else px.dropna(how="all").index[-1]
    anchor = anchor_date(end, period, px.index)
    rows = {}
    for col in px.columns:
        s = px[col].dropna().loc[:end]
        row = {"return": np.nan, "start_date": pd.NaT, "end_date": pd.NaT, "start_price": np.nan, "end_price": np.nan}
        start = s.loc[:anchor]
        if not s.empty and not start.empty:
            e_date, s_date = s.index[-1], start.index[-1]
            e_px, s_px = float(s.iloc[-1]), float(start.iloc[-1])
            fresh = (end - e_date).days <= max_stale_days and (anchor - s_date).days <= max_stale_days
            row.update(start_date=s_date, end_date=e_date, start_price=s_px, end_price=e_px)
            if fresh and s_px > 0 and e_px > 0:
                row["return"] = e_px / s_px - 1.0
        rows[col] = row
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "name"
    return out


def _quarter_alias() -> str:
    try:
        pd.date_range("2020-01-01", periods=2, freq="QE")
        return "QE"
    except ValueError:  # pandas < 2.2
        return "Q"


def quarterly_returns(prices: pd.DataFrame, complete_only: bool = True) -> pd.DataFrame:
    """Calendar-quarter returns (quarter-end close to quarter-end close), PeriodIndex.

    With ``complete_only`` the current quarter is dropped until the data reaches
    its last business day.
    """
    px = prices.sort_index()
    q_px = px.resample(_quarter_alias()).last()
    q_px = q_px.where(q_px > 0)
    q_ret = q_px / q_px.shift(1) - 1.0
    q_ret.index = q_ret.index.to_period("Q")
    if complete_only and len(q_ret):
        last_q = q_ret.index[-1]
        last_bday = pd.offsets.BDay().rollback(last_q.end_time.normalize())
        if px.dropna(how="all").index[-1] < last_bday:
            q_ret = q_ret.iloc[:-1]
    q_ret.index.name = "quarter"
    return q_ret


def next_quarter(as_of) -> pd.Period:
    """The calendar quarter after the one containing ``as_of``."""
    return pd.Timestamp(as_of).to_period("Q") + 1


def seasonal_table(q_ret: pd.DataFrame, quarter: int, years: int = 5, min_years: int | None = None) -> pd.DataFrame:
    """Each name's return in quarter ``quarter`` (1-4) over the last ``years`` completed years.

    Columns: one per year, then ``mean``, ``median``, ``positive`` (count),
    ``hit_rate``, ``best``, ``worst`` and ``n``. Names with fewer than
    ``min_years`` observations (default: all ``years``) get NaN statistics.
    Sorted by mean, highest first.
    """
    if quarter not in (1, 2, 3, 4):
        raise ValueError("quarter must be 1, 2, 3 or 4")
    min_years = years if min_years is None else min_years
    rows = q_ret[q_ret.index.quarter == quarter].tail(years)
    per_year = rows.T
    per_year.columns = [p.year for p in rows.index]
    n = rows.notna().sum()
    stats = pd.DataFrame({
        "mean": rows.mean(), "median": rows.median(), "positive": (rows > 0).sum(),
        "hit_rate": (rows > 0).sum() / n.replace(0, np.nan), "best": rows.max(), "worst": rows.min(), "n": n,
    })
    stats.loc[n < min_years, ["mean", "median", "hit_rate", "best", "worst"]] = np.nan
    out = pd.concat([per_year, stats], axis=1)
    out.index.name = "name"
    return out.sort_values("mean", ascending=False, na_position="last")


# --------------------------------------------------------------------------- backtest


def _contiguous(q_ret: pd.DataFrame) -> pd.DataFrame:
    full = pd.period_range(q_ret.index.min(), q_ret.index.max(), freq="Q")
    return q_ret.reindex(full)


def seasonal_signal(q_ret: pd.DataFrame, years: int = 5) -> pd.DataFrame:
    """For each quarter t: the mean of the same calendar quarter over the ``years`` years before t.

    Uses only quarters t-4, t-8, ..., t-4*years, so the signal for t never sees t.
    NaN unless all ``years`` observations exist.
    """
    q = _contiguous(q_ret)
    stack = np.stack([q.shift(4 * k).to_numpy(dtype="float64") for k in range(1, years + 1)])
    count = np.isfinite(stack).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(stack, axis=0)
    mean[count < years] = np.nan
    return pd.DataFrame(mean, index=q.index, columns=q.columns)


@dataclass
class BacktestResult:
    """Output of :func:`backtest`. ``returns`` holds one column per portfolio."""

    returns: pd.DataFrame
    holdings: pd.DataFrame
    ic: pd.Series
    signal: pd.DataFrame
    realized: pd.DataFrame
    top_n: int
    years: int

    @property
    def top(self) -> str:
        return f"Top {self.top_n}"

    @property
    def bottom(self) -> str:
        return f"Bottom {self.top_n}"

    def growth(self) -> pd.DataFrame:
        """Growth of $1 for the long-only portfolios, starting at 1 the quarter before the first trade."""
        cols = [self.top, "Equal weight", self.bottom]
        g = (1 + self.returns[cols]).cumprod()
        start = pd.DataFrame([[1.0] * len(cols)], columns=cols, index=[self.returns.index[0] - 1])
        return pd.concat([start, g])

    def stats(self) -> pd.DataFrame:
        """Per portfolio: annualised return, volatility, Sharpe (rf = 0), win rate, drawdown, t-stat."""
        rows = {}
        ew = self.returns["Equal weight"]
        for col in self.returns.columns:
            r = self.returns[col].dropna()
            n = len(r)
            growth = (1 + r).cumprod()
            sd = r.std(ddof=1)
            rows[col] = {
                "quarters": n,
                "avg_quarter": r.mean(),
                "ann_return": growth.iloc[-1] ** (4 / n) - 1 if n else np.nan,
                "ann_vol": sd * 2,
                "sharpe": r.mean() / sd * 2 if sd > 0 else np.nan,
                "t_stat": r.mean() / sd * math.sqrt(n) if sd > 0 else np.nan,
                "win_rate": (r > 0).mean(),
                "beats_equal_weight": (r > ew.loc[r.index]).mean() if col in (self.top, self.bottom) else np.nan,
                "max_drawdown": (growth / growth.cummax() - 1).min(),
                "best": r.max(),
                "worst": r.min(),
            }
        return pd.DataFrame.from_dict(rows, orient="index")

    def ic_summary(self) -> pd.Series:
        """Rank information coefficient: does a higher seasonal average mean a higher next-quarter return?"""
        ic = self.ic.dropna()
        sd = ic.std(ddof=1)
        return pd.Series({
            "quarters": len(ic), "mean_ic": ic.mean(), "median_ic": ic.median(),
            "t_stat": ic.mean() / sd * math.sqrt(len(ic)) if sd > 0 else np.nan,
            "share_positive": (ic > 0).mean(), "sign_hit_rate": self.sign_hit_rate(),
            "base_rate_positive": self.base_rate_positive(),
        })

    def ic_by_quarter(self) -> pd.Series:
        """Mean rank IC grouped by the quarter being predicted (Q1..Q4)."""
        ic = self.ic.dropna()
        out = ic.groupby(ic.index.quarter).mean()
        out.index = [f"Q{q}" for q in out.index]
        return out

    def base_rate_positive(self) -> float:
        """Share of name-quarters with a positive return: the hit rate of always guessing "up"."""
        r = self.realized.to_numpy()[np.isfinite(self.signal.to_numpy())]
        r = r[np.isfinite(r)]
        return float((r > 0).mean()) if r.size else float("nan")

    def sign_hit_rate(self) -> float:
        """Share of name-quarters where the seasonal average and the realised return had the same sign."""
        s, r = self.signal.to_numpy(), self.realized.to_numpy()
        mask = np.isfinite(s) & np.isfinite(r) & (s != 0) & (r != 0)
        return float((np.sign(s[mask]) == np.sign(r[mask])).mean()) if mask.any() else float("nan")


def backtest(q_ret: pd.DataFrame, years: int = 5, top_n: int = 3, min_names: int | None = None) -> BacktestResult:
    """Walk-forward test of the seasonal signal.

    At the start of every quarter t, rank the names by their average return in
    the same calendar quarter over the previous ``years`` years. Hold the top
    ``top_n`` and the bottom ``top_n`` equally weighted for quarter t and compare
    with an equal-weight portfolio of every ranked name. Quarters with fewer than
    ``min_names`` ranked names (default ``2 * top_n``) are skipped.
    """
    min_names = 2 * top_n if min_names is None else min_names
    q = _contiguous(q_ret)
    signal = seasonal_signal(q, years)
    top_col, bottom_col = f"Top {top_n}", f"Bottom {top_n}"
    rets, holds, ics = {}, {}, {}
    for t in q.index:
        s, r = signal.loc[t], q.loc[t]
        ok = s.notna() & r.notna()
        if ok.sum() < min_names:
            continue
        s, r = s[ok], r[ok]
        ranked = s.sort_values(ascending=False, kind="mergesort")
        top, bottom = list(ranked.index[:top_n]), list(ranked.index[-top_n:])
        top_r, bottom_r = r[top].mean(), r[bottom].mean()
        rets[t] = {top_col: top_r, "Equal weight": r.mean(), bottom_col: bottom_r, "Top − Bottom": top_r - bottom_r}
        holds[t] = {"top": ", ".join(top), "bottom": ", ".join(bottom), "names_ranked": int(ok.sum())}
        ics[t] = s.rank().corr(r.rank())
    if not rets:
        raise ValueError(f"no quarter had {min_names} names with {years} years of history; download from an earlier start date")
    returns = pd.DataFrame.from_dict(rets, orient="index")
    returns.index = pd.PeriodIndex(returns.index, freq="Q", name="quarter")
    holdings = pd.DataFrame.from_dict(holds, orient="index")
    holdings.index = returns.index
    ic = pd.Series(ics, name="rank_ic")
    ic.index = returns.index
    return BacktestResult(returns=returns, holdings=holdings, ic=ic, signal=signal.loc[returns.index],
                          realized=q.loc[returns.index], top_n=top_n, years=years)


# --------------------------------------------------------------------------- tables

_COUNT = {"n", "positive", "quarters", "names_ranked"}
_PLAIN = {"sharpe", "t_stat", "mean_ic", "median_ic", "start_price", "end_price", "rank_ic"} | _COUNT
_RATE = {"hit_rate", "win_rate", "beats_equal_weight", "share_positive", "sign_hit_rate", "base_rate_positive"}


def styled(df: pd.DataFrame | pd.Series):
    """Format a result table for display: returns as percentages, rates as whole percentages, dates as days."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    fmt = {}
    for c in df.columns:
        col = df[c]
        if pd.api.types.is_datetime64_any_dtype(col):
            fmt[c] = lambda d: d.strftime("%Y-%m-%d") if pd.notna(d) else "–"
        elif pd.api.types.is_float_dtype(col):
            fmt[c] = ("{:.0f}" if str(c) in _COUNT else "{:.2f}" if str(c) in _PLAIN
                      else "{:.0%}" if str(c) in _RATE else "{:+.2%}")
    out = df.style.format(fmt, na_rep="–")
    if all(str(i) in _PLAIN | _RATE for i in df.index):  # a one-column summary: format by row instead
        out = df.style.format(na_rep="–", formatter=lambda v: v)
        for i in df.index:
            spec = "{:.0f}" if str(i) in _COUNT else "{:.2f}" if str(i) in _PLAIN else "{:.0%}"
            out = out.format(spec, subset=pd.IndexSlice[[i], :], na_rep="–")
    return out


# --------------------------------------------------------------------------- charts

NAVY = "#14285a"
INK = "#1a1a1a"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
FONT = 'Arial, "Helvetica Neue", Helvetica, sans-serif'
LINE_COLORS = ("#2a78d6", "#1baf7a", "#eb6834")  # top, equal weight, bottom


def _nice_axis(vmin: float, vmax: float, max_ticks: int = 8, pad: float = 0.18) -> tuple[float, float, float]:
    """(low, high, step) for a value axis that includes 0 and leaves room for bar-end labels."""
    lo, hi = min(vmin, 0.0), max(vmax, 0.0)
    span = (hi - lo) or abs(hi) or abs(lo) or 0.01
    lo_p = lo - pad * span if lo < 0 else 0.0
    hi_p = hi + pad * span if hi > 0 else 0.0
    raw = (hi_p - lo_p) / max_ticks
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw - 1e-15)
    low = math.floor(lo_p / step + 1e-9) * step
    high = math.ceil(hi_p / step - 1e-9) * step
    return round(low, 12), round(high, 12), step


def _decimals(x: float) -> int:
    text = f"{x:.8f}".rstrip("0").rstrip(".")
    return len(text.split(".")[1]) if "." in text else 0


def bar_chart(
    values: pd.Series,
    title: str,
    subtitle: str | None = None,
    percent: bool = True,
    decimals: int = 1,
    hover: pd.Series | None = None,
    color: str = NAVY,
) -> go.Figure:
    """Horizontal bars sorted from highest to lowest, value labels at the bar ends."""
    v = values.dropna().sort_values(ascending=False)
    if v.empty:
        raise ValueError(f"{title}: nothing to plot (all values are NaN)")
    names = [str(n) for n in v.index]
    labels = [f"{x:.{decimals}%}" if percent else f"{x:.{decimals}f}" for x in v]
    if hover is None:
        custom = [f"<b>{html.escape(n)}</b><br>{lab}" for n, lab in zip(names, labels)]
    else:
        custom = [hover.get(k, f"<b>{html.escape(str(k))}</b>") for k in v.index]
    low, high, step = _nice_axis(float(v.min()), float(v.max()))
    ticks = np.round(np.arange(low, high + step / 2, step), 12)
    tickformat = f".{_decimals(step * 100)}%" if percent else f".{_decimals(step)}f"
    fig = go.Figure(go.Bar(
        x=v.to_numpy(), y=names, orientation="h", marker=dict(color=color, line=dict(width=0)),
        text=labels, textposition="outside", cliponaxis=False, constraintext="none",
        textfont=dict(color=INK, size=12), customdata=custom, hovertemplate="%{customdata}<extra></extra>",
    ))
    head = f"<b>{html.escape(title)}</b>"
    if subtitle:
        head += f"<br><span style='font-size:12px;color:{INK2}'>{html.escape(subtitle)}</span>"
    fig.update_layout(
        title=dict(text=head, x=0.01, xref="container", xanchor="left", y=1, yref="container", yanchor="top",
                   pad=dict(t=14), font=dict(size=18, color=INK, family=FONT)),
        height=max(260, 120 + 30 * len(names)), margin=dict(l=10, r=30, t=84 if subtitle else 60, b=36),
        paper_bgcolor="white", plot_bgcolor="white", font=dict(family=FONT, color=INK, size=13),
        bargap=0.45, showlegend=False, hoverlabel=dict(bgcolor="white", font=dict(color=INK, family=FONT)),
        xaxis=dict(range=[low, high], tickvals=ticks, tickformat=tickformat, showgrid=False, showline=False,
                   zeroline=True, zerolinecolor=INK, zerolinewidth=1, ticks="outside", ticklen=4,
                   tickcolor=MUTED, tickfont=dict(color=INK2, size=12), fixedrange=True),
        yaxis=dict(categoryorder="array", categoryarray=names[::-1], automargin=True, showgrid=False,
                   ticks="", tickfont=dict(color=INK, size=13), fixedrange=True),
    )
    return fig


def _fmt_date(d) -> str:
    return pd.Timestamp(d).strftime("%b %d, %Y") if pd.notna(d) else "n/a"


def returns_chart(prices: pd.DataFrame, universe: Mapping[str, str], period: str, title: str,
                  end=None) -> tuple[go.Figure, pd.DataFrame]:
    """Trailing ``period`` returns as a bar chart, plus the table behind it."""
    table = period_returns(prices, period, end=end)
    table.insert(0, "ticker", [universe.get(n, "") for n in table.index])
    ok = table.dropna(subset=["return"])
    start, stop = ok["start_date"].mode().iloc[0], ok["end_date"].mode().iloc[0]
    subtitle = f"{PERIOD_LABELS.get(period.upper(), period)} · {_fmt_date(start)} → {_fmt_date(stop)}"
    hover = pd.Series({
        n: (f"<b>{html.escape(n)}</b> ({row.ticker})<br>{row['return']:+.2%}<br>"
            f"{_fmt_date(row.start_date)} → {_fmt_date(row.end_date)}")
        for n, row in ok.iterrows()
    })
    fig = bar_chart(table["return"], title, subtitle, hover=hover)
    return fig, table.sort_values("return", ascending=False, na_position="last")


def seasonal_chart(prices: pd.DataFrame, universe: Mapping[str, str], target: pd.Period | int | None = None,
                   years: int = 5, title: str = "") -> tuple[go.Figure, pd.DataFrame]:
    """Average return of the target quarter over the last ``years`` years, as a bar chart plus table.

    ``target`` is a quarterly Period (e.g. ``next_quarter(...)``), a quarter number
    1-4, or None for the quarter after the latest price.
    """
    q_ret = quarterly_returns(prices)
    if target is None:
        target = next_quarter(prices.dropna(how="all").index[-1])
    quarter = target.quarter if isinstance(target, pd.Period) else int(target)
    table = seasonal_table(q_ret, quarter, years)
    year_cols = [c for c in table.columns if isinstance(c, (int, np.integer))]
    span = f"{year_cols[0]}–{year_cols[-1]}" if year_cols else "n/a"
    upcoming = f" · upcoming quarter: Q{quarter} {target.year}" if isinstance(target, pd.Period) else ""
    subtitle = f"Average Q{quarter} return over the last {len(year_cols)} years ({span}){upcoming}"
    hover = {}
    for n, row in table.iterrows():
        if pd.isna(row["mean"]):
            continue
        lines = "<br>".join(f"Q{quarter} {y}: {row[y]:+.1%}" for y in year_cols if pd.notna(row[y]))
        hover[n] = (f"<b>{html.escape(str(n))}</b> ({universe.get(n, '')})<br>Average {row['mean']:+.2%} · "
                    f"positive {int(row['positive'])} of {int(row['n'])}<br>{lines}")
    fig = bar_chart(table["mean"], f"{title} · Q{quarter} Seasonality" if title else f"Q{quarter} Seasonality",
                    subtitle, hover=pd.Series(hover, dtype=object))
    table.insert(0, "ticker", [universe.get(n, "") for n in table.index])
    return fig, table


def growth_chart(result: BacktestResult, title: str, subtitle: str | None = None) -> go.Figure:
    """Growth of $1 for the top, equal-weight and bottom portfolios of a backtest."""
    g = result.growth()
    x = g.index.to_timestamp(how="end").normalize()
    fig = go.Figure()
    for col, color in zip(g.columns, LINE_COLORS):
        fig.add_scatter(x=x, y=g[col], name=col, mode="lines", line=dict(color=color, width=2),
                        hovertemplate=f"{col}: $%{{y:.2f}}<extra></extra>")
        fig.add_annotation(x=x[-1], y=float(g[col].iloc[-1]), text=f" {col} ${g[col].iloc[-1]:.2f}", showarrow=False,
                           xanchor="left", font=dict(color=INK2, size=12))
    head = f"<b>{html.escape(title)}</b>"
    if subtitle:
        head += f"<br><span style='font-size:12px;color:{INK2}'>{html.escape(subtitle)}</span>"
    fig.update_layout(
        title=dict(text=head, x=0.01, xref="container", xanchor="left", y=1, yref="container", yanchor="top",
                   pad=dict(t=14), font=dict(size=18, color=INK, family=FONT)),
        height=440, margin=dict(l=10, r=150, t=84 if subtitle else 60, b=70), paper_bgcolor="white",
        plot_bgcolor="white", font=dict(family=FONT, color=INK, size=13), hovermode="x unified",
        legend=dict(orientation="h", y=-0.12, yanchor="top", x=0, font=dict(color=INK2)),
        xaxis=dict(range=[x[0], x[-1]], showgrid=False, tickfont=dict(color=INK2), hoverformat="%b %Y"),
        yaxis=dict(gridcolor=GRID, tickprefix="$", tickformat=".2f", tickfont=dict(color=INK2), zeroline=False),
    )
    return fig


def save_html(path: str | Path, figures: Iterable[go.Figure], title: str = "Sector & Commodity Seasonality") -> Path:
    """Write the given figures into one standalone HTML page (loads plotly.js from its CDN)."""
    path = Path(path)
    figures = list(figures)
    parts = [
        pio.to_html(f, full_html=False, include_plotlyjs="cdn" if i == 0 else False, default_width="100%",
                    config={"displayModeBar": False, "responsive": True})
        for i, f in enumerate(figures)
    ]
    body = "\n".join(f"<div class='fig'>{p}</div>" for p in parts)
    page = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(title)}</title>"
        "<style>body{margin:0;background:#f7f7f5;font-family:Arial,sans-serif}"
        ".wrap{max-width:1180px;margin:0 auto;padding:16px;display:grid;gap:16px;"
        "grid-template-columns:repeat(auto-fill,minmax(520px,1fr))}"
        ".fig{background:#fff;border:1px solid rgba(0,0,0,.08);border-radius:10px;padding:8px;min-width:0;overflow:hidden}"
        "@media(max-width:560px){.wrap{grid-template-columns:1fr}}</style></head>"
        f"<body><div class='wrap'>{body}</div>"
        "<script>window.addEventListener('load',()=>window.dispatchEvent(new Event('resize')))</script>"
        "</body></html>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path
