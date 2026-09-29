"""Tests for seasonality.py on synthetic prices with known answers. No network."""
from __future__ import annotations

import sys
import types
import warnings

import numpy as np
import pandas as pd
import pytest

import seasonality as sz


# ----------------------------------------------------------------------- helpers

def bdays(start, end):
    return pd.bdate_range(start, end)


def seasonal_prices(names, start="2005-01-03", end="2025-12-31", seed=0, strength=0.04, noise=0.002):
    """Daily prices where each name has a fixed return pattern by calendar quarter."""
    rng = np.random.default_rng(seed)
    idx = bdays(start, end)
    quarters = idx.quarter.to_numpy()
    days_per_q = pd.Series(1, index=idx).groupby([idx.year, idx.quarter]).transform("size").to_numpy()
    cols = {}
    for i, name in enumerate(names):
        pattern = strength * np.array([np.sin(i + q) for q in range(4)])  # different season per name
        daily = pattern[quarters - 1] / days_per_q + rng.normal(0, noise, len(idx))
        cols[name] = 100 * np.exp(np.cumsum(daily))
    return pd.DataFrame(cols, index=idx)


# ----------------------------------------------------------------------- data loading

def fake_download_frame(tickers, idx, group_by="column"):
    fields = ["Close", "High", "Low", "Open", "Volume"]
    data = {}
    for f in fields:
        for j, t in enumerate(tickers):
            data[(f, t)] = np.arange(len(idx), dtype=float) + 10 * (j + 1)
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Price", "Ticker"])
    if group_by == "ticker":
        df = df.swaplevel(axis=1).sort_index(axis=1)
    return df


def test_extract_close_handles_yfinance_layouts():
    idx = pd.date_range("2024-01-02", periods=3, freq="B", tz="America/New_York")
    by_col = sz._extract_close(fake_download_frame(["AAA", "BBB"], idx), ["AAA", "BBB"])
    assert list(by_col.columns) == ["AAA", "BBB"] and by_col.index.tz is None
    assert by_col["BBB"].tolist() == [20.0, 21.0, 22.0]
    by_ticker = sz._extract_close(fake_download_frame(["AAA", "BBB"], idx, "ticker"), ["AAA", "BBB"])
    pd.testing.assert_frame_equal(by_col, by_ticker)
    flat = pd.DataFrame({"Open": [1.0, 2], "Close": [3.0, 4]}, index=pd.date_range("2024-01-02", periods=2))
    assert sz._extract_close(flat, ["ONE"])["ONE"].tolist() == [3.0, 4.0]
    missing = sz._extract_close(fake_download_frame(["AAA"], idx), ["AAA", "ZZZ"])
    assert missing["ZZZ"].isna().all()
    assert sz._extract_close(pd.DataFrame(), ["A"]).empty


def test_load_prices_renames_orders_and_drops_empty(monkeypatch):
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    calls = {}

    def download(tickers, **kw):
        calls.update(kw, tickers=tickers)
        df = fake_download_frame(tickers, idx)
        df[("Close", "DEAD")] = np.nan
        return df

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=download))
    universe = {"Beta": "BBB", "Alpha": "AAA", "Gone": "DEAD"}
    with pytest.warns(UserWarning, match="Gone"):
        px = sz.load_prices(universe, start="2024-01-01")
    assert list(px.columns) == ["Beta", "Alpha"]
    assert calls["auto_adjust"] is True and calls["group_by"] == "column" and calls["start"] == "2024-01-01"
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=lambda t, **k: pd.DataFrame()))
    with pytest.raises(RuntimeError, match="no data"):
        sz.load_prices({"A": "AAA"})


def test_universes_are_well_formed():
    assert len(sz.SECTORS) == 11 and "Healthcare" in sz.SECTORS
    for u in (sz.SECTORS, sz.COMMODITIES, sz.COMMODITY_ETFS):
        assert len(set(u.values())) == len(u)


# ----------------------------------------------------------------------- period returns

def test_anchor_dates():
    end = pd.Timestamp("2026-09-25")  # a Friday
    assert sz.anchor_date(end, "1W") == pd.Timestamp("2026-09-18")
    assert sz.anchor_date(end, "1M") == pd.Timestamp("2026-08-25")
    assert sz.anchor_date(end, "1y") == pd.Timestamp("2025-09-25")
    assert sz.anchor_date(end, "MTD") == pd.Timestamp("2026-08-31")
    assert sz.anchor_date(end, "QTD") == pd.Timestamp("2026-06-30")
    assert sz.anchor_date(end, "YTD") == pd.Timestamp("2025-12-31")
    assert sz.anchor_date(end, "1D", bdays("2026-09-01", "2026-09-30")) == pd.Timestamp("2026-09-24")
    with pytest.raises(ValueError):
        sz.anchor_date(end, "fortnight")


def test_period_returns_exact_and_stale():
    idx = bdays("2026-08-03", "2026-09-25")
    px = pd.DataFrame({"Up": np.linspace(100, 110, len(idx)), "Flat": 50.0}, index=idx)
    px.loc["2026-09-01":, "Stale"] = np.nan
    px.loc[:"2026-08-31", "Stale"] = 20.0
    out = sz.period_returns(px, "1W")
    assert out.loc["Up", "start_date"] == pd.Timestamp("2026-09-18")
    assert out.loc["Up", "return"] == pytest.approx(px["Up"].iloc[-1] / px.loc["2026-09-18", "Up"] - 1)
    assert out.loc["Flat", "return"] == 0.0
    assert np.isnan(out.loc["Stale", "return"])
    mtd = sz.period_returns(px, "MTD")
    assert mtd.loc["Up", "start_date"] == pd.Timestamp("2026-08-31")
    # a weekend end date uses Friday's close; a weekend anchor rolls back to the prior close
    wk = sz.period_returns(px, "1W", end="2026-09-20")  # Sunday; anchor Sunday 2026-09-13
    assert wk.loc["Up", "end_date"] == pd.Timestamp("2026-09-18")
    assert wk.loc["Up", "start_date"] == pd.Timestamp("2026-09-11")


# ----------------------------------------------------------------------- quarters & seasonality

def test_quarterly_returns_and_completion():
    idx = bdays("2024-01-02", "2024-09-27")  # Q3 2024 ends Mon Sep 30: incomplete
    px = pd.DataFrame({"A": 100.0}, index=idx)
    px.loc["2024-04-01":, "A"] = 110.0
    px.loc["2024-07-01":, "A"] = 99.0
    q = sz.quarterly_returns(px)
    assert list(q.index.astype(str)) == ["2024Q1", "2024Q2"]
    assert np.isnan(q.loc["2024Q1", "A"]) and q.loc["2024Q2", "A"] == pytest.approx(0.10)
    assert len(sz.quarterly_returns(px, complete_only=False)) == 3
    # a quarter ending on a weekend is complete once the last business day is in
    idx2 = bdays("2024-10-01", "2025-03-31")
    q2 = sz.quarterly_returns(pd.DataFrame({"A": np.linspace(1, 2, len(idx2))}, index=idx2))
    assert str(q2.index[-1]) == "2025Q1"
    idx3 = bdays("2025-01-02", "2025-06-27")  # Q2 2025 ends Mon Jun 30, data stops the Friday before
    assert str(sz.quarterly_returns(pd.DataFrame({"A": 1.0}, index=idx3)).index[-1]) == "2025Q1"


def test_next_quarter():
    assert str(sz.next_quarter("2026-09-29")) == "2026Q4"
    assert str(sz.next_quarter("2026-12-15")) == "2027Q1"


def test_seasonal_table_uses_last_n_same_quarters():
    periods = pd.period_range("2018Q1", "2025Q4", freq="Q")
    q = pd.DataFrame({"A": np.arange(len(periods)) / 100.0, "New": np.nan}, index=periods)
    q.loc[pd.Period("2024Q4"):, "New"] = 0.05
    t = sz.seasonal_table(q, 4, years=5)
    assert [c for c in t.columns if isinstance(c, (int, np.integer))] == [2021, 2022, 2023, 2024, 2025]
    expected = [q.loc[pd.Period(f"{y}Q4"), "A"] for y in range(2021, 2026)]
    assert t.loc["A", "mean"] == pytest.approx(np.mean(expected))
    assert t.loc["A", "positive"] == 5 and t.loc["A", "hit_rate"] == 1.0
    assert t.loc["New", "n"] == 2 and np.isnan(t.loc["New", "mean"])
    assert t.index[0] == "A"  # NaN means sort last
    assert sz.seasonal_table(q, 4, years=5, min_years=2).loc["New", "mean"] == pytest.approx(0.05)
    with pytest.raises(ValueError):
        sz.seasonal_table(q, 5)


# ----------------------------------------------------------------------- backtest

def test_signal_has_no_look_ahead():
    periods = pd.period_range("2010Q1", "2020Q4", freq="Q")
    rng = np.random.default_rng(1)
    q = pd.DataFrame(rng.normal(0, 0.05, (len(periods), 3)), index=periods, columns=list("abc"))
    sig = sz.seasonal_signal(q, years=5)
    t = pd.Period("2018Q3")
    manual = np.mean([q.loc[t - 4 * k, "a"] for k in range(1, 6)])
    assert sig.loc[t, "a"] == pytest.approx(manual)
    assert sig.loc[:pd.Period("2014Q4")].isna().all().all() and sig.loc[pd.Period("2015Q1")].notna().all()
    bumped = q.copy()
    bumped.loc[t] += 10.0  # changing quarter t must not change the signal at t
    assert sz.seasonal_signal(bumped, 5).loc[t, "a"] == pytest.approx(sig.loc[t, "a"])
    q2 = q.drop(index=pd.Period("2012Q2"))  # a gap is re-filled as NaN, not collapsed
    assert np.isnan(sz.seasonal_signal(q2, 5).loc[pd.Period("2016Q2"), "a"])


def test_backtest_on_persistent_seasonality():
    names = [f"N{i}" for i in range(9)]
    q = sz.quarterly_returns(seasonal_prices(names))
    bt = sz.backtest(q, years=5, top_n=3)
    assert bt.returns.index[0] == pd.Period("2010Q2")  # first quarter with 5 prior same-quarter returns
    assert list(bt.returns.columns) == ["Top 3", "Equal weight", "Bottom 3", "Top − Bottom"]
    np.testing.assert_allclose(bt.returns["Top − Bottom"], bt.returns["Top 3"] - bt.returns["Bottom 3"])
    assert bt.ic.mean() > 0.8 and bt.ic_summary()["share_positive"] > 0.9
    assert bt.sign_hit_rate() > 0.8
    assert 0 < bt.base_rate_positive() < 1 and "base_rate_positive" in bt.ic_summary()
    stats = bt.stats()
    assert stats.loc["Top 3", "ann_return"] > stats.loc["Equal weight", "ann_return"] > stats.loc["Bottom 3", "ann_return"]
    assert stats.loc["Top 3", "beats_equal_weight"] > 0.9 and np.isnan(stats.loc["Equal weight", "beats_equal_weight"])
    assert (stats["max_drawdown"] <= 0).all()
    g = bt.growth()
    assert (g.iloc[0] == 1.0).all() and len(g) == len(bt.returns) + 1
    assert list(bt.ic_by_quarter().index) == ["Q1", "Q2", "Q3", "Q4"]
    # holdings are the names with the highest signal that quarter
    t = bt.returns.index[5]
    top = bt.signal.loc[t].sort_values(ascending=False).index[:3]
    assert bt.holdings.loc[t, "top"] == ", ".join(top)
    assert bt.returns.loc[t, "Top 3"] == pytest.approx(bt.realized.loc[t, top].mean())


def test_backtest_on_noise_has_no_edge():
    names = [f"N{i}" for i in range(10)]
    q = sz.quarterly_returns(seasonal_prices(names, strength=0.0, noise=0.01, seed=3))
    bt = sz.backtest(q, years=5, top_n=3)
    assert abs(bt.ic_summary()["mean_ic"]) < 0.15


def test_backtest_skips_thin_quarters_and_errors_when_empty():
    names = [f"N{i}" for i in range(6)]
    px = seasonal_prices(names, start="2005-01-03")
    px.loc[:"2014-12-31", ["N4", "N5"]] = np.nan  # only 4 names before 2015
    bt = sz.backtest(sz.quarterly_returns(px), years=5, top_n=3)
    assert (bt.holdings["names_ranked"] >= 6).all()
    with pytest.raises(ValueError, match="history"):
        sz.backtest(sz.quarterly_returns(px.loc["2021":]), years=5)


# ----------------------------------------------------------------------- charts

def test_nice_axis():
    lo, hi, step = sz._nice_axis(-0.030, 0.019)
    assert (lo, hi, step) == (-0.04, 0.03, 0.01)
    lo, hi, step = sz._nice_axis(0.002, 0.012)
    assert lo == 0.0 and hi >= 0.012 * 1.2 - 1e-12 and step in (0.001, 0.002, 0.0025)
    lo, hi, step = sz._nice_axis(-0.4, -0.1)
    assert hi == 0.0 and lo <= -0.4 * 1.2
    assert sz._decimals(0.25) == 2 and sz._decimals(1.0) == 0 and sz._decimals(2.5) == 1


def test_bar_chart_matches_reference_layout():
    vals = pd.Series({"Utilities": -0.030, "Healthcare": 0.019, "Energy": -0.012, "Info Tech": 0.011, "Bad": np.nan})
    fig = sz.bar_chart(vals, "S&P Sector Returns", "1 week")
    bar = fig.data[0]
    assert bar.orientation == "h" and list(bar.y) == ["Healthcare", "Info Tech", "Energy", "Utilities"]
    assert list(bar.text) == ["1.9%", "1.1%", "-1.2%", "-3.0%"] and bar.textposition == "outside"
    assert list(fig.layout.yaxis.categoryarray) == ["Utilities", "Energy", "Info Tech", "Healthcare"]  # top = largest
    assert tuple(fig.layout.xaxis.range) == (-0.04, 0.03) and fig.layout.xaxis.tickformat == ".0%"
    assert fig.layout.xaxis.showgrid is False and fig.layout.xaxis.zeroline is True
    assert "S&amp;P Sector Returns" in fig.layout.title.text and "1 week" in fig.layout.title.text
    ic = sz.bar_chart(pd.Series({"Q1": 0.12, "Q2": -0.05}), "IC", percent=False, decimals=2)
    assert list(ic.data[0].text) == ["0.12", "-0.05"] and ic.layout.xaxis.tickformat.endswith("f")
    with pytest.raises(ValueError):
        sz.bar_chart(pd.Series({"a": np.nan}), "empty")


def test_returns_and_seasonal_charts():
    names = list(sz.SECTORS)
    px = seasonal_prices(names, end="2026-09-25")
    fig, table = sz.returns_chart(px, sz.SECTORS, "1W", "S&P Sector Returns")
    assert len(fig.data[0].y) == 11 and "Sep 18, 2026 → Sep 25, 2026" in fig.layout.title.text
    assert table["ticker"].tolist()[0] in sz.SECTORS.values() and table["return"].is_monotonic_decreasing
    assert "XLV" in "".join(fig.data[0].customdata)
    fig2, t2 = sz.seasonal_chart(px, sz.SECTORS, sz.next_quarter(px.index[-1]), 5, "S&P Sector Returns")
    assert "Q4 Seasonality" in fig2.layout.title.text and "2021–2025" in fig2.layout.title.text
    assert "upcoming quarter: Q4 2026" in fig2.layout.title.text
    assert [c for c in t2.columns if isinstance(c, (int, np.integer))] == [2021, 2022, 2023, 2024, 2025]
    assert "Q4 2023" in "".join(fig2.data[0].customdata)
    fig3, _ = sz.seasonal_chart(px, sz.SECTORS, 2, 5)
    assert "Q2 Seasonality" in fig3.layout.title.text and "upcoming" not in fig3.layout.title.text
    fig4, _ = sz.seasonal_chart(px, sz.SECTORS)  # default target = next quarter
    assert "Q4" in fig4.layout.title.text


def test_growth_chart_and_save_html(tmp_path):
    names = [f"N{i}" for i in range(8)]
    bt = sz.backtest(sz.quarterly_returns(seasonal_prices(names)), 5, 3)
    fig = sz.growth_chart(bt, "Backtest", "sub")
    assert [d.name for d in fig.data] == ["Top 3", "Equal weight", "Bottom 3"] and len(fig.layout.annotations) == 3
    bars = sz.bar_chart(pd.Series({"a": 0.01, "b": -0.02}), "Bars")
    out = sz.save_html(tmp_path / "r" / "report.html", [fig, bars])
    page = out.read_text()
    assert page.startswith("<!DOCTYPE html>") and page.count("Plotly.newPlot") == 2
    assert page.count("cdn.plot.ly") == 1


def test_styled_tables_render():
    names = [f"N{i}" for i in range(8)]
    px = seasonal_prices(names, end="2026-09-25")
    bt = sz.backtest(sz.quarterly_returns(px), 5, 3)
    stats_html = sz.styled(bt.stats()).to_html()
    assert "%" in stats_html and "quarters" in stats_html
    ic_html = sz.styled(bt.ic_summary().to_frame("x")).to_html()
    assert "%" in ic_html  # share_positive as a percentage
    _, recent = sz.returns_chart(px, {n: n for n in names}, "1M", "t")
    html_out = sz.styled(recent).to_html()
    assert "2026-09-25" in html_out and "%" in html_out
    _, season = sz.seasonal_chart(px, {n: n for n in names}, 4, 5, "t")
    assert "%" in sz.styled(season).to_html()
