"""Tests for the engine inside Sector_Seasonal_Backtest.ipynb.

The notebook is standalone, so the engine lives in one of its cells (tagged "engine"). These tests execute
that cell and check the logic on synthetic prices with known answers. No network.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "Sector_Seasonal_Backtest.ipynb"


@pytest.fixture(scope="module")
def E():
    nb = json.loads(NOTEBOOK.read_text())
    cells = [c for c in nb["cells"] if "engine" in c.get("metadata", {}).get("tags", [])]
    assert len(cells) == 1, "exactly one engine cell expected"
    ns: dict = {}
    exec(compile("".join(cells[0]["source"]), "engine", "exec"), ns)
    return type("Engine", (), ns)


SECTORS = {f"Sector {i}": f"S{i}" for i in range(8)}


def synthetic_prices(start="2000-01-03", end="2026-09-28", seed=0, season=0.03, noise=0.01):
    """Each sector has a persistent quarterly pattern; SPY is the average of the sectors."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    q = idx.quarter.to_numpy()
    cols = {}
    for i, t in enumerate(SECTORS.values()):
        pattern = season * np.array([np.sin(i + k) for k in range(4)])
        daily = pattern[q - 1] / 63 + 0.0002 + rng.normal(0, noise, len(idx))
        cols[t] = 50 * np.exp(np.cumsum(daily))
    px = pd.DataFrame(cols, index=idx)
    px["SPY"] = px.mean(axis=1)
    return px


@pytest.fixture(scope="module")
def px():
    return synthetic_prices()


# ------------------------------------------------------------------------------------ calendar & returns

def test_quarter_calendar_lag_and_incomplete_quarter(E):
    idx = pd.bdate_range("2024-01-02", "2024-09-27")  # Q3 2024 ends Monday Sep 30: incomplete
    cal = E.quarter_calendar(idx, lag=1)
    assert [str(q) for q in cal.index] == ["2024Q1", "2024Q2"]
    assert cal.loc[pd.Period("2024Q1"), "quarter_end"] == pd.Timestamp("2024-03-29")
    assert cal.loc[pd.Period("2024Q1"), "rebalance"] == pd.Timestamp("2024-04-01")
    assert E.quarter_calendar(idx, lag=0).loc[pd.Period("2024Q2"), "rebalance"] == pd.Timestamp("2024-06-28")
    done = E.quarter_calendar(pd.bdate_range("2024-01-02", "2024-09-30"))
    assert str(done.index[-1]) == "2024Q3" and pd.isna(done["rebalance"].iloc[-1])  # no day after yet


def test_holding_returns_with_zero_lag_equal_quarterly_returns(E, px):
    cal0 = E.quarter_calendar(px.index, lag=0)
    pd.testing.assert_frame_equal(E.quarterly_returns(px, cal0), E.holding_returns(px, cal0))
    cal1 = E.quarter_calendar(px.index, lag=1)
    q = pd.Period("2010Q2")
    buy, sell = cal1.loc[q - 1, "rebalance"], cal1.loc[q, "rebalance"]
    assert E.holding_returns(px, cal1).loc[q, "S0"] == pytest.approx(px.loc[sell, "S0"] / px.loc[buy, "S0"] - 1)


def test_seasonal_average_uses_only_past_same_quarters(E, px):
    cal = E.quarter_calendar(px.index)
    qr = E.quarterly_returns(px[["S0"]], cal)
    avg = E.seasonal_average(qr, 5)
    q = pd.Period("2015Q4")
    assert avg.loc[q, "S0"] == pytest.approx(np.mean([qr.loc[q - 4 * k, "S0"] for k in range(1, 6)]))
    bumped = qr.copy()
    bumped.loc[q] += 1.0
    assert E.seasonal_average(bumped, 5).loc[q, "S0"] == pytest.approx(avg.loc[q, "S0"])
    assert avg.loc[:pd.Period("2004Q4"), "S0"].isna().all()  # 2000Q1 has no prior close, so 2005Q1 is first


def test_cash_returns(E, px):
    cal = E.quarter_calendar(px.index)
    irx = pd.Series(4.0, index=px.index)
    cash = E.cash_returns(irx, cal)
    assert cash.iloc[1:].eq(0.01).all() and cash.iloc[0] == 0.0
    assert E.cash_returns(None, cal).eq(0.0).all()


# ------------------------------------------------------------------------------------ the rule

def hand_built(E, empty="cash", cost=0.0):
    """Two sectors, prices set so the signal and outcome of one quarter are known exactly."""
    idx = pd.bdate_range("2000-01-03", "2012-12-31")
    px = pd.DataFrame(100.0, index=idx, columns=["A", "B", "C", "D", "E", "SPY"])
    rng = np.random.default_rng(1)
    for c in px.columns:
        px[c] = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.005, len(idx))))
    sectors = {n: n for n in ["A", "B", "C", "D", "E"]}
    cal = E.quarter_calendar(px.index, lag=0)
    return E.run_backtest(px, cal, sectors, 3, empty_quarter=empty, cost_bps=cost, min_universe=5), px, cal


def test_rule_matches_its_definition(E, px):
    cal = E.quarter_calendar(px.index, lag=1)
    bt = E.run_backtest(px, cal, SECTORS, 5, cost_bps=0.0)
    w = bt.weights[E.RULE][bt.sectors]
    bought = w > 0
    should = bt.eligible & (bt.seasonal_avg > 0) & (bt.prior_return < bt.seasonal_avg)
    pd.testing.assert_frame_equal(bought, should, check_names=False)
    held = w.sum(axis=1)
    assert np.allclose(held[held > 0], 1.0)  # equal weights sum to one
    q = w.index[bought.any(axis=1)][10]
    picks = w.columns[bought.loc[q]]
    assert bt.returns.loc[q, E.RULE] == pytest.approx(bt.hold.loc[q, picks].mean())
    seasonal = bt.weights[E.SEASONAL][bt.sectors] > 0
    assert (seasonal | ~bought).all().all()  # every dip-buy pick is also a seasonal-only pick


def test_example_from_the_brief(E):
    """Prior quarter -5% and a +3% seasonal average: bought. Above the average or a negative average: not."""
    q = pd.Period("2020Q4")
    avg = pd.DataFrame({"Staples": [0.03], "Tech": [0.03], "Energy": [-0.02]}, index=[q])
    prior = pd.DataFrame({"Staples": [-0.05], "Tech": [0.06], "Energy": [-0.10]}, index=[q])
    eligible = avg.notna()
    pick = eligible & (avg > 0.0) & (prior < avg - 0.0)
    assert pick.loc[q].to_dict() == {"Staples": True, "Tech": False, "Energy": False}


def test_empty_quarters_follow_the_setting(E):
    bt, _, _ = hand_built(E, empty="cash")
    idle = bt.weights[E.RULE][bt.sectors].sum(axis=1) == 0
    assert idle.any()
    assert np.allclose(bt.returns.loc[idle, E.RULE], bt.cash.loc[idle[idle].index])
    bt_spy, _, _ = hand_built(E, empty="spy")
    idle2 = bt_spy.weights[E.RULE][bt_spy.sectors].sum(axis=1) == 0
    assert np.allclose(bt_spy.returns.loc[idle2, E.RULE], bt_spy.returns.loc[idle2, E.SPY_LABEL])
    with pytest.raises(ValueError):
        hand_built(E, empty="gold")


def test_trading_costs(E):
    bt0, _, _ = hand_built(E, cost=0.0)
    bt1, _, _ = hand_built(E, cost=10.0)
    diff = bt0.returns[E.RULE] - bt1.returns[E.RULE]
    assert np.allclose(diff, bt1.turnover[E.RULE] * 10 / 1e4)
    assert bt1.turnover[E.SPY_LABEL].iloc[0] == pytest.approx(1.0) and bt1.turnover[E.SPY_LABEL].iloc[1:].abs().max() < 1e-9


def test_performance_numbers(E, px):
    cal = E.quarter_calendar(px.index)
    bt = E.run_backtest(px, cal, SECTORS, 5)
    p = E.performance(bt)
    r = bt.returns[E.RULE]
    assert p.loc["CAGR", E.RULE] == pytest.approx((1 + r).prod() ** (4 / len(r)) - 1)
    eq = pd.concat([pd.Series([1.0]), (1 + r).cumprod()])
    assert p.loc["Max drawdown", E.RULE] == pytest.approx((eq / eq.cummax() - 1).min())
    assert p.loc["CAGR vs SPY", E.RULE] == pytest.approx(p.loc["CAGR", E.RULE] - p.loc["CAGR", E.SPY_LABEL])
    assert np.isnan(p.loc["t-stat vs SPY", E.SPY_LABEL])
    assert p.attrs["period"].startswith(str(bt.returns.index[0]))
    sub = E.performance(bt, bt.returns.index[:20])
    assert sub.attrs["period"].endswith("(20 quarters)")


def test_breakdowns(E, px):
    cal = E.quarter_calendar(px.index)
    bt = E.run_backtest(px, cal, SECTORS, 5)
    trades = bt.trades()
    assert len(trades) == int((bt.weights[E.RULE][bt.sectors] > 0).sum().sum())
    sb = E.sector_breakdown(bt)
    assert sb["times_bought"].sum() == len(trades)
    fe = E.filter_effect(bt)
    seasonal = int((bt.eligible & (bt.seasonal_avg > 0)).sum().sum())
    assert fe["sector_quarters"].sum() == seasonal  # no missing values counted
    assert fe.iloc[0]["sector_quarters"] == len(trades)
    assert 0 <= fe["win_rate"].min() and fe["win_rate"].max() <= 1
    yrs = E.calendar_years(bt)
    assert yrs.index[0] == bt.returns.index[0].year
    assert "nothing qualified" in "".join(bt.holdings()) or (bt.weights[E.RULE][bt.sectors].sum(axis=1) > 0).all()


# ------------------------------------------------------------------------------------ research tools

def test_grid_and_out_of_sample(E, px):
    cal = E.quarter_calendar(px.index)
    grid, bts = E.grid_search(px, cal, SECTORS, [3, 5, 10], [0.0, 0.01], cost_bps=5.0)
    assert len(grid) == 6 and set(grid.index.names) == {"lookback_years", "min_avg"}
    common = grid.attrs["common"]
    assert common[0] == bts[(10, 0.0)].returns.index[0]  # everyone measured from the longest lookback's start
    oos = E.in_and_out_of_sample(bts, "2017Q1")
    best = oos.attrs["best"]
    assert oos.loc[best, "in-sample Sharpe"] == oos["in-sample Sharpe"].max()
    labels = E.setting_labels(oos)
    assert labels.index[0].endswith("%") and "years · avg >" in labels.index[0]
    with pytest.raises(ValueError):
        E.in_and_out_of_sample(bts, "2026Q1")


def test_luck_test(E, px):
    cal = E.quarter_calendar(px.index)
    bt = E.run_backtest(px, cal, SECTORS, 5)
    sims, p = E.random_pick_test(bt, n_sims=300, seed=1)
    assert sims.shape == (300,) and 0 <= p <= 1
    # if the rule held every eligible sector, random picks are identical to it
    bt.weights[E.RULE].loc[:, bt.sectors] = bt.eligible.astype(float).div(bt.eligible.sum(axis=1), axis=0).to_numpy()
    rets = bt.hold[bt.sectors].where(bt.eligible).mean(axis=1) - bt.costs[E.RULE]
    bt.returns[E.RULE] = rets
    sims2, p2 = E.random_pick_test(bt, n_sims=50, seed=2)
    assert np.allclose(sims2, E.performance(bt).loc["CAGR", E.RULE]) and p2 == 1.0


def test_seasonal_pattern_is_found_and_noise_is_not(E):
    strong = synthetic_prices(season=0.06, noise=0.004, seed=3)
    cal = E.quarter_calendar(strong.index)
    bt = E.run_backtest(strong, cal, SECTORS, 5, cost_bps=0.0)
    fe = E.filter_effect(bt)
    assert E.performance(bt).loc["CAGR", E.SEASONAL] > E.performance(bt).loc["CAGR", E.EW]
    noise = synthetic_prices(season=0.0, noise=0.01, seed=4)
    bt_n = E.run_backtest(noise, E.quarter_calendar(noise.index), SECTORS, 5, cost_bps=0.0)
    _, p = E.random_pick_test(bt_n, n_sims=400, seed=5)
    assert p > 0.01  # no edge on pure noise
    assert fe.shape == (2, 4)


def test_current_signal(E, px):
    partial = px.loc[:"2026-08-14"]
    cal = E.quarter_calendar(partial.index)
    sig = E.current_signal(partial, cal, SECTORS, 5)
    assert sig.attrs["next"] == "2026Q4" and "2026Q3 to date" in sig.attrs["current"]
    assert sig.attrs["years"] == "2021–2025"
    last_close = cal["quarter_end"].iloc[-1]
    assert sig.loc["Sector 0", "current_quarter"] == pytest.approx(partial["S0"].iloc[-1] / partial.loc[last_close, "S0"] - 1)
    qr = E.quarterly_returns(partial[["S0"]], cal)
    q4 = qr[qr.index.quarter == 4].tail(5)["S0"].mean()
    assert sig.loc["Sector 0", "next_quarter_average"] == pytest.approx(q4)
    expect = (sig["next_quarter_average"] > 0) & (sig["current_quarter"] < sig["next_quarter_average"])
    assert (sig["qualifies"] == expect).all()
    done = px.loc[:"2026-06-30"]
    sig2 = E.current_signal(done, E.quarter_calendar(done.index), SECTORS, 5)
    assert sig2.attrs["next"] == "2026Q3" and "complete" in sig2.attrs["current"]


def test_scorecard_and_charts(E, px):
    cal = E.quarter_calendar(px.index)
    bt = E.run_backtest(px, cal, SECTORS, 5)
    grid, bts = E.grid_search(px, cal, SECTORS, [3, 5], [0.0, 0.01])
    oos = E.in_and_out_of_sample(bts, "2016Q1")
    sims, p = E.random_pick_test(bt, n_sims=100)
    card = E.scorecard(bt, grid, oos, p)
    assert len(card) == 7 and set(card["pass"]) <= {"✔ pass", "✘ fail"}
    E.set_style()
    figs = [E.equity_chart(bt), E.drawdown_chart(bt), E.yearly_chart(bt),
            E.bar_chart(E.sector_breakdown(bt)["times_bought"], "Times", fmt="{:.0f}"),
            E.heatmap(grid, "Sharpe", "Sharpe", "sub", center=0.3),
            E.luck_chart(sims, 0.08, 0.05, p)]
    import matplotlib.pyplot as plt
    assert all(isinstance(f, plt.Figure) for f in figs)
    labels = [t.get_text() for t in figs[0].axes[0].texts]
    assert any(l.startswith(E.RULE) for l in labels)
    plt.close("all")


def test_load_prices_parses_yfinance(E, monkeypatch):
    import sys
    import types
    idx = pd.date_range("2024-01-02", periods=4, freq="B", tz="America/New_York")
    cols = pd.MultiIndex.from_product([["Close", "Open"], ["AAA", "BBB"]], names=["Price", "Ticker"])
    raw = pd.DataFrame(np.arange(16, dtype=float).reshape(4, 4) + 1, index=idx, columns=cols)
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=lambda *a, **k: raw))
    out = E.load_prices(["AAA", "BBB"], start="2024-01-01")
    assert list(out.columns) == ["AAA", "BBB"] and out.index.tz is None and out["AAA"].iloc[0] == 1.0
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=lambda *a, **k: pd.DataFrame()))
    with pytest.raises(RuntimeError, match="no data"):
        E.load_prices(["AAA"], start="2024-01-01")
