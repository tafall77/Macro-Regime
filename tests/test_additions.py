"""Tests for demo data, walk-forward backtest, HTML report, figures and the MacroRegime façade."""
import numpy as np
import pandas as pd
import pytest

import config
from data.synthetic import FakeFredSession, synthetic_raw
from model.backtest import walk_forward
from regime import MacroRegime
from reporting import figures as F
from reporting.report import Report, build_history, build_report, plotly_cdn_url, plotly_loader_script


def test_synthetic_raw_shapes_and_regimes():
    raw = synthetic_raw(seed=1, end="2025-08-31")
    from data.transforms import CATALOGUE, series_ids_for
    assert set(raw) >= set(series_ids_for(CATALOGUE)) | {"_states"}
    assert raw["_states"].index[-1] == pd.Timestamp("2025-08-31") and set(raw["_states"].unique()) == {0, 1, 2}
    assert raw["FEDFUNDS"].min() >= 0.05
    noisy = synthetic_raw(seed=1, end="2025-08-31", noise={"pmi": 5.0})
    assert noisy[config.PMI_SERIES_ID].std() > raw[config.PMI_SERIES_ID].std()
    assert synthetic_raw(seed=2)["_states"].index[-1] < pd.Timestamp.today()


def test_walk_forward_is_out_of_sample(fetcher, true_states):
    td = fetcher.training_data()
    bt = walk_forward(td, n_states=3, refit_every=36, min_train=240, n_restarts=1, n_iter=100)
    assert bt.posteriors.index[0] == td.frame.index[240]
    assert bt.posteriors.index[-1] == td.end
    assert np.allclose(bt.posteriors.sum(axis=1), 1.0)
    assert len(bt.fits) == int(np.ceil((len(td) - 240) / 36))
    assert (bt.fits["train_end"] < bt.fits["scored_to"]).all()  # never trained on what it scores
    assert bt.accuracy(true_states) > 0.6
    summ = bt.summary()
    assert summ["share"].sum() == pytest.approx(1.0) and (summ["avg_run_months"] > 1).all()
    with pytest.raises(ValueError):
        walk_forward(td, min_train=10_000)


def test_figures_build(fitted_engine):
    hist = fitted_engine.filtered_.iloc[-60:]
    fig = F.history_figure(hist, {0: "bearish", 1: "transition", 2: "bullish"}, hist.index[-3])
    assert len(fig.data) == 3 and fig.data[0].name == "Bearish" and fig.data[0].fillcolor == F.REGIME_COLORS["bearish"]
    fig2 = F.history_figure(hist)  # unlabeled fallback
    assert fig2.data[1].name == "State 1"
    dist = F.distribution_figure(["A", "B"], [0.3, 0.7], [0.6, 0.4])
    assert dist.data[1].y[0] == pytest.approx(60)
    assert F.state_color(4, None) != F.state_color(0, "bullish")


def test_build_history_extends_past_training_end(tmp_path, raw_series, fitted_engine):
    series = {k: v for k, v in raw_series.items() if not k.startswith("_")}
    extra = pd.bdate_range("2025-09-01", "2025-09-30")
    for sid in ("DGS10", "DGS2"):
        series[sid] = pd.concat([series[sid], pd.Series(series[sid].iloc[-1], index=extra)])
    from data.fred_fetcher import FredFetcher
    f = FredFetcher(api_key="k", cache_dir=tmp_path / "fred", manual_dir=tmp_path / "m", session=FakeFredSession(series))
    f.refresh()
    hist = build_history(f, fitted_engine, years=5)
    assert hist.index[-1] == pd.Timestamp("2025-09-30")
    assert len(hist) == len(fitted_engine.filtered_.loc["2020-09-30":]) + 1  # training path + one live month
    assert np.allclose(hist.sum(axis=1), 1.0)


def test_report_html_and_save(fetcher, overrides, fitted_engine, labeler, tmp_path):
    from scoring.live_score import live_score
    overrides.set_override("ism_pmi", 43)
    score = live_score(fetcher, overrides, fitted_engine, labeler)
    rep = build_report(score, fetcher, fitted_engine, history_years=10)
    assert isinstance(rep, Report) and rep._repr_html_() == rep.html
    for needle in ("Using override: 43.0", "Scenario mode: 1 override", "mr-gauge-actual", "mr-history", "mr-ind",
                   "Plotly.newPlot", "__mrPlotlyReady", "State legend", "provisional"):
        assert needle in rep.html, needle
    assert rep.html.count("Plotly.newPlot") == 5
    out = rep.save(tmp_path / "r" / "report.html")
    page = out.read_text()
    assert page.startswith("<!DOCTYPE html>") and plotly_cdn_url() in page and "<script>!function" not in page
    inline = rep.save(tmp_path / "inline.html", inline_js=True)
    assert inline.stat().st_size > 1_000_000  # plotly.js embedded
    assert "cdn.plot.ly" in plotly_cdn_url() and "requirejs" in plotly_loader_script()
    slim = build_report(score, fetcher, fitted_engine, include_indicator_history=False, history_years=None)
    assert "mr-ind" not in slim.html and slim.html.count("Plotly.newPlot") == 4


def test_macro_regime_demo_end_to_end(tmp_path):
    mr = MacroRegime(demo=True, cache_dir=tmp_path / "demo", seed=1)
    assert not mr.is_fitted and "unfitted" in repr(mr)
    engine = mr.refit(n_restarts=2)
    assert mr.is_fitted and mr.engine.fit_id == engine.fit_id
    assert (tmp_path / "demo" / "model" / "hmm_engine.pkl").exists()
    states = mr.states()
    assert list(states["suggested"]) == ["bearish", "transition", "bullish"]
    assert mr.label() == {0: "bearish", 1: "transition", 2: "bullish"}
    assert not mr.score().labels.provisional
    base = mr.summary()
    assert base["actual"].sum() == pytest.approx(1.0) and (base["delta"] == 0).all()
    assert mr.override(ism_pmi=42, yield_curve=-0.5) == {"ism_pmi": 42.0, "yield_curve": -0.5}
    scen = mr.summary()
    np.testing.assert_allclose(scen["actual"], base["actual"])
    assert scen.loc[0, "delta"] >= 0
    with pytest.raises(KeyError):
        mr.override(gdp=1)
    mr.override(yield_curve=None)
    assert mr.overrides.overrides() == {"ism_pmi": 42.0}
    rep = mr.report(history_years=5)
    assert "Using override: 42.0" in rep.html
    mr.clear_overrides()
    assert not mr.score().has_overrides
    assert mr.true_states is not None and mr.panel().shape[1] == 4
    # demo mode never touches the real cache paths
    assert mr.fetcher.cache_dir == tmp_path / "demo" / "fred"
    # a second instance on the same cache reuses the fit without refitting
    again = MacroRegime(demo=True, cache_dir=tmp_path / "demo")
    assert again.is_fitted and again.engine.fit_id == engine.fit_id
    # refit() defaults to refreshing only when there is no cache yet
    n = len(again.fetcher.session.calls)
    again.refit(n_restarts=1)
    assert len(again.fetcher.session.calls) == n


def test_macro_regime_real_mode_paths(tmp_path):
    mr = MacroRegime(api_key="k", cache_dir=tmp_path / "real")
    assert not mr.demo and mr.true_states is None and mr.fetcher.api_key == "k"
    assert mr.fetcher.cache_dir == tmp_path / "real" / "fred"
    assert MacroRegime(demo=True).root == config.CACHE_DIR / "demo"
