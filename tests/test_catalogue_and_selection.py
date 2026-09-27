"""Wider indicator catalogue, manual-only series, indicator selection, façade indicator choice."""
import numpy as np
import pandas as pd
import pytest

import config
from data.fred_fetcher import FredError, FredFetcher
from data.synthetic import FakeFredSession, synthetic_raw, write_demo_manual_csvs
from data.transforms import CATALOGUE, CATALOGUE_KEYS, MANUAL_ONLY_SERIES, features_from_panel, get_spec, select, to_month_end
from model.selection import evaluate_indicators
from regime import MacroRegime

WIDE = ["yield_curve", "core_pce", "fed_funds", "ism_pmi", "credit_spread", "unemployment", "jobless_claims",
        "payrolls", "cpi", "ism_services_pmi"]


@pytest.fixture
def wide_fetcher(tmp_path, raw_series):
    write_demo_manual_csvs(raw_series, tmp_path / "manual")
    f = FredFetcher(api_key="k", cache_dir=tmp_path / "fred", manual_dir=tmp_path / "manual",
                    session=FakeFredSession.from_raw(raw_series), indicators=WIDE)
    f.refresh()
    return f


def test_catalogue_is_complete_and_categorised():
    assert len(CATALOGUE) == 15
    cats = {s.category for s in CATALOGUE}
    assert cats == {"Interest rates / credit", "Inflation", "Labour market", "Activity / sentiment"}
    assert {s.key for s in CATALOGUE if s.manual_only} == {"ism_services_pmi", "chicago_pmi", "consumer_confidence"}
    assert MANUAL_ONLY_SERIES == {"ISM_SERVICES_PMI", "CHICAGO_PMI", "CONF_BOARD_CCI"}
    assert all(s.bullish_sign in (-1, 1) for s in CATALOGUE)
    assert select(["cpi", "ppi"])[1].feature_name == "ppi_mom_3m"
    with pytest.raises(KeyError):
        select(["gdp"])
    assert set(config.DEFAULT_INDICATORS.split(",")) <= set(CATALOGUE_KEYS)


def test_weekly_claims_are_monthly_means():
    weeks = pd.date_range("2024-01-06", "2024-03-30", freq="W-SAT")
    s = pd.Series(np.arange(len(weeks), dtype=float) * 1000 + 200_000, index=weeks)
    m = to_month_end(s, "mean")
    jan = s[s.index.month == 1].mean()
    assert m.loc["2024-01-31"] == pytest.approx(jan)
    assert get_spec("jobless_claims").monthly_agg == "mean"
    head = get_spec("jobless_claims").headline({"ICSA": s})
    assert head.loc["2024-01-31"] == pytest.approx(jan / 1000)


def test_new_transforms():
    idx = pd.date_range("2020-01-31", periods=15, freq="ME")
    un = pd.Series(np.linspace(4.0, 5.4, 15), idx)
    f = get_spec("unemployment").feature(un)
    assert f.iloc[:12].isna().all() and f.iloc[12] == pytest.approx(un.iloc[12] - un.iloc[0])
    pay = pd.Series(100_000 + np.arange(15) * 150.0, idx)
    head = get_spec("payrolls").headline({"PAYEMS": pay})
    assert head.iloc[-1] == pytest.approx(150.0)
    ppi = pd.Series(100 * 1.005 ** np.arange(15), idx)
    h = get_spec("ppi").headline({"PPIACO": ppi})
    assert h.iloc[-1] == pytest.approx(0.5, abs=1e-6)
    assert get_spec("ppi").feature(h).iloc[-1] == pytest.approx(0.5, abs=1e-6)
    assert get_spec("credit_spread").feature(pd.Series([2.0], idx[:1])).iloc[0] == 2.0


def test_wide_fetcher_builds_full_panel_with_manual_series(wide_fetcher):
    panel = wide_fetcher.panel()
    assert list(panel.columns) == WIDE and wide_fetcher.keys == tuple(WIDE)
    assert wide_fetcher.manual_only_ids == {"ISM_SERVICES_PMI"}
    assert wide_fetcher.get_actual("ism_services_pmi", "2025-08-31") > 30
    feats = wide_fetcher.features().dropna()
    assert feats.shape[1] == len(WIDE) and feats.index[-1] == pd.Timestamp("2025-08-31")
    td = wide_fetcher.training_data()
    assert set(td.feature_signs) == set(feats.columns)
    # features_from_panel infers the specs from the columns when not given
    pd.testing.assert_frame_equal(features_from_panel(panel), wide_fetcher.features())


def test_manual_only_series_without_csv_fails_clearly(tmp_path, raw_series):
    f = FredFetcher(api_key="k", cache_dir=tmp_path / "fred", manual_dir=tmp_path / "manual",
                    session=FakeFredSession.from_raw(raw_series), indicators=["yield_curve", "chicago_pmi"])
    with pytest.raises(FredError, match="not published on FRED"):
        f.refresh()
    assert not any("CHICAGO_PMI" == c["series_id"] for c in f.session.calls)  # never asked FRED for it


def test_wide_model_fits_scores_and_overrides(wide_fetcher, tmp_path):
    from data.overrides import OverrideStore
    from model.hmm_engine import HMMEngine
    from model.labeler import StateLabeler
    from scoring.live_score import live_score
    engine = HMMEngine(n_states=3, n_restarts=1, n_iter=100).fit(wide_fetcher.training_data())
    assert engine.feature_names == tuple(s.feature_name for s in wide_fetcher.indicators)
    store = OverrideStore(wide_fetcher, tmp_path / "o.json")
    un_actual = wide_fetcher.get_actual("unemployment", "2025-08-31")
    store.set_override("unemployment", un_actual + 1.0)
    store.set_override("ism_services_pmi", 40.0)
    s = live_score(wide_fetcher, store, engine, StateLabeler(tmp_path / "l.json"))
    assert len(s.indicators) == len(WIDE) and s.has_overrides
    un = next(i for i in s.indicators if i.key == "unemployment")
    assert un.overridden and un.feature_scenario == pytest.approx(un.feature_actual + 1.0)
    assert s.prob(2, scenario=True) <= s.prob(2) + 1e-9  # rising unemployment never makes it more bullish


def test_evaluate_indicators_ranks_candidates(wide_fetcher):
    base = ["yield_curve", "core_pce", "fed_funds", "ism_pmi"]
    cands = ["unemployment", "credit_spread", "cpi"]
    df = evaluate_indicators(wide_fetcher, base, cands, n_states=3, refit_every=60, min_train=240, n_restarts=1, n_iter=60)
    assert list(df.index) == sorted(cands, key=lambda k: (not df.loc[k, "keep"], -df.loc[k, "relevance_f"]))
    for col in ("relevance_f", "redundancy_r2", "confidence", "confidence_base", "switches_per_year", "agreement", "keep"):
        assert col in df.columns
    assert (df["relevance_f"] > 0).all() and df["redundancy_r2"].between(-0.01, 1.0).all()
    assert df.loc["cpi", "redundancy_r2"] > 0.5  # CPI is largely explained by Core PCE
    assert df["confidence_base"].nunique() == 1


def test_macro_regime_with_custom_indicators(tmp_path):
    mr = MacroRegime(demo=True, cache_dir=tmp_path / "d", indicators=["yield_curve", "ism_pmi", "unemployment", "chicago_pmi"])
    assert mr.indicators == ("yield_curve", "ism_pmi", "unemployment", "chicago_pmi")
    assert (tmp_path / "d" / "manual" / "CHICAGO_PMI.csv").exists()
    mr.refit(n_restarts=1)
    assert mr.panel().shape[1] == 4
    mr.override(chicago_pmi=38)
    with pytest.raises(KeyError, match="inactive"):
        mr.override(core_pce=3.0)
    assert "Using override: 38.0" in mr.report(history_years=3, include_indicator_history=False).html
    df = mr.evaluate_indicators(candidates=["core_pce", "credit_spread"], refit_every=60, min_train=240, n_restarts=1, n_iter=60)
    assert set(df.index) == {"core_pce", "credit_spread"} and (tmp_path / "d" / "fred_eval").exists()
    assert (tmp_path / "d" / "fred" / "panel.parquet").exists()  # production cache untouched by the eval
