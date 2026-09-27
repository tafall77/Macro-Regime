import numpy as np
import pandas as pd
import pytest

import config
from data.fred_fetcher import FredFetcher, StaleDataError
from data.overrides import OverrideStore
from scoring.live_score import EngineLoader, live_score, scenario_panel
from tests.conftest import FakeFredSession


def test_no_override_means_identical_runs(fetcher, overrides, fitted_engine, labeler):
    s = live_score(fetcher, overrides, fitted_engine, labeler)
    assert s.as_of == pd.Timestamp("2025-08-31") and s.fit_id == fitted_engine.fit_id
    np.testing.assert_allclose(s.probs_actual, s.probs_scenario)
    assert s.probs_actual.sum() == pytest.approx(1.0)
    assert not s.has_overrides and all(i.effective == i.actual for i in s.indicators)
    assert s.top_actual == s.top_scenario and s.delta(s.top_actual) == 0.0
    assert s.persistence_actual == pytest.approx(fitted_engine.persistence(s.top_actual))
    # re-scoring the final training month reproduces the fitted posterior for that month
    np.testing.assert_allclose(s.probs_actual, fitted_engine.filtered_.iloc[-1].to_numpy(), atol=1e-8)
    d = s.to_dict()
    assert set(d["score_actual"]) == {"Bearish", "Transition", "Bullish"} and d["labels_provisional"]


def test_override_changes_only_the_scenario_run(fetcher, overrides, fitted_engine, labeler):
    base = live_score(fetcher, overrides, fitted_engine, labeler)
    overrides.set_override("ism_pmi", 41.0)
    overrides.set_override("yield_curve", -0.6)
    s = live_score(fetcher, overrides, fitted_engine, labeler)
    np.testing.assert_allclose(s.probs_actual, base.probs_actual)  # actual is untouched
    assert s.has_overrides
    pmi = next(i for i in s.indicators if i.key == "ism_pmi")
    assert pmi.overridden and pmi.effective == 41.0 and pmi.actual == base.indicators[3].actual
    assert pmi.feature_scenario == pytest.approx(-9.0) and pmi.feature_actual == pytest.approx(pmi.actual - 50)
    ff = next(i for i in s.indicators if i.key == "fed_funds")
    assert not ff.overridden and ff.feature_actual == ff.feature_scenario
    # a bearish scenario must raise the bearish probability and lower the bullish one
    assert s.delta(0) > 0 and s.delta(2) <= 0
    assert s.prob(0, scenario=True) > s.prob(0)


def test_fed_funds_override_is_applied_through_the_rate_of_change(fetcher, overrides, fitted_engine, labeler):
    n = config.FED_FUNDS_ROC_MONTHS
    prev = fetcher.get_actual("fed_funds", fetcher.latest_date() - pd.offsets.MonthEnd(n))
    overrides.set_override("fed_funds", prev + 1.0)  # a 100bp hike vs. n months ago
    s = live_score(fetcher, overrides, fitted_engine, labeler)
    ff = next(i for i in s.indicators if i.key == "fed_funds")
    assert ff.feature_scenario == pytest.approx(1.0)
    assert ff.feature_actual == pytest.approx(fetcher.get_actual("fed_funds", s.as_of) - prev)


def test_lagging_indicator_is_carried_forward_and_scored_after_training_end(tmp_path, raw_series, fitted_engine, labeler):
    # Yields extend to Oct 2025 while PCE/PMI/FF stop in Aug 2025 -> as_of = Oct, two rows after training end
    series = {k: v for k, v in raw_series.items() if not k.startswith("_")}
    extra_days = pd.bdate_range("2025-09-01", "2025-10-31")
    for sid in ("DGS10", "DGS2"):
        tail = pd.Series(series[sid].iloc[-1], index=extra_days)
        series[sid] = pd.concat([series[sid], tail])
    f = FredFetcher(api_key="k", cache_dir=tmp_path / "fred", manual_dir=tmp_path / "m", session=FakeFredSession(series))
    f.refresh()
    store = OverrideStore(f)
    s = live_score(f, store, fitted_engine, labeler)
    assert s.as_of == pd.Timestamp("2025-10-31")
    pce = next(i for i in s.indicators if i.key == "core_pce")
    assert pce.actual_date == pd.Timestamp("2025-08-31") and pce.actual == f.get_actual("core_pce", "2025-08-31")
    assert s.probs_actual.sum() == pytest.approx(1.0)
    # the two post-training rows were filtered from the last fitted posterior
    rows = f.features(f.panel().ffill(limit=3)).loc["2025-09-30":"2025-10-31"]
    np.testing.assert_allclose(s.probs_actual, fitted_engine.score(rows), atol=1e-10)


def test_stale_indicator_raises_a_clear_error(tmp_path, raw_series, fitted_engine, labeler):
    series = {k: v for k, v in raw_series.items() if not k.startswith("_")}
    series[config.PMI_SERIES_ID] = series[config.PMI_SERIES_ID].loc[:"2024-12-31"]  # 8 months stale
    f = FredFetcher(api_key="k", cache_dir=tmp_path / "fred", manual_dir=tmp_path / "m", session=FakeFredSession(series))
    f.refresh()
    with pytest.raises(StaleDataError, match="pmi_vs_50"):
        live_score(f, OverrideStore(f), fitted_engine, labeler)


def test_scenario_panel_only_touches_the_latest_row(fetcher, overrides):
    overrides.set_override("core_pce", 9.9)
    actual, scenario, as_of = scenario_panel(fetcher, overrides)
    diff = (actual != scenario) & ~(actual.isna() & scenario.isna())
    assert diff.sum().sum() == 1 and bool(diff.loc[as_of, "core_pce"])


def test_engine_loader_reloads_when_file_changes(tmp_path, fitted_engine):
    path = fitted_engine.save(tmp_path / "hmm.pkl")
    loader = EngineLoader(path)
    first = loader.get()
    assert loader.get() is first
    fitted_engine.save(path)
    import os
    os.utime(path, (os.path.getmtime(path) + 5, os.path.getmtime(path) + 5))
    assert loader.get() is not first
