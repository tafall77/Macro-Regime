import numpy as np
import pandas as pd
import pytest

import config
from data.fred_fetcher import CacheMissingError, FredError, FredFetcher, StaleDataError, _parse_observations
from data.transforms import FEATURE_NAMES, INDICATOR_KEYS
from model.hmm_engine import ACTUAL_PROVENANCE, TrainingData
from tests.conftest import FakeFredSession


def test_parse_observations_drops_missing_markers():
    s = _parse_observations("X", [{"date": "2024-01-01", "value": "1.5"}, {"date": "2024-02-01", "value": "."},
                                  {"date": "2024-03-01", "value": "2.5"}])
    assert list(s.index) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-01")]
    assert s.dtype == "float64"


def test_refresh_writes_cache_and_panel(fetcher, fake_session):
    assert fetcher.panel_path.exists() and fetcher.meta_path.exists()
    for sid in fetcher.series_ids:
        assert fetcher.raw_path(sid).exists()
    panel = fetcher.panel()
    assert list(panel.columns) == list(INDICATOR_KEYS)
    assert panel.index.freqstr in ("ME", "M") or (panel.index == pd.date_range(panel.index[0], panel.index[-1], freq="ME")).all()
    assert panel.index[-1] == pd.Timestamp("2025-08-31")
    # every request carried the key and the observation start
    assert all(c["api_key"] == "test-key" and c["observation_start"] == "1985-01-01" for c in fake_session.calls)
    assert set(c["series_id"] for c in fake_session.calls) == set(fetcher.series_ids)
    meta = fetcher.meta()
    assert meta["indicators"]["ism_pmi"]["last"] == "2025-08-31"


def test_pagination_is_followed(tmp_path, raw_series):
    session = FakeFredSession({k: v for k, v in raw_series.items() if not k.startswith("_")}, page_limit=1000)
    f = FredFetcher(api_key="k", cache_dir=tmp_path / "fred", manual_dir=tmp_path / "m", session=session)
    s = f.fetch_series("DGS10")
    assert len(s) > 1000
    offsets = [c["offset"] for c in session.calls if c["series_id"] == "DGS10"]
    assert offsets[:3] == [0, 1000, 2000]


def test_reload_from_disk_matches_refresh(fetcher, tmp_path):
    fresh = FredFetcher(api_key="k", cache_dir=fetcher.cache_dir, manual_dir=tmp_path / "m", session=None)
    pd.testing.assert_frame_equal(fresh.panel(), fetcher.panel(), check_freq=False)
    assert fresh.load_raw("FEDFUNDS").name == "FEDFUNDS"


def test_missing_cache_raises(tmp_path):
    f = FredFetcher(api_key="k", cache_dir=tmp_path / "nothing", manual_dir=tmp_path / "m", session=object())
    with pytest.raises(CacheMissingError):
        f.panel()
    with pytest.raises(CacheMissingError):
        f.load_raw("DGS10")


def test_missing_api_key_and_unknown_series_raise(tmp_path, fake_session):
    f = FredFetcher(api_key="", cache_dir=tmp_path / "fred", manual_dir=tmp_path / "m", session=fake_session)
    with pytest.raises(FredError, match="FRED_API_KEY"):
        f.fetch_series("DGS10")
    f.api_key = "k"
    with pytest.raises(FredError, match="does not exist"):
        f.fetch_series("NOPE")


def test_get_actual_and_latest_date(fetcher, raw_series):
    latest = fetcher.latest_date()
    assert latest == pd.Timestamp("2025-08-31")
    assert fetcher.latest_date("core_pce") == latest
    # month granularity: any day in the month maps to that month's value
    v_end = fetcher.get_actual("ism_pmi", "2025-08-31")
    v_mid = fetcher.get_actual("ism_pmi", "2025-08-12")
    assert v_end == v_mid == pytest.approx(float(raw_series[config.PMI_SERIES_ID].loc["2025-08-01"]))
    # yield spread is 10y - 2y at the last business day of the month
    d10, d2 = raw_series["DGS10"], raw_series["DGS2"]
    last_day = d10.loc[:"2025-08-31"].index[-1]
    assert fetcher.get_actual("yield_curve", "2025-08-31") == pytest.approx(d10[last_day] - d2[last_day], abs=1e-5)
    with pytest.raises(KeyError):
        fetcher.get_actual("not_an_indicator", "2025-08-31")
    with pytest.raises(StaleDataError):
        fetcher.get_actual("fed_funds", "1980-01-31")


def test_get_actual_is_as_of_within_stale_limit_then_raises(tmp_path, raw_series):
    # Truncate the PMI series so it ends in 2016 (what FRED's discontinued NAPM looks like)
    series = {k: v for k, v in raw_series.items() if not k.startswith("_")}
    series[config.PMI_SERIES_ID] = series[config.PMI_SERIES_ID].loc[:"2016-06-30"]
    f = FredFetcher(api_key="k", cache_dir=tmp_path / "fred", manual_dir=tmp_path / "m",
                    session=FakeFredSession(series), stale_limit_months=3)
    f.refresh()
    assert f.latest_date("ism_pmi") == pd.Timestamp("2016-06-30")
    assert f.get_actual("ism_pmi", "2016-09-30") == f.get_actual("ism_pmi", "2016-06-30")  # carried 3 months
    with pytest.raises(StaleDataError, match="stale|more than"):
        f.get_actual("ism_pmi", "2016-10-31")


def test_manual_csv_supplements_and_overrides_fred(tmp_path, raw_series):
    series = {k: v for k, v in raw_series.items() if not k.startswith("_")}
    pmi_id = config.PMI_SERIES_ID
    series[pmi_id] = series[pmi_id].loc[:"2025-06-30"]  # FRED stops in June
    manual_dir = tmp_path / "manual"
    manual_dir.mkdir()
    (manual_dir / f"{pmi_id}.csv").write_text("date,value\n2025-06-01,49.9\n2025-07-01,48.2\n2025-08-01,47.1\n")
    f = FredFetcher(api_key="k", cache_dir=tmp_path / "fred", manual_dir=manual_dir, session=FakeFredSession(series))
    f.refresh()
    assert f.get_actual("ism_pmi", "2025-08-31") == pytest.approx(47.1)
    assert f.get_actual("ism_pmi", "2025-06-30") == pytest.approx(49.9)  # manual wins on overlap
    assert f.get_actual("ism_pmi", "2025-05-31") == pytest.approx(float(raw_series[pmi_id].loc["2025-05-01"]))


def test_manual_csv_is_used_when_fred_rejects_the_series(tmp_path, raw_series):
    series = {k: v for k, v in raw_series.items() if not k.startswith("_")}
    pmi_id = config.PMI_SERIES_ID
    pmi = series.pop(pmi_id)
    manual_dir = tmp_path / "manual"
    manual_dir.mkdir()
    pmi.rename("value").rename_axis("date").to_csv(manual_dir / f"{pmi_id}.csv")
    f = FredFetcher(api_key="k", cache_dir=tmp_path / "fred", manual_dir=manual_dir, session=FakeFredSession(series))
    f.refresh()
    assert f.latest_date("ism_pmi") == pd.Timestamp("2025-08-31")


def test_training_data_is_actual_only_and_complete(fetcher):
    td = fetcher.training_data()
    assert isinstance(td, TrainingData)
    assert td.provenance == ACTUAL_PROVENANCE and td.is_actual
    assert td.feature_names == FEATURE_NAMES
    assert td.start >= pd.Timestamp(config.TRAINING_START)
    assert td.end == pd.Timestamp("2025-08-31")
    assert not td.frame.isna().any().any()
    assert td.feature_signs == {"yc_spread": 1, "core_pce_yoy": -1, f"fed_funds_chg_{config.FED_FUNDS_ROC_MONTHS}m": -1, "pmi_vs_50": 1}
    # windowing
    td5 = fetcher.training_data(window_years=5)
    assert td5.start >= pd.Timestamp("2020-08-31")
    td_end = fetcher.training_data(end="2020-12-15")
    assert td_end.end == pd.Timestamp("2020-12-31")


def test_features_match_get_actual_for_last_row(fetcher):
    feats = fetcher.features().dropna()
    last = feats.index[-1]
    assert feats.loc[last, "pmi_vs_50"] == pytest.approx(fetcher.get_actual("ism_pmi", last) - 50)
    assert feats.loc[last, "yc_spread"] == pytest.approx(fetcher.get_actual("yield_curve", last))
    n = config.FED_FUNDS_ROC_MONTHS
    prev = last - pd.offsets.MonthEnd(n)
    assert feats.loc[last, f"fed_funds_chg_{n}m"] == pytest.approx(
        fetcher.get_actual("fed_funds", last) - fetcher.get_actual("fed_funds", prev))
