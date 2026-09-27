import numpy as np
import pandas as pd
import pytest

import config
from data import transforms as T


def test_month_end_and_months_between():
    assert T.month_end("2024-03-15") == pd.Timestamp("2024-03-31")
    assert T.month_end(pd.Timestamp("2024-02-29")) == pd.Timestamp("2024-02-29")
    assert T.months_between("2024-01-31", "2024-04-30") == 3
    assert T.months_between("2023-11-30", "2024-01-31") == 2


def test_to_month_end_keeps_gap_months_as_nan():
    idx = pd.to_datetime(["2024-01-02", "2024-01-31", "2024-03-05"])
    s = pd.Series([1.0, 2.0, 3.0], index=idx)
    m = T.to_month_end(s)
    assert list(m.index) == list(pd.date_range("2024-01-31", "2024-03-31", freq="ME"))
    assert m.iloc[0] == 2.0  # last obs of January
    assert np.isnan(m.iloc[1])
    assert m.iloc[2] == 3.0


def test_yield_spread_is_10y_minus_2y():
    idx = pd.date_range("2024-01-31", periods=3, freq="ME")
    m = {"DGS10": pd.Series([4.0, 4.1, 4.2], idx), "DGS2": pd.Series([4.5, 4.0, 3.5], idx)}
    np.testing.assert_allclose(T.yield_spread(m).to_numpy(), [-0.5, 0.1, 0.7])


def test_core_pce_yoy_is_12_month_pct_change():
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    level = pd.Series(100.0 * (1.03) ** (np.arange(24) / 12), idx)  # exactly 3%/yr
    yoy = T.core_pce_yoy({"PCEPILFE": level})
    assert yoy.iloc[:12].isna().all()
    np.testing.assert_allclose(yoy.iloc[12:].to_numpy(), 3.0, atol=1e-9)


def test_fed_funds_feature_is_n_month_change_and_pmi_is_level_minus_50():
    idx = pd.date_range("2024-01-31", periods=6, freq="ME")
    ff = pd.Series([5.0, 5.0, 5.25, 5.5, 5.5, 5.25], idx)
    spec = T.get_spec("fed_funds")
    feat = spec.feature(ff)
    n = config.FED_FUNDS_ROC_MONTHS
    assert feat.iloc[:n].isna().all()
    np.testing.assert_allclose(feat.iloc[n:].to_numpy(), (ff - ff.shift(n)).iloc[n:].to_numpy())
    pmi = T.get_spec("ism_pmi").feature(pd.Series([47.0, 53.0], idx[:2]))
    np.testing.assert_allclose(pmi.to_numpy(), [-3.0, 3.0])


def test_registry_shape():
    assert T.INDICATOR_KEYS == ("yield_curve", "core_pce", "fed_funds", "ism_pmi")
    assert len(T.FEATURE_NAMES) == 4
    assert set(T.ALL_SERIES_IDS) == {"DGS10", "DGS2", "PCEPILFE", "FEDFUNDS", config.PMI_SERIES_ID}
    with pytest.raises(KeyError):
        T.get_spec("nope")


def test_features_from_panel_reindexes_to_full_months():
    idx = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-04-30"])  # March missing
    panel = pd.DataFrame(
        {"yield_curve": [1.0, 1.1, 1.2], "core_pce": [2.0, 2.1, 2.2], "fed_funds": [5.0, 5.0, 5.5], "ism_pmi": [50, 51, 52]},
        index=idx,
    )
    feats = T.features_from_panel(panel)
    assert len(feats) == 4
    assert list(feats.columns) == list(T.FEATURE_NAMES)
    assert np.isnan(feats.loc["2024-03-31", "yc_spread"])
    assert feats.loc["2024-04-30", "pmi_vs_50"] == 2.0
