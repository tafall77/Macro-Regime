import numpy as np
import pandas as pd
import pytest

from model.hmm_engine import (
    ACTUAL_PROVENANCE,
    SYNTHETIC_PROVENANCE,
    HMMEngine,
    NotFittedError,
    ProvenanceError,
    TrainingData,
)
from tests.conftest import TRUE_TRANSMAT


def test_training_data_validation():
    idx = pd.date_range("2020-01-31", periods=5, freq="ME")
    good = pd.DataFrame({"a": np.arange(5.0), "b": np.ones(5)}, index=idx)
    td = TrainingData(good, SYNTHETIC_PROVENANCE)
    assert td.feature_names == ("a", "b") and len(td) == 5 and td.start == idx[0] and td.end == idx[-1]
    good.iloc[0, 0] = 99  # the wrapper holds a defensive copy
    assert td.frame.iloc[0, 0] == 0.0
    with pytest.raises(ProvenanceError):
        TrainingData(good, "override_scenario")
    with pytest.raises(TypeError):
        TrainingData(good.reset_index(drop=True), SYNTHETIC_PROVENANCE)
    with pytest.raises(ValueError):
        TrainingData(good.assign(a=[1, np.nan, 3, 4, 5]), SYNTHETIC_PROVENANCE)
    with pytest.raises(ValueError):
        TrainingData(good.iloc[::-1], SYNTHETIC_PROVENANCE)
    with pytest.raises(ValueError):
        TrainingData(good.iloc[:0], SYNTHETIC_PROVENANCE)
    with pytest.raises(TypeError):
        TrainingData(good.to_numpy(), SYNTHETIC_PROVENANCE)


def test_fit_rejects_anything_but_training_data(fetcher):
    engine = HMMEngine(n_states=2, n_restarts=1, n_iter=5)
    frame = fetcher.features().dropna()
    with pytest.raises(ProvenanceError):
        engine.fit(frame)  # a bare DataFrame is not accepted
    with pytest.raises(ProvenanceError):
        engine.fit(frame.to_numpy())
    with pytest.raises(NotFittedError):
        engine.score(np.zeros(4))
    with pytest.raises(NotFittedError):
        engine.persistence(0)


def test_fit_recovers_the_synthetic_regimes(fitted_engine, true_states):
    e = fitted_engine
    assert e.fitted and e.n_states == 3 and e.fit_id and len(e.fit_id) == 10
    assert e.training_provenance == ACTUAL_PROVENANCE
    # canonical order: state 0 most bearish-looking, state 2 most bullish-looking
    means = e.state_means()
    assert means.loc[0, "pmi_vs_50"] < means.loc[1, "pmi_vs_50"] < means.loc[2, "pmi_vs_50"]
    assert means.loc[0, "yc_spread"] < means.loc[2, "yc_spread"]
    assert np.all(np.diff(e.state_scores_) >= 0)
    # the filtered path should agree with the true simulated regime most of the time
    decoded = e.filtered_.to_numpy().argmax(axis=1)
    truth = true_states.reindex(e.filtered_.index).to_numpy()
    assert (decoded == truth).mean() > 0.85
    # transition matrix diagonal should be sticky like the generator
    assert np.all(np.diag(e.transmat_) > 0.8)
    assert e.transmat_.sum(axis=1) == pytest.approx(np.ones(3))
    assert np.abs(np.diag(e.transmat_) - np.diag(TRUE_TRANSMAT)).max() < 0.1


def test_score_returns_distribution_and_matches_hmmlearn(fitted_engine):
    e = fitted_engine
    frame = e.filtered_  # dates
    # 1) undated vector: one step forward from the last fitted posterior
    x = np.array([1.5, 1.9, -0.2, 5.0])
    p = e.score(x)
    assert p.shape == (3,) and p.sum() == pytest.approx(1.0) and (p >= 0).all()
    assert e.score(x[None, :]) == pytest.approx(p)
    assert e.score(pd.Series(x, index=e.feature_names)) == pytest.approx(p)
    # 2) scoring the full training path reproduces the stored filtered posteriors exactly
    hist = _original_training_frame(e)
    path = e.score_path(hist)
    np.testing.assert_allclose(path.to_numpy(), e.filtered_.to_numpy(), atol=1e-8)
    # and the final smoothed posterior from hmmlearn equals the final filtered posterior
    Zs = (hist.to_numpy() - e.mean_) / e.scale_
    smoothed = e._model.predict_proba(Zs)
    np.testing.assert_allclose(smoothed[-1], e.filtered_.iloc[-1].to_numpy(), atol=1e-6)
    # 3) dated re-score of just the last row starts from the previous month's posterior
    p_last = e.score(hist.iloc[[-1]])
    np.testing.assert_allclose(p_last, e.filtered_.iloc[-1].to_numpy(), atol=1e-8)
    # 4) dated row before training start uses the initial distribution
    early = hist.iloc[[0]].copy()
    early.index = [hist.index[0] - pd.offsets.MonthEnd(12)]
    np.testing.assert_allclose(e.score(early), e.filtered_.iloc[0].to_numpy(), atol=1e-8)


def _original_training_frame(e: HMMEngine) -> pd.DataFrame:
    """The engine caches standardised filtered posteriors, not X; rebuild X from the fetcher cache."""
    # The session-scoped engine was fitted on the synthetic fetcher; we regenerate the same data.
    from tests.conftest import FakeFredSession, synthetic_raw
    from data.fred_fetcher import FredFetcher
    import tempfile
    raw = synthetic_raw(seed=0)
    with tempfile.TemporaryDirectory() as d:
        f = FredFetcher(api_key="k", cache_dir=f"{d}/fred", manual_dir=f"{d}/m",
                        session=FakeFredSession({k: v for k, v in raw.items() if not k.startswith("_")}))
        f.refresh()
        return f.training_data().frame


def test_score_input_validation(fitted_engine):
    e = fitted_engine
    with pytest.raises(ValueError):
        e.score(np.zeros(3))
    with pytest.raises(ValueError):
        e.score(np.array([np.nan, 0, 0, 0]))
    with pytest.raises(KeyError):
        e.score(pd.DataFrame({"yc_spread": [1.0]}, index=[pd.Timestamp("2025-08-31")]))
    # extra columns are fine and order does not matter
    df = pd.DataFrame({n: [0.0] for n in reversed(e.feature_names)} | {"extra": [1.0]}, index=[pd.Timestamp("2025-09-30")])
    assert e.score(df).sum() == pytest.approx(1.0)


def test_scenario_moves_probability_in_the_expected_direction(fitted_engine):
    e = fitted_engine
    bearish = np.array([-0.5, 4.0, 0.4, -5.0])   # inverted curve, hot inflation, hiking, PMI 45
    bullish = np.array([1.9, 1.8, -0.3, 6.0])    # steep curve, cool inflation, cutting, PMI 56
    assert e.score(bearish).argmax() == 0
    assert e.score(bullish).argmax() == 2
    # feeding the same bullish row repeatedly converges the posterior on the bullish state
    p = e.score(np.tile(bullish, (6, 1)))
    assert p[2] > 0.95


def test_persistence_and_stationary_distribution(fitted_engine):
    e = fitted_engine
    for k in range(3):
        assert e.persistence(k) == pytest.approx(1.0 / (1.0 - e.transmat_[k, k]))
        assert e.persistence(k) > 5  # sticky regimes
    pi = e.stationary_distribution()
    assert pi.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(pi @ e.transmat_, pi, atol=1e-8)
    e2 = HMMEngine(n_states=2)
    e2.fitted = True
    e2.transmat_ = np.array([[1.0, 0.0], [0.5, 0.5]])
    assert e2.persistence(0) == float("inf") and e2.persistence(1) == 2.0


def test_describe_and_metadata(fitted_engine):
    d = fitted_engine.describe()
    assert list(d.index) == [0, 1, 2]
    for col in ("persistence_months", "stationary_prob", "train_share", "growth_score"):
        assert col in d.columns
    assert d["train_share"].sum() == pytest.approx(1.0)
    meta = fitted_engine.to_metadata()
    assert meta["fit_id"] == fitted_engine.fit_id and meta["n_train"] == fitted_engine.n_train
    assert meta["training_provenance"] == ACTUAL_PROVENANCE


def test_save_load_round_trip(fitted_engine, tmp_path):
    path = fitted_engine.save(tmp_path / "m" / "hmm.pkl")
    assert path.exists() and path.with_suffix(".json").exists()
    loaded = HMMEngine.load(path)
    assert loaded.fit_id == fitted_engine.fit_id
    x = np.array([0.5, 2.5, 0.0, 1.0])
    np.testing.assert_allclose(loaded.score(x), fitted_engine.score(x))
    with pytest.raises(NotFittedError):
        HMMEngine.load(tmp_path / "missing.pkl")


def test_two_state_engine_and_min_rows(fetcher):
    td = fetcher.training_data(window_years=8)
    e = HMMEngine(n_states=2, n_restarts=1, n_iter=100).fit(td)
    assert e.score(np.zeros(4)).shape == (2,)
    with pytest.raises(ValueError):
        HMMEngine(n_states=2).fit(TrainingData(td.frame.iloc[:10], SYNTHETIC_PROVENANCE))
    with pytest.raises(ValueError):
        HMMEngine(n_states=1)
