"""Shared fixtures: a fake FRED API serving a synthetic 3-regime macro history."""
from __future__ import annotations

import pandas as pd
import pytest

import config
from data.fred_fetcher import FredFetcher
from data.overrides import OverrideStore
from model.hmm_engine import HMMEngine
from model.labeler import StateLabeler

from data.synthetic import (  # noqa: F401  (re-exported for the tests)
    STATE_FF_LEVEL, STATE_PCE, STATE_PMI, STATE_SPREAD, TRUE_TRANSMAT, FakeFredSession, FakeResponse, simulate_states,
)

END = "2025-08-31"


def synthetic_raw(seed: int = 0, end: str = END, pmi_series_id: str | None = None):
    from data import synthetic
    return synthetic.synthetic_raw(seed=seed, end=end, pmi_series_id=pmi_series_id)


@pytest.fixture(scope="session")
def raw_series() -> dict[str, pd.Series]:
    return synthetic_raw(seed=1)


@pytest.fixture(scope="session")
def true_states(raw_series) -> pd.Series:
    return raw_series["_states"]


@pytest.fixture
def fake_session(raw_series) -> FakeFredSession:
    return FakeFredSession({k: v for k, v in raw_series.items() if not k.startswith("_")})


@pytest.fixture
def fetcher(tmp_path, fake_session) -> FredFetcher:
    f = FredFetcher(api_key="test-key", cache_dir=tmp_path / "fred", manual_dir=tmp_path / "manual",
                    session=fake_session, observation_start="1985-01-01")
    f.refresh()
    return f


@pytest.fixture
def overrides(tmp_path, fetcher) -> OverrideStore:
    return OverrideStore(fetcher, tmp_path / "overrides.json")


@pytest.fixture(scope="session")
def fitted_engine(tmp_path_factory, raw_series) -> HMMEngine:
    """One fitted engine per session (EM is the slow part)."""
    base = tmp_path_factory.mktemp("engine")
    session = FakeFredSession({k: v for k, v in raw_series.items() if not k.startswith("_")})
    f = FredFetcher(api_key="test-key", cache_dir=base / "fred", manual_dir=base / "manual", session=session)
    f.refresh()
    engine = HMMEngine(n_states=3, n_restarts=2, n_iter=200, random_state=3)
    engine.fit(f.training_data())
    return engine


@pytest.fixture
def labeler(tmp_path) -> StateLabeler:
    return StateLabeler(tmp_path / "labels.json")
