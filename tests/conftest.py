"""Shared fixtures: a fake FRED API serving a synthetic 3-regime macro history."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

import config
from data.fred_fetcher import FredFetcher
from data.overrides import OverrideStore
from model.hmm_engine import HMMEngine
from model.labeler import StateLabeler

START, END = "1990-01-31", "2025-08-31"
TRUE_TRANSMAT = np.array([[0.95, 0.04, 0.01], [0.05, 0.90, 0.05], [0.01, 0.04, 0.95]])
# per-state targets: spread (pp), core pce yoy (%), fed funds monthly drift (pp), pmi (index)
STATE_SPREAD = np.array([-0.4, 0.9, 1.9])
STATE_PCE = np.array([4.0, 2.6, 1.8])
STATE_FF_DRIFT = np.array([0.12, 0.0, -0.10])
STATE_PMI = np.array([45.0, 50.5, 56.0])


def simulate_states(n: int, rng: np.random.Generator) -> np.ndarray:
    states = np.empty(n, dtype=int)
    states[0] = 1
    for t in range(1, n):
        states[t] = rng.choice(3, p=TRUE_TRANSMAT[states[t - 1]])
    return states


def synthetic_raw(seed: int = 0, end: str = END, pmi_series_id: str | None = None) -> dict[str, pd.Series]:
    """Raw FRED-style series consistent with the transforms in data.transforms."""
    rng = np.random.default_rng(seed)
    months = pd.date_range(START, end, freq="ME")
    n = len(months)
    states = simulate_states(n, rng)

    # Fed funds: level path driven by the state drift, clipped at zero
    ff = np.empty(n)
    ff[0] = 5.0
    for t in range(1, n):
        ff[t] = max(0.05, ff[t - 1] + STATE_FF_DRIFT[states[t]] + rng.normal(0, 0.03))
    # 2Y ~ fed funds + noise; 10Y = 2Y + state spread
    dgs2_m = ff + 0.3 + rng.normal(0, 0.08, n)
    dgs10_m = dgs2_m + STATE_SPREAD[states] + rng.normal(0, 0.12, n)
    # Core PCE index whose YoY inflation tracks the state target
    monthly_inf = (STATE_PCE[states] / 12.0) / 100.0 + rng.normal(0, 0.0003, n)
    pce = 60.0 * np.cumprod(1.0 + monthly_inf)
    pmi = STATE_PMI[states] + rng.normal(0, 1.2, n)

    # Daily business-day series for the yields (forward-filled month values + noise)
    days = pd.bdate_range(months[0] - pd.offsets.MonthBegin(1), months[-1])
    m_index = days.to_period("M")
    month_pos = pd.Series(np.arange(n), index=months.to_period("M"))
    pos = month_pos.reindex(m_index).to_numpy()
    dgs10_d = dgs10_m[pos] + rng.normal(0, 0.02, len(days))
    dgs2_d = dgs2_m[pos] + rng.normal(0, 0.02, len(days))

    first_of_month = months - pd.offsets.MonthBegin(1)
    raw = {
        "DGS10": pd.Series(dgs10_d, index=days),
        "DGS2": pd.Series(dgs2_d, index=days),
        "PCEPILFE": pd.Series(pce, index=first_of_month),
        "FEDFUNDS": pd.Series(ff, index=first_of_month),
        (pmi_series_id or config.PMI_SERIES_ID): pd.Series(pmi, index=first_of_month),
    }
    raw["_states"] = pd.Series(states, index=months)
    return raw


@dataclass
class FakeResponse:
    payload: dict
    status_code: int = 200

    def json(self):
        return self.payload

    @property
    def text(self):
        return json.dumps(self.payload)

    def raise_for_status(self):
        pass


@dataclass
class FakeFredSession:
    """Mimics requests.Session.get for the FRED observations endpoint, with pagination."""

    series: dict[str, pd.Series]
    page_limit: int | None = None  # force server-side paging when set
    calls: list[dict] = field(default_factory=list)
    missing_marker_every: int = 97  # sprinkle "." (FRED's missing value) into the stream
    fail_ids: set[str] = field(default_factory=set)

    def get(self, url, params=None, timeout=None):
        params = dict(params or {})
        self.calls.append(params)
        sid = params["series_id"]
        if not params.get("api_key"):
            return FakeResponse({"error_code": 400, "error_message": "Bad Request. Variable api_key is not set."}, 400)
        if sid in self.fail_ids or sid not in self.series:
            return FakeResponse({"error_code": 400, "error_message": "Bad Request. The series does not exist."}, 400)
        s = self.series[sid]
        start = pd.Timestamp(params.get("observation_start", "1776-07-04"))
        s = s[s.index >= start]
        rows = []
        for i, (d, v) in enumerate(s.items()):
            value = "." if (self.missing_marker_every and i % self.missing_marker_every == 5) else f"{v:.6f}"
            rows.append({"date": d.strftime("%Y-%m-%d"), "value": value})
        limit = min(int(params.get("limit", 100000)), self.page_limit or 10**9)
        offset = int(params.get("offset", 0))
        page = rows[offset : offset + limit]
        return FakeResponse({"count": len(rows), "offset": offset, "limit": limit, "observations": page})


@pytest.fixture(scope="session")
def raw_series() -> dict[str, pd.Series]:
    return synthetic_raw(seed=0)


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
