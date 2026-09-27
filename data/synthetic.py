"""Synthetic 3-regime macro history and a fake FRED session.

Used by the test-suite and by *demo mode* (``MacroRegime(demo=True)``), so the
whole pipeline — refresh, refit, label, score, report — runs without a FRED key.
The generator is consistent with the transforms in :mod:`data.transforms`: the
raw series are daily yields, a Core PCE index, a Fed Funds level and a PMI.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config

START = "1990-01-31"
TRUE_TRANSMAT = np.array([[0.95, 0.04, 0.01], [0.05, 0.90, 0.05], [0.01, 0.04, 0.95]])
STATE_SPREAD = np.array([-0.4, 0.9, 1.9])      # 10Y-2Y, pp
STATE_PCE = np.array([4.0, 2.6, 1.8])          # core PCE YoY, %
STATE_FF_LEVEL = np.array([5.5, 3.0, 1.0])     # Fed Funds level each regime pulls toward, %
FF_REVERSION = 0.12                             # monthly mean-reversion speed
STATE_PMI = np.array([45.0, 50.5, 56.0])       # PMI level


def simulate_states(n: int, rng: np.random.Generator) -> np.ndarray:
    states = np.empty(n, dtype=int)
    states[0] = 1
    for t in range(1, n):
        states[t] = rng.choice(3, p=TRUE_TRANSMAT[states[t - 1]])
    return states


DEFAULT_NOISE = {"spread": 0.20, "pce": 0.0004, "ff": 0.05, "pmi": 1.5}


def synthetic_raw(seed: int = 0, end: str | None = None, pmi_series_id: str | None = None,
                  noise: dict[str, float] | None = None) -> dict[str, pd.Series]:
    """Raw FRED-style series (plus ``_states``, the true regime path, for evaluation).

    ``noise`` scales the within-regime dispersion of each series; larger values make
    the regimes overlap more (closer to real data, harder to recover).
    """
    nz = {**DEFAULT_NOISE, **(noise or {})}
    end = end or (pd.Timestamp.today().normalize() - pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")
    rng = np.random.default_rng(seed)
    months = pd.date_range(START, end, freq="ME")
    n = len(months)
    states = simulate_states(n, rng)

    ff = np.empty(n)
    ff[0] = 5.0
    for t in range(1, n):  # hikes in the bearish regime, cuts in the bullish one, no hard floor pile-up
        ff[t] = max(0.05, ff[t - 1] + FF_REVERSION * (STATE_FF_LEVEL[states[t]] - ff[t - 1]) + rng.normal(0, nz['ff']))
    dgs2_m = ff + 0.3 + rng.normal(0, 0.08, n)
    dgs10_m = dgs2_m + STATE_SPREAD[states] + rng.normal(0, nz['spread'], n)
    monthly_inf = (STATE_PCE[states] / 12.0) / 100.0 + rng.normal(0, nz['pce'], n)
    pce = 60.0 * np.cumprod(1.0 + monthly_inf)
    pmi = STATE_PMI[states] + rng.normal(0, nz['pmi'], n)

    days = pd.bdate_range(months[0] - pd.offsets.MonthBegin(1), months[-1])
    pos = pd.Series(np.arange(n), index=months.to_period("M")).reindex(days.to_period("M")).to_numpy()
    first_of_month = months - pd.offsets.MonthBegin(1)
    # --- the wider catalogue, each with a regime-dependent level plus noise ---
    def regime_series(levels, sd, smooth=0.0):
        target = np.asarray(levels)[states]
        out = np.empty(n)
        out[0] = target[0]
        for t in range(1, n):
            out[t] = smooth * out[t - 1] + (1 - smooth) * target[t] + rng.normal(0, sd)
        return out

    credit = regime_series([3.4, 2.3, 1.7], nz.get("credit", 0.15), smooth=0.5)
    unrate = regime_series([6.5, 5.0, 4.0], nz.get("unrate", 0.08), smooth=0.85)
    claims = regime_series([420_000, 330_000, 250_000], nz.get("claims", 12_000), smooth=0.6)
    payrolls_chg = regime_series([-120.0, 120.0, 220.0], nz.get("payrolls", 60.0))
    payems = 110_000 + np.cumsum(payrolls_chg)
    cpi = 100.0 * np.cumprod(1.0 + (STATE_PCE[states] + 0.5) / 1200.0 + rng.normal(0, nz["pce"] * 1.5, n))
    core_cpi = 100.0 * np.cumprod(1.0 + (STATE_PCE[states] + 0.3) / 1200.0 + rng.normal(0, nz["pce"], n))
    ppi = 100.0 * np.cumprod(1.0 + (STATE_PCE[states] - 1.0) / 1200.0 + rng.normal(0, 0.004, n))
    sentiment = regime_series([65.0, 85.0, 95.0], nz.get("sentiment", 3.0), smooth=0.5)
    services_pmi = regime_series([47.0, 52.0, 57.0], nz["pmi"])
    chicago_pmi = regime_series([43.0, 51.0, 58.0], nz["pmi"] * 1.6)
    conf_board = regime_series([70.0, 100.0, 120.0], nz.get("confidence", 4.0), smooth=0.5)
    weeks = pd.date_range(months[0] - pd.offsets.MonthBegin(1), months[-1], freq="W-SAT")
    wpos = pd.Series(np.arange(n), index=months.to_period("M")).reindex(weeks.to_period("M")).to_numpy()
    raw = {
        "BAA10Y": pd.Series(credit[pos] + rng.normal(0, 0.02, len(days)), index=days),
        "UNRATE": pd.Series(np.round(unrate, 1), index=first_of_month),
        "ICSA": pd.Series(claims[wpos] + rng.normal(0, 8_000, len(weeks)), index=weeks),
        "PAYEMS": pd.Series(payems, index=first_of_month),
        "CPIAUCSL": pd.Series(cpi, index=first_of_month),
        "CPILFESL": pd.Series(core_cpi, index=first_of_month),
        "PPIACO": pd.Series(ppi, index=first_of_month),
        "UMCSENT": pd.Series(sentiment, index=first_of_month),
        "ISM_SERVICES_PMI": pd.Series(services_pmi, index=first_of_month),
        "CHICAGO_PMI": pd.Series(chicago_pmi, index=first_of_month),
        "CONF_BOARD_CCI": pd.Series(conf_board, index=first_of_month),
        "DGS10": pd.Series(dgs10_m[pos] + rng.normal(0, 0.02, len(days)), index=days),
        "DGS2": pd.Series(dgs2_m[pos] + rng.normal(0, 0.02, len(days)), index=days),
        "PCEPILFE": pd.Series(pce, index=first_of_month),
        "FEDFUNDS": pd.Series(ff, index=first_of_month),
        (pmi_series_id or config.PMI_SERIES_ID): pd.Series(pmi, index=first_of_month),
        "_states": pd.Series(states, index=months),
    }
    return raw


def write_demo_manual_csvs(raw: dict[str, pd.Series], manual_dir) -> list:
    """Write the not-on-FRED series of a synthetic history as manual CSV supplements."""
    from pathlib import Path

    from data.transforms import MANUAL_ONLY_SERIES

    manual_dir = Path(manual_dir)
    manual_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for sid in MANUAL_ONLY_SERIES:
        if sid in raw:
            path = manual_dir / f"{sid}.csv"
            raw[sid].rename("value").rename_axis("date").to_csv(path)
            written.append(path)
    return written


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
    """Mimics ``requests.Session.get`` for the FRED observations endpoint, with pagination."""

    series: dict[str, pd.Series]
    page_limit: int | None = None
    calls: list[dict] = field(default_factory=list)
    missing_marker_every: int = 97  # sprinkle "." (FRED's missing marker) into the stream
    fail_ids: set[str] = field(default_factory=set)

    @classmethod
    def from_raw(cls, raw: dict[str, pd.Series], **kw) -> "FakeFredSession":
        return cls({k: v for k, v in raw.items() if not k.startswith("_")}, **kw)

    def get(self, url, params=None, timeout=None):
        params = dict(params or {})
        self.calls.append(params)
        sid = params["series_id"]
        if not params.get("api_key"):
            return FakeResponse({"error_code": 400, "error_message": "Bad Request. Variable api_key is not set."}, 400)
        if sid in self.fail_ids or sid not in self.series:
            return FakeResponse({"error_code": 400, "error_message": "Bad Request. The series does not exist."}, 400)
        s = self.series[sid]
        s = s[s.index >= pd.Timestamp(params.get("observation_start", "1776-07-04"))]
        rows = []
        for i, (d, v) in enumerate(s.items()):
            value = "." if (self.missing_marker_every and i % self.missing_marker_every == 5) else f"{v:.6f}"
            rows.append({"date": d.strftime("%Y-%m-%d"), "value": value})
        limit = min(int(params.get("limit", 100000)), self.page_limit or 10**9)
        offset = int(params.get("offset", 0))
        return FakeResponse({"count": len(rows), "offset": offset, "limit": limit, "observations": rows[offset : offset + limit]})
