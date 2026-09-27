"""Which indicators earn their place? A walk-forward ablation.

For a base set B and each candidate c, fit the HMM walk-forward on B and on
B ∪ {c} and compare what comes out *out of sample*:

* ``relevance_f``   — one-way F statistic of the candidate feature across the
  regimes the **base** model already finds (does c differ between regimes at
  all, or is it noise with respect to them?).
* ``redundancy_r2`` — how much of c the base features already explain
  (a linear R²; high means c adds little that is new).
* ``confidence``    — mean max posterior of the B ∪ {c} model, out of sample
  (sharper reads), against the base's ``confidence_base``.
* ``switches_per_year`` — regime switches per year out of sample (a noisier
  feature set flips more), against the base.
* ``agreement``     — share of months where the two models decode the same
  regime (stability of the story).
* ``keep``          — heuristic recommendation: relevant, not redundant, and no
  worse than the base on confidence or switching.

This is a screening tool, not a proof. Run it on real data before widening
``MACRO_REGIME_INDICATORS``; every number here is a heuristic.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config
from data.fred_fetcher import FredFetcher
from data.transforms import get_spec, select
from model.backtest import BacktestResult, walk_forward
from model.training_data import TrainingData

log = logging.getLogger(__name__)


def _f_stat(x: np.ndarray, groups: np.ndarray) -> float:
    ids = np.unique(groups)
    if len(ids) < 2:
        return float("nan")
    grand = x.mean()
    ssb = sum(((x[groups == g].mean() - grand) ** 2) * (groups == g).sum() for g in ids)
    ssw = sum(((x[groups == g] - x[groups == g].mean()) ** 2).sum() for g in ids)
    dfb, dfw = len(ids) - 1, len(x) - len(ids)
    return float((ssb / dfb) / (ssw / dfw)) if ssw > 0 else float("inf")


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    tot = ((y - y.mean()) ** 2).sum()
    return float(1 - resid @ resid / tot) if tot > 0 else float("nan")


def _switches_per_year(decoded: pd.Series) -> float:
    years = max((decoded.index[-1] - decoded.index[0]).days / 365.25, 1e-9)
    return float((decoded != decoded.shift()).sum() - 1) / years


def _run(training: TrainingData, cols: list[str], n_states: int, **wf) -> BacktestResult:
    sub = TrainingData(training.frame[cols].dropna(), training.provenance, training.source,
                       {c: training.feature_signs.get(c, 0) for c in cols})
    return walk_forward(sub, n_states=n_states, **wf)


def evaluate_indicators(
    fetcher: FredFetcher,
    base: list[str],
    candidates: list[str],
    n_states: int = config.N_STATES,
    refit_every: int = 12,
    min_train: int = 120,
    n_restarts: int = 2,
    n_iter: int = 200,
) -> pd.DataFrame:
    """Score each candidate against the base set. Returns one row per candidate, best first."""
    training = fetcher.training_data()
    base_feats = [s.feature_name for s in select(base)]
    wf = dict(refit_every=refit_every, min_train=min_train, n_restarts=n_restarts, n_iter=n_iter)
    base_bt = _run(training, base_feats, n_states, **wf)
    base_dec = base_bt.decoded
    base_conf = float(base_bt.posteriors.max(axis=1).mean())
    base_sw = _switches_per_year(base_dec)
    frame = training.frame

    rows = []
    for key in candidates:
        spec = get_spec(key)
        feat = spec.feature_name
        common = frame.index.intersection(base_dec.index)
        x = frame.loc[common, feat].to_numpy()
        relevance = _f_stat(x, base_dec.loc[common].to_numpy())
        redundancy = _r2(x, frame.loc[common, base_feats].to_numpy())
        try:
            bt = _run(training, base_feats + [feat], n_states, **wf)
            both = base_dec.index.intersection(bt.decoded.index)
            agreement = float((bt.decoded.loc[both] == base_dec.loc[both]).mean())
            conf = float(bt.posteriors.max(axis=1).mean())
            sw = _switches_per_year(bt.decoded)
        except Exception as exc:  # noqa: BLE001 - report the failure in the table rather than abort
            log.warning("candidate %s failed: %s", key, exc)
            agreement = conf = sw = float("nan")
        keep = bool(relevance >= 10 and redundancy <= 0.8 and conf >= base_conf - 0.02 and sw <= base_sw * 1.25 + 0.1)
        rows.append({
            "indicator": key, "category": spec.category, "feature": feat, "relevance_f": relevance,
            "redundancy_r2": redundancy, "confidence": conf, "confidence_base": base_conf,
            "switches_per_year": sw, "switches_base": base_sw, "agreement": agreement, "keep": keep,
        })
    out = pd.DataFrame(rows).set_index("indicator")
    return out.sort_values(["keep", "relevance_f"], ascending=[False, False])
