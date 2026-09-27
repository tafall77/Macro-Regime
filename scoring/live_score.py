"""Live scoring: the one cheap call the dashboard makes on refresh or override change.

It pulls the latest actual headline values from the FRED cache, builds the
feature rows since the last training month, scores them with the fitted HMM,
then does the same with the override store's hypothetical values layered onto
the *latest* month only, and returns both state distributions plus persistence.

No fitting happens here. The engine is loaded from disk and reloaded only when
the monthly refit job writes a new file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from data.fred_fetcher import FredFetcher, StaleDataError
from data.overrides import OverrideStore
from data.transforms import INDICATORS, features_from_panel
from model.hmm_engine import HMMEngine
from model.labeler import LabelSet, StateLabeler


@dataclass(frozen=True)
class IndicatorSnapshot:
    key: str
    name: str
    headline_label: str
    unit: str
    actual: float
    actual_date: pd.Timestamp
    effective: float
    overridden: bool
    feature_name: str
    feature_unit: str
    feature_actual: float
    feature_scenario: float


@dataclass(frozen=True)
class LiveScore:
    as_of: pd.Timestamp
    fit_id: str
    training_end: pd.Timestamp
    fitted_at: str
    indicators: tuple[IndicatorSnapshot, ...]
    probs_actual: np.ndarray
    probs_scenario: np.ndarray
    labels: LabelSet
    persistence: np.ndarray  # expected months per state
    state_table: pd.DataFrame = field(repr=False)
    refreshed_at: str | None = None

    # -- convenience ---------------------------------------------------------
    @property
    def n_states(self) -> int:
        return len(self.probs_actual)

    @property
    def has_overrides(self) -> bool:
        return any(i.overridden for i in self.indicators)

    @property
    def top_actual(self) -> int:
        return int(np.argmax(self.probs_actual))

    @property
    def top_scenario(self) -> int:
        return int(np.argmax(self.probs_scenario))

    def prob(self, state: int, scenario: bool = False) -> float:
        arr = self.probs_scenario if scenario else self.probs_actual
        return float(arr[int(state)])

    def delta(self, state: int) -> float:
        """Scenario minus actual probability for ``state``."""
        return self.prob(state, scenario=True) - self.prob(state)

    @property
    def persistence_actual(self) -> float:
        return float(self.persistence[self.top_actual])

    @property
    def persistence_scenario(self) -> float:
        return float(self.persistence[self.top_scenario])

    def label(self, state: int) -> str:
        return self.labels.name(state)

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": str(self.as_of.date()),
            "fit_id": self.fit_id,
            "training_end": str(self.training_end.date()),
            "fitted_at": self.fitted_at,
            "labels": {k: v for k, v in self.labels.labels.items()},
            "labels_provisional": self.labels.provisional,
            "indicators": {
                i.key: {
                    "actual": i.actual,
                    "actual_date": str(i.actual_date.date()),
                    "effective": i.effective,
                    "overridden": i.overridden,
                    "feature_actual": i.feature_actual,
                    "feature_scenario": i.feature_scenario,
                }
                for i in self.indicators
            },
            "score_actual": {self.label(k): self.prob(k) for k in range(self.n_states)},
            "score_scenario": {self.label(k): self.prob(k, True) for k in range(self.n_states)},
            "top_actual": self.label(self.top_actual),
            "top_scenario": self.label(self.top_scenario),
            "persistence_actual_months": self.persistence_actual,
            "persistence_scenario_months": self.persistence_scenario,
        }


class EngineLoader:
    """Loads the fitted engine from disk; reloads when the scheduler writes a new file."""

    def __init__(self, path: str | Path = config.MODEL_PATH) -> None:
        self.path = Path(path)
        self._engine: HMMEngine | None = None
        self._mtime: float | None = None

    def get(self) -> HMMEngine:
        mtime = os.path.getmtime(self.path) if self.path.exists() else None
        if self._engine is None or mtime != self._mtime:
            self._engine = HMMEngine.load(self.path)
            self._mtime = mtime
        return self._engine


def actual_panel_asof(fetcher: FredFetcher) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Headline panel carried forward (within the stale limit) to the latest month.

    Lagging indicators (Core PCE, PMI) are forward-filled so every indicator has a
    value on the latest month; anything older than the stale limit stays NaN and
    is reported as :class:`StaleDataError` by the caller.
    """
    panel = fetcher.panel().dropna(how="all")
    as_of = pd.Timestamp(panel.index[-1])
    return panel.ffill(limit=fetcher.stale_limit_months), as_of


def scenario_panel(fetcher: FredFetcher, overrides: OverrideStore) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """(actual panel, scenario panel, as_of) with overrides applied to the latest month only."""
    actual, as_of = actual_panel_asof(fetcher)
    scenario = actual.copy()
    for spec in INDICATORS:
        if overrides.is_overridden(spec.key):
            scenario.loc[as_of, spec.key] = overrides.get_effective(spec.key, as_of)
    return actual, scenario, as_of


def _live_rows(features: pd.DataFrame, training_end: pd.Timestamp, as_of: pd.Timestamp) -> pd.DataFrame:
    """Rows the forward filter must step through: everything after the last training month,
    or just the latest month when it is (re)scoring the final training month."""
    feats = features.loc[:as_of]
    rows = feats.loc[feats.index > training_end]
    if rows.empty:
        rows = feats.iloc[[-1]]
    if rows.isna().any().any():
        bad = sorted(rows.columns[rows.isna().any()].tolist())
        raise StaleDataError(
            f"features {bad} are missing/stale as of {as_of.date()}; refresh FRED or supply a manual series"
        )
    return rows


def live_score(
    fetcher: FredFetcher,
    overrides: OverrideStore,
    engine: HMMEngine,
    labeler: StateLabeler | None = None,
) -> LiveScore:
    """Score the latest actual data and the override scenario. Cheap; never refits."""
    actual_p, scenario_p, as_of = scenario_panel(fetcher, overrides)
    f_actual = features_from_panel(actual_p, fetcher.indicators)
    f_scenario = features_from_panel(scenario_p, fetcher.indicators)

    rows_actual = _live_rows(f_actual, engine.training_end, as_of)
    rows_scenario = f_scenario.loc[rows_actual.index]

    probs_actual = engine.score(rows_actual)
    probs_scenario = engine.score(rows_scenario)

    snapshots = []
    for spec in fetcher.indicators:
        actual_date = fetcher.latest_date(spec.key)
        actual_val = float(actual_p.loc[as_of, spec.key])
        snapshots.append(
            IndicatorSnapshot(
                key=spec.key,
                name=spec.name,
                headline_label=spec.headline_label,
                unit=spec.headline_unit,
                actual=actual_val,
                actual_date=actual_date,
                effective=float(scenario_p.loc[as_of, spec.key]),
                overridden=overrides.is_overridden(spec.key),
                feature_name=spec.feature_name,
                feature_unit=spec.feature_unit,
                feature_actual=float(rows_actual.iloc[-1][spec.feature_name]),
                feature_scenario=float(rows_scenario.iloc[-1][spec.feature_name]),
            )
        )

    labels = (labeler or StateLabeler()).resolve(engine)
    return LiveScore(
        as_of=as_of,
        fit_id=engine.fit_id,
        training_end=engine.training_end,
        fitted_at=engine.fitted_at.isoformat(timespec="seconds"),
        indicators=tuple(snapshots),
        probs_actual=probs_actual,
        probs_scenario=probs_scenario,
        labels=labels,
        persistence=engine.persistences(),
        state_table=engine.state_means(),
        refreshed_at=fetcher.meta().get("refreshed_at"),
    )
