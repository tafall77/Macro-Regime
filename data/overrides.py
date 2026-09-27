"""Scenario overrides: hypothetical headline values layered over the actual data.

The store is a plain ``{indicator: value}`` mapping, optionally persisted to a
small JSON session file so a dashboard restart keeps the scenario. Overrides are
expressed in **headline units** (see :mod:`data.transforms`) and only ever apply
to the *latest* available month of an indicator: :meth:`OverrideStore.get_effective`
returns the override for the latest date (or later) and the actual value for any
earlier date.

This module deliberately has no way to produce a feature matrix or a
:class:`model.hmm_engine.TrainingData`; the training path in
:mod:`scheduler` never imports it for data.
"""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Protocol

import pandas as pd

import config
from data.transforms import get_spec, month_end


class ActualSource(Protocol):
    """What the store needs from the actual-data side (satisfied by FredFetcher)."""

    def get_actual(self, indicator: str, date) -> float: ...

    def latest_date(self, indicator: str | None = None) -> pd.Timestamp: ...


class OverrideStore:
    def __init__(self, actual_source: ActualSource | None = None, path: str | Path | None = None) -> None:
        self._actual = actual_source
        self._path = Path(path) if path is not None else None
        self._overrides: dict[str, float] = {}
        self._lock = threading.RLock()
        self._load()

    # ------------------------------------------------------------ persistence
    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
        except (OSError, ValueError):
            return
        for key, value in (data.get("overrides") or {}).items():
            try:
                self._overrides[key] = self._validate(key, value)
            except (KeyError, ValueError):
                continue  # ignore junk from an old session file

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"overrides": dict(self._overrides)}
        self._path.write_text(json.dumps(payload, indent=2))

    # --------------------------------------------------------------- mutation
    @staticmethod
    def _validate(indicator: str, value) -> float:
        get_spec(indicator)
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{indicator}: override must be numeric, got {value!r}") from None
        if not math.isfinite(v):
            raise ValueError(f"{indicator}: override must be finite, got {value!r}")
        return v

    def set_override(self, indicator: str, value) -> float:
        with self._lock:
            v = self._validate(indicator, value)
            self._overrides[indicator] = v
            self._save()
            return v

    def clear_override(self, indicator: str) -> None:
        with self._lock:
            get_spec(indicator)
            if self._overrides.pop(indicator, None) is not None:
                self._save()

    def clear_all(self) -> None:
        with self._lock:
            if self._overrides:
                self._overrides.clear()
                self._save()

    # ---------------------------------------------------------------- reading
    def overrides(self) -> dict[str, float]:
        with self._lock:
            return dict(self._overrides)

    def is_overridden(self, indicator: str) -> bool:
        return indicator in self._overrides

    def get_override(self, indicator: str) -> float | None:
        return self._overrides.get(indicator)

    def __len__(self) -> int:
        return len(self._overrides)

    def __contains__(self, indicator: str) -> bool:
        return indicator in self._overrides

    def get_effective(self, indicator: str, date) -> float:
        """Override if one is set and ``date`` is the indicator's latest month (or later); else actual."""
        if self._actual is None:
            raise RuntimeError("OverrideStore has no actual_source; pass a FredFetcher")
        get_spec(indicator)
        override = self._overrides.get(indicator)
        if override is not None and month_end(date) >= self._actual.latest_date(indicator):
            return override
        return self._actual.get_actual(indicator, date)


_default_store: OverrideStore | None = None


def default_store() -> OverrideStore:
    """Process-wide store backed by the session file in :mod:`config`."""
    global _default_store
    if _default_store is None:
        from data.fred_fetcher import default_fetcher  # local import keeps the layering one-way

        _default_store = OverrideStore(default_fetcher(), config.OVERRIDES_PATH)
    return _default_store


def set_override(indicator: str, value) -> float:
    return default_store().set_override(indicator, value)


def clear_override(indicator: str) -> None:
    default_store().clear_override(indicator)


def clear_all() -> None:
    default_store().clear_all()


def get_effective(indicator: str, date) -> float:
    return default_store().get_effective(indicator, date)
