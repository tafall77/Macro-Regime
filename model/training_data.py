"""The typed container that is the *only* thing :meth:`HMMEngine.fit` accepts.

The point of this indirection is provenance: the override store can produce
numbers, but it cannot produce a :class:`TrainingData`. The single production
factory is :meth:`data.fred_fetcher.FredFetcher.training_data`, which reads the
actual FRED cache and nothing else. ``fit()`` refuses anything that is not a
``TrainingData`` carrying an allowed provenance, so a hypothetical value can only
reach the walk-forward refit by someone deliberately forging the wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd

ACTUAL_PROVENANCE = "fred_actual"
SYNTHETIC_PROVENANCE = "synthetic"  # for tests and offline experiments only
ALLOWED_PROVENANCE = frozenset({ACTUAL_PROVENANCE, SYNTHETIC_PROVENANCE})


class ProvenanceError(TypeError):
    """Raised when training data does not come from the actual-data path."""


@dataclass(frozen=True, eq=False)
class TrainingData:
    frame: pd.DataFrame
    provenance: str
    source: str = ""
    feature_signs: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.frame, pd.DataFrame):
            raise TypeError("TrainingData.frame must be a pandas DataFrame")
        if self.provenance not in ALLOWED_PROVENANCE:
            raise ProvenanceError(
                f"provenance {self.provenance!r} is not allowed; expected one of {sorted(ALLOWED_PROVENANCE)}"
            )
        if not isinstance(self.frame.index, pd.DatetimeIndex):
            raise TypeError("TrainingData.frame must be indexed by date")
        if self.frame.empty:
            raise ValueError("TrainingData.frame is empty")
        if not self.frame.index.is_monotonic_increasing:
            raise ValueError("TrainingData.frame index must be sorted ascending")
        if self.frame.index.has_duplicates:
            raise ValueError("TrainingData.frame index has duplicate dates")
        if self.frame.isna().any().any():
            raise ValueError("TrainingData.frame contains NaN; drop incomplete rows first")
        # defensive copy so later mutation of the caller's frame cannot alter the training set
        object.__setattr__(self, "frame", self.frame.astype("float64").copy())
        object.__setattr__(self, "feature_signs", dict(self.feature_signs))

    @property
    def is_actual(self) -> bool:
        return self.provenance == ACTUAL_PROVENANCE

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(str(c) for c in self.frame.columns)

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.frame.index[0])

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp(self.frame.index[-1])

    def __len__(self) -> int:
        return len(self.frame)

    def __repr__(self) -> str:
        return (
            f"TrainingData(rows={len(self)}, features={list(self.feature_names)}, "
            f"{self.start.date()}..{self.end.date()}, provenance={self.provenance!r})"
        )
