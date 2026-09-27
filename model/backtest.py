"""Walk-forward (out-of-sample) evaluation of the regime engine.

At each refit point the engine is fitted on data strictly before it and then
*filters* forward through the next ``refit_every`` months without seeing them
in EM, exactly like the monthly production loop. The result is an honest
out-of-sample posterior for every month after ``min_train``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from model.hmm_engine import HMMEngine
from model.training_data import TrainingData

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    posteriors: pd.DataFrame     # out-of-sample filtered probabilities, one column per state
    fits: pd.DataFrame           # one row per refit: start, end, n_train, fit_id, log-likelihood
    n_states: int

    @property
    def decoded(self) -> pd.Series:
        return pd.Series(self.posteriors.to_numpy().argmax(axis=1), index=self.posteriors.index, name="state")

    def regime_shares(self) -> pd.Series:
        return self.decoded.value_counts(normalize=True).sort_index()

    def average_run_length(self) -> pd.Series:
        """Mean length (months) of consecutive runs, per state — the empirical persistence."""
        s = self.decoded
        runs = (s != s.shift()).cumsum()
        lengths = s.groupby(runs).agg(["first", "size"])
        return lengths.groupby("first")["size"].mean().rename("avg_run_months")

    def accuracy(self, truth: pd.Series) -> float:
        """Share of months whose decoded state matches ``truth`` (synthetic data only)."""
        t = truth.reindex(self.posteriors.index)
        mask = t.notna()
        return float((self.decoded[mask] == t[mask].astype(int)).mean())

    def summary(self) -> pd.DataFrame:
        df = pd.DataFrame({"share": self.regime_shares(), "avg_run_months": self.average_run_length()})
        df.index.name = "state"
        return df


def walk_forward(
    training_data: TrainingData,
    n_states: int = config.N_STATES,
    refit_every: int = 12,
    min_train: int = 120,
    window_years: int | None = None,
    n_restarts: int = 2,
    n_iter: int = 200,
    random_state: int = config.RANDOM_SEED,
) -> BacktestResult:
    frame = training_data.frame
    if len(frame) <= min_train:
        raise ValueError(f"need more than min_train={min_train} rows, got {len(frame)}")
    post_chunks: list[pd.DataFrame] = []
    fits: list[dict] = []
    cols = [f"state_{k}" for k in range(n_states)]
    for cut in range(min_train, len(frame), refit_every):
        train = frame.iloc[:cut]
        if window_years:
            train = train.loc[train.index[-1] - pd.DateOffset(years=window_years):]
        td = TrainingData(train, training_data.provenance, training_data.source, training_data.feature_signs)
        engine = HMMEngine(n_states=n_states, n_restarts=n_restarts, n_iter=n_iter, random_state=random_state).fit(td)
        test = frame.iloc[cut : cut + refit_every]
        post = engine.score_path(test)
        post.columns = cols
        post_chunks.append(post)
        fits.append({"train_start": td.start, "train_end": td.end, "n_train": len(td), "fit_id": engine.fit_id,
                     "log_likelihood": engine.log_likelihood_, "scored_to": test.index[-1]})
        log.info("walk-forward: trained to %s, scored %s..%s", td.end.date(), test.index[0].date(), test.index[-1].date())
    posteriors = pd.concat(post_chunks)
    posteriors.index.name = "date"
    return BacktestResult(posteriors=posteriors, fits=pd.DataFrame(fits), n_states=n_states)
