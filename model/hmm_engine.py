"""Gaussian hidden Markov model over the transformed macro feature matrix.

* :meth:`HMMEngine.fit` – full EM refit on a :class:`TrainingData` (actual data
  only, enforced by type + provenance). Run monthly by :mod:`scheduler`.
* :meth:`HMMEngine.score` – forward (filtering) recursion from the last fitted
  posterior through one or more new observation rows; returns the state
  probabilities after the last row. Cheap: no EM, no retraining. Called for both
  the *actual* and the *scenario* rows by :mod:`scoring.live_score`.
* :meth:`HMMEngine.persistence` – expected remaining duration of a state from the
  transition-matrix diagonal, in months.

States are re-ordered after every fit by a "growth score" (bullish-signed sum of
standardised state means) so that state 0 is the most bearish-looking and state
``n_states - 1`` the most bullish-looking. That keeps indices reasonably stable
across refits; the human still confirms the labels via :mod:`model.labeler`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

import config
from model.training_data import (  # noqa: F401  (re-exported for callers)
    ACTUAL_PROVENANCE,
    ALLOWED_PROVENANCE,
    SYNTHETIC_PROVENANCE,
    ProvenanceError,
    TrainingData,
)

log = logging.getLogger(__name__)

MIN_TRAINING_ROWS = 36
_LOG_EPS = 1e-300


class NotFittedError(RuntimeError):
    pass


class HMMEngine:
    def __init__(
        self,
        n_states: int = config.N_STATES,
        covariance_type: str = "full",
        n_iter: int = 500,
        tol: float = 1e-4,
        n_restarts: int = 5,
        random_state: int = config.RANDOM_SEED,
        min_covar: float = 1e-3,
    ) -> None:
        if n_states < 2:
            raise ValueError("n_states must be >= 2")
        self.n_states = int(n_states)
        self.covariance_type = covariance_type
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.n_restarts = max(1, int(n_restarts))
        self.random_state = int(random_state)
        self.min_covar = float(min_covar)

        # populated by fit()
        self.fitted: bool = False
        self.feature_names: tuple[str, ...] = ()
        self.feature_signs: dict[str, int] = {}
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.startprob_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covars_: np.ndarray | None = None
        self.state_scores_: np.ndarray | None = None
        self.filtered_: pd.DataFrame | None = None
        self.log_likelihood_: float | None = None
        self.converged_: bool | None = None
        self.training_start: pd.Timestamp | None = None
        self.training_end: pd.Timestamp | None = None
        self.n_train: int = 0
        self.training_provenance: str | None = None
        self.training_source: str = ""
        self.fitted_at: datetime | None = None
        self.fit_id: str | None = None
        self._model: GaussianHMM | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, training_data: TrainingData) -> "HMMEngine":
        """Fit by EM on actual data only.

        Raises :class:`ProvenanceError` for anything that is not a
        :class:`TrainingData` with an allowed provenance. A raw DataFrame, a
        scenario feature row, or anything derived from the override store can
        therefore never be fitted, by construction.
        """
        if not isinstance(training_data, TrainingData):
            raise ProvenanceError(
                "HMMEngine.fit() only accepts TrainingData built by "
                "FredFetcher.training_data(); got " + type(training_data).__name__
            )
        if training_data.provenance not in ALLOWED_PROVENANCE:
            raise ProvenanceError(f"disallowed provenance {training_data.provenance!r}")
        if len(training_data) < MIN_TRAINING_ROWS:
            raise ValueError(f"need at least {MIN_TRAINING_ROWS} rows, got {len(training_data)}")

        frame = training_data.frame
        X = frame.to_numpy(dtype="float64")
        mean = X.mean(axis=0)
        scale = X.std(axis=0, ddof=0)
        scale = np.where(scale > 0, scale, 1.0)
        Z = (X - mean) / scale

        best: GaussianHMM | None = None
        best_ll = -np.inf
        for r in range(self.n_restarts):
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                tol=self.tol,
                min_covar=self.min_covar,
                random_state=self.random_state + r,
            )
            model.fit(Z)
            ll = float(model.score(Z))
            if np.isfinite(ll) and ll > best_ll:
                best, best_ll = model, ll
        if best is None:
            raise RuntimeError("HMM fitting failed for every restart")

        self._model = best
        self.feature_names = training_data.feature_names
        self.feature_signs = {
            name: int(training_data.feature_signs.get(name, 0)) for name in self.feature_names
        }
        self.mean_, self.scale_ = mean, scale
        self.startprob_ = np.asarray(best.startprob_, dtype="float64").copy()
        self.transmat_ = np.asarray(best.transmat_, dtype="float64").copy()
        self.means_ = np.asarray(best.means_, dtype="float64").copy()
        self.covars_ = np.asarray(best.covars_, dtype="float64").copy()  # full (K, F, F)
        self._canonicalize_states()

        self.filtered_ = pd.DataFrame(
            self._filter(Z, np.log(np.clip(self.startprob_, _LOG_EPS, None))),
            index=frame.index,
            columns=[f"state_{k}" for k in range(self.n_states)],
        )
        self.log_likelihood_ = best_ll
        self.converged_ = bool(best.monitor_.converged)
        self.training_start, self.training_end = training_data.start, training_data.end
        self.n_train = len(training_data)
        self.training_provenance = training_data.provenance
        self.training_source = training_data.source
        self.fitted_at = datetime.now(timezone.utc)
        self.fitted = True
        self.fit_id = self._compute_fit_id()
        log.info(
            "HMM fitted: %d states, %d rows %s..%s, loglik %.2f, fit_id %s",
            self.n_states, self.n_train, self.training_start.date(), self.training_end.date(),
            self.log_likelihood_, self.fit_id,
        )
        return self

    def _growth_scores(self) -> np.ndarray:
        """Bullish-signed sum of standardised state means; higher = more bullish-looking."""
        signs = np.array([self.feature_signs.get(n, 0) for n in self.feature_names], dtype="float64")
        if not signs.any():  # no signs known: order by the first feature
            signs = np.zeros(len(self.feature_names))
            signs[0] = 1.0
        return self.means_ @ signs

    def _canonicalize_states(self) -> None:
        scores = self._growth_scores()
        order = np.argsort(scores, kind="stable")
        self.startprob_ = self.startprob_[order]
        self.transmat_ = self.transmat_[np.ix_(order, order)]
        self.means_ = self.means_[order]
        self.covars_ = self.covars_[order]
        self.state_scores_ = scores[order]
        # keep the wrapped hmmlearn model consistent (used for cross-checks)
        m = self._model
        m.startprob_ = self.startprob_.copy()
        m.transmat_ = self.transmat_.copy()
        m.means_ = self.means_.copy()
        m.covars_ = self.covars_.copy()

    def _compute_fit_id(self) -> str:
        h = hashlib.sha1()
        h.update(
            "|".join(
                [
                    str(self.training_start.date()),
                    str(self.training_end.date()),
                    str(self.n_states),
                    ",".join(self.feature_names),
                    np.array2string(np.round(self.means_, 4), separator=","),
                    np.array2string(np.round(self.transmat_, 4), separator=","),
                ]
            ).encode()
        )
        return h.hexdigest()[:10]

    # -------------------------------------------------------------- scoring
    def _require_fitted(self) -> None:
        if not self.fitted:
            raise NotFittedError("HMMEngine is not fitted; run scheduler.refit() first")

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype="float64") - self.mean_) / self.scale_

    def _emission_logprob(self, Z: np.ndarray) -> np.ndarray:
        """(T, K) log N(z_t | mu_k, Sigma_k)."""
        Z = np.atleast_2d(Z)
        out = np.empty((Z.shape[0], self.n_states))
        for k in range(self.n_states):
            out[:, k] = multivariate_normal(
                mean=self.means_[k], cov=self.covars_[k], allow_singular=True
            ).logpdf(Z)
        return out

    def _filter(self, Z: np.ndarray, log_prior: np.ndarray) -> np.ndarray:
        """Forward recursion. ``log_prior`` is the log state distribution *before* seeing Z[0]."""
        logB = self._emission_logprob(Z)
        logA = np.log(np.clip(self.transmat_, _LOG_EPS, None))
        post = np.empty_like(logB)
        lp = np.asarray(log_prior, dtype="float64")
        for t in range(logB.shape[0]):
            la = lp + logB[t]
            la -= logsumexp(la)
            post[t] = np.exp(la)
            lp = logsumexp(la[:, None] + logA, axis=0)
        return post

    def _coerce(self, features) -> tuple[np.ndarray, pd.DatetimeIndex | None]:
        """Accept a DataFrame/Series/array; return (T, F) array + optional dates."""
        if isinstance(features, pd.DataFrame):
            missing = [c for c in self.feature_names if c not in features.columns]
            if missing:
                raise KeyError(f"feature frame is missing columns {missing}")
            df = features.loc[:, list(self.feature_names)]
            dates = df.index if isinstance(df.index, pd.DatetimeIndex) else None
            X = df.to_numpy(dtype="float64")
        elif isinstance(features, pd.Series):
            if set(self.feature_names) <= set(features.index):
                X = features.loc[list(self.feature_names)].to_numpy(dtype="float64")[None, :]
            else:
                X = features.to_numpy(dtype="float64")[None, :]
            dates = None
        else:
            X = np.asarray(features, dtype="float64")
            if X.ndim == 1:
                X = X[None, :]
            dates = None
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"expected {len(self.feature_names)} features {list(self.feature_names)}, got shape {X.shape}"
            )
        if not np.isfinite(X).all():
            raise ValueError("feature rows contain NaN/inf; a stale or missing indicator?")
        return X, dates

    def _log_prior_for(self, first_date) -> np.ndarray:
        """Log prior over states before observing a row dated ``first_date``.

        * ``None`` (undated input): step forward from the last fitted posterior.
        * Dated: step forward from the filtered posterior at the last training
          month strictly before ``first_date``; if there is none, use the fitted
          initial distribution.
        """
        if first_date is None:
            prev = self.filtered_.iloc[-1].to_numpy()
        else:
            earlier = self.filtered_.loc[self.filtered_.index < pd.Timestamp(first_date)]
            if earlier.empty:
                return np.log(np.clip(self.startprob_, _LOG_EPS, None))
            prev = earlier.iloc[-1].to_numpy()
        return np.log(np.clip(prev @ self.transmat_, _LOG_EPS, None))

    def score(self, features, *, start_date=None) -> np.ndarray:
        """State probabilities after filtering through ``features`` (one or more rows).

        ``features`` may be a 1-D feature vector, a 2-D array, a Series indexed by
        feature name, or a DataFrame with a date index. With a date index the
        recursion starts from the posterior at the last training month before the
        first row (so re-scoring the final training month with an override does
        not double count it); otherwise from the last fitted posterior.
        """
        self._require_fitted()
        X, dates = self._coerce(features)
        anchor = pd.Timestamp(start_date) if start_date is not None else (dates[0] if dates is not None else None)
        post = self._filter(self._standardize(X), self._log_prior_for(anchor))
        return post[-1]

    def score_path(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Filtered posterior for every row of a dated feature frame."""
        self._require_fitted()
        X, dates = self._coerce(frame)
        post = self._filter(self._standardize(X), self._log_prior_for(dates[0] if dates is not None else None))
        return pd.DataFrame(post, index=frame.index, columns=self.filtered_.columns)

    def persistence(self, state: int) -> float:
        """Expected duration (months) of ``state`` = 1 / (1 - p_ii)."""
        self._require_fitted()
        p = float(self.transmat_[int(state), int(state)])
        if p >= 1.0:
            return float("inf")
        return 1.0 / (1.0 - p)

    def persistences(self) -> np.ndarray:
        return np.array([self.persistence(k) for k in range(self.n_states)])

    def stationary_distribution(self) -> np.ndarray:
        self._require_fitted()
        w, v = np.linalg.eig(self.transmat_.T)
        i = int(np.argmin(np.abs(w - 1.0)))
        pi = np.real(v[:, i])
        pi = np.abs(pi) / np.abs(pi).sum()
        return pi

    # ------------------------------------------------------------ describing
    def state_means(self) -> pd.DataFrame:
        """Per-state feature means in original (un-standardised) units."""
        self._require_fitted()
        means = self.means_ * self.scale_ + self.mean_
        return pd.DataFrame(means, columns=list(self.feature_names), index=pd.RangeIndex(self.n_states, name="state"))

    def describe(self) -> pd.DataFrame:
        """One row per state: feature means, persistence, stationary probability, training share."""
        df = self.state_means()
        df["growth_score"] = self.state_scores_
        df["persistence_months"] = self.persistences()
        df["stationary_prob"] = self.stationary_distribution()
        argmax = self.filtered_.to_numpy().argmax(axis=1)
        df["train_share"] = [float((argmax == k).mean()) for k in range(self.n_states)]
        return df

    def to_metadata(self) -> dict:
        self._require_fitted()
        return {
            "fit_id": self.fit_id,
            "fitted_at": self.fitted_at.isoformat(timespec="seconds"),
            "n_states": self.n_states,
            "covariance_type": self.covariance_type,
            "feature_names": list(self.feature_names),
            "training_start": str(self.training_start.date()),
            "training_end": str(self.training_end.date()),
            "n_train": self.n_train,
            "training_provenance": self.training_provenance,
            "training_source": self.training_source,
            "log_likelihood": self.log_likelihood_,
            "converged": self.converged_,
            "transmat": np.round(self.transmat_, 6).tolist(),
            "persistence_months": np.round(self.persistences(), 3).tolist(),
            "state_means": {
                str(k): {f: round(float(v), 6) for f, v in row.items()}
                for k, row in self.state_means().iterrows()
            },
        }

    # ----------------------------------------------------------- persistence
    def save(self, path: str | Path = config.MODEL_PATH) -> Path:
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        path.with_suffix(".json").write_text(json.dumps(self.to_metadata(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path = config.MODEL_PATH) -> "HMMEngine":
        path = Path(path)
        if not path.exists():
            raise NotFittedError(f"no fitted model at {path}; run scheduler.refit() first")
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain an HMMEngine")
        return obj

    def __repr__(self) -> str:
        if not self.fitted:
            return f"HMMEngine(n_states={self.n_states}, unfitted)"
        return (
            f"HMMEngine(n_states={self.n_states}, fit_id={self.fit_id}, "
            f"trained {self.training_start.date()}..{self.training_end.date()}, n={self.n_train})"
        )
