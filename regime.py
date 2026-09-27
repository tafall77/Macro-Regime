"""Notebook-friendly façade over the whole engine.

    from regime import MacroRegime
    mr = MacroRegime(demo=True)        # or MacroRegime(api_key="...") for real FRED data
    mr.refit()                         # pulls (unless demo), fits, saves
    mr.label()                         # accept the heuristic labels (or mr.label({0: "bearish", ...}))
    mr.override(ism_pmi=47, yield_curve=-0.3)
    mr.report()                        # renders inline in Jupyter; .save("regime.html") for a file

Demo mode uses the synthetic 3-regime history in :mod:`data.synthetic` and a
separate cache directory, so it never touches your real cache or FRED key.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from data.fred_fetcher import FredFetcher
from data.overrides import OverrideStore
from data.synthetic import FakeFredSession, synthetic_raw, write_demo_manual_csvs
from data.transforms import CATALOGUE_KEYS, INDICATORS, select
from model.backtest import BacktestResult, walk_forward
from model.hmm_engine import HMMEngine, NotFittedError
from model.labeler import StateLabeler
from reporting.report import Report, build_report
from scoring.live_score import EngineLoader, LiveScore, live_score


class MacroRegime:
    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str | Path | None = None,
        demo: bool = False,
        n_states: int | None = None,
        seed: int = 1,
        stale_limit_months: int | None = None,
        indicators: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.demo = demo
        self.n_states = n_states or config.N_STATES
        root = Path(cache_dir) if cache_dir is not None else (config.CACHE_DIR / "demo" if demo else config.CACHE_DIR)
        self.root = root
        specs = select(indicators) if indicators is not None else INDICATORS
        session = None
        if demo:
            self._raw = synthetic_raw(seed=seed)
            session = FakeFredSession.from_raw(self._raw)
            api_key = api_key or "demo"
            write_demo_manual_csvs(self._raw, root / "manual")  # the not-on-FRED indicators, as CSV supplements
        self.fetcher = FredFetcher(api_key=api_key, cache_dir=root / "fred", manual_dir=root / "manual",
                                   session=session, stale_limit_months=stale_limit_months, indicators=specs)
        self.overrides = OverrideStore(self.fetcher, root / "overrides.json")
        self.labeler = StateLabeler(root / "model" / "labels.json")
        self.engine_path = root / "model" / "hmm_engine.pkl"
        self._loader = EngineLoader(self.engine_path)

    # ------------------------------------------------------------------ data
    def refresh(self) -> pd.DataFrame:
        """Re-pull the actual series (FRED, or the synthetic generator in demo mode)."""
        return self.fetcher.refresh()

    def panel(self) -> pd.DataFrame:
        return self.fetcher.panel()

    def features(self) -> pd.DataFrame:
        return self.fetcher.features()

    @property
    def true_states(self) -> pd.Series | None:
        """Demo mode only: the regime path the synthetic data was generated from."""
        return self._raw["_states"] if self.demo else None

    # ----------------------------------------------------------------- model
    def refit(self, refresh: bool | None = None, window_years: int | None = None, n_restarts: int = 5) -> HMMEngine:
        """Walk-forward refit on actual data only (see scheduler.refit)."""
        import scheduler

        if refresh is None:
            refresh = not self.fetcher.has_cache()
        engine = scheduler.refit(fetcher=self.fetcher, engine_path=self.engine_path, labeler=self.labeler,
                                 refresh=refresh, n_states=self.n_states, window_years=window_years,
                                 n_restarts=n_restarts, overrides=self.overrides)
        return engine

    @property
    def engine(self) -> HMMEngine:
        return self._loader.get()

    @property
    def is_fitted(self) -> bool:
        try:
            self._loader.get()
            return True
        except NotFittedError:
            return False

    def states(self) -> pd.DataFrame:
        """Per-state feature means, persistence, suggested and confirmed labels."""
        return self.labeler.describe(self.engine)

    def label(self, mapping: dict[int, str] | None = None) -> dict[int, str]:
        """Confirm state labels for the current fit; ``None`` accepts the heuristic suggestion."""
        if mapping is None:
            return self.labeler.accept_suggestion(self.engine)
        return self.labeler.set_labels(self.engine.fit_id, mapping)

    # -------------------------------------------------------------- scenario
    def override(self, **values: float) -> dict[str, float]:
        """Set overrides in headline units, e.g. ``override(ism_pmi=47, fed_funds=4.5)``."""
        for key, value in values.items():
            if key not in self.fetcher.keys:
                raise KeyError(f"unknown or inactive indicator {key!r}; active: {self.fetcher.keys}")
            if value is None:
                self.overrides.clear_override(key)
            else:
                self.overrides.set_override(key, value)
        return self.overrides.overrides()

    def clear_overrides(self) -> None:
        self.overrides.clear_all()

    # --------------------------------------------------------------- scoring
    def score(self) -> LiveScore:
        return live_score(self.fetcher, self.overrides, self.engine, self.labeler)

    def summary(self) -> pd.DataFrame:
        """Actual vs scenario probabilities per state as a small DataFrame."""
        s = self.score()
        return pd.DataFrame(
            {"label": [s.label(k) for k in range(s.n_states)],
             "actual": s.probs_actual, "scenario": s.probs_scenario,
             "delta": s.probs_scenario - s.probs_actual, "persistence_months": s.persistence},
            index=pd.RangeIndex(s.n_states, name="state"),
        )

    def report(self, history_years: int | None = 15, title: str = "Macro Regime Engine",
               include_indicator_history: bool = True) -> Report:
        """HTML report: renders inline in a notebook; ``.save(path)`` writes a standalone file."""
        return build_report(self.score(), self.fetcher, self.engine, history_years=history_years, title=title,
                            include_indicator_history=include_indicator_history)

    def backtest(self, refit_every: int = 12, min_train: int = 120, window_years: int | None = None,
                 n_restarts: int = 2, n_iter: int = 200) -> BacktestResult:
        """Out-of-sample walk-forward evaluation on the actual history."""
        return walk_forward(self.fetcher.training_data(), n_states=self.n_states, refit_every=refit_every,
                            min_train=min_train, window_years=window_years, n_restarts=n_restarts, n_iter=n_iter)

    @property
    def indicators(self) -> tuple[str, ...]:
        return self.fetcher.keys

    def evaluate_indicators(self, candidates: list[str] | None = None, base: list[str] | None = None, **kwargs) -> pd.DataFrame:
        """Rank candidate indicators by what they add to the base set (see model.selection).

        Uses a separate ``fred_eval`` cache so the production cache is untouched. In real
        mode this pulls the candidates' series from FRED (needs the API key).
        """
        from model.selection import evaluate_indicators

        base_keys = list(base) if base is not None else list(self.fetcher.keys)
        cand_keys = [k for k in (candidates or CATALOGUE_KEYS) if k not in base_keys]
        session = FakeFredSession.from_raw(self._raw) if self.demo else self.fetcher.session
        eval_fetcher = FredFetcher(api_key=self.fetcher.api_key, cache_dir=self.root / "fred_eval",
                                   manual_dir=self.fetcher.manual_dir, session=session,
                                   stale_limit_months=self.fetcher.stale_limit_months, indicators=base_keys + cand_keys)
        eval_fetcher.refresh()
        return evaluate_indicators(eval_fetcher, base_keys, cand_keys, n_states=self.n_states, **kwargs)

    def __repr__(self) -> str:
        mode = "demo" if self.demo else "fred"
        fit = self.engine.fit_id if self.is_fitted else "unfitted"
        return f"MacroRegime(mode={mode}, cache={self.root}, fit={fit}, overrides={self.overrides.overrides()})"
