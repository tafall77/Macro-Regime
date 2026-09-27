"""Monthly walk-forward refit of the HMM — entirely separate from live scoring.

    python scheduler.py --once            # refresh FRED, refit, save, print state table
    python scheduler.py --once --no-refresh
    python scheduler.py --daemon          # sleep until the 2nd of each month 06:00 UTC, refit, repeat
    python scheduler.py --cron            # print a crontab line instead

The refit reads **actual** data only. :func:`assert_actual_only` re-derives the
training matrix from the FRED cache and compares it with what is about to be
fitted; if an override store is supplied it additionally proves that no
overridden indicator's hypothetical value made it into the last training row.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
from data.fred_fetcher import FredFetcher
from data.transforms import features_from_panel, get_spec
from model.hmm_engine import ACTUAL_PROVENANCE, HMMEngine, ProvenanceError, TrainingData
from model.labeler import StateLabeler

log = logging.getLogger("scheduler")


def assert_actual_only(training_data: TrainingData, fetcher: FredFetcher, overrides=None) -> None:
    """Guard: the training set must be exactly what the actual cache implies.

    Raises :class:`ProvenanceError` if the provenance is wrong, if the frame
    differs from the fetcher's own actual feature matrix over the same window, or
    (when an override store is given) if an overridden indicator's hypothetical
    value appears in the final training row.
    """
    if not isinstance(training_data, TrainingData):
        raise ProvenanceError(f"expected TrainingData, got {type(training_data).__name__}")
    if training_data.provenance != ACTUAL_PROVENANCE:
        raise ProvenanceError(
            f"refit requires provenance {ACTUAL_PROVENANCE!r}, got {training_data.provenance!r}"
        )
    expected = fetcher.features().dropna().loc[training_data.start : training_data.end]
    expected = expected.loc[:, list(training_data.feature_names)]
    try:
        pd.testing.assert_frame_equal(
            training_data.frame, expected.astype("float64"), check_names=False, check_freq=False
        )
    except AssertionError as exc:
        raise ProvenanceError("training frame does not match the actual FRED feature matrix") from exc

    if overrides is None or not overrides.overrides():
        return
    # Build what the scenario row *would* look like and prove it is not what we train on.
    from scoring.live_score import scenario_panel  # local import: scoring depends on the model, not vice versa

    actual_p, scenario_p, as_of = scenario_panel(fetcher, overrides)
    f_actual = features_from_panel(actual_p, fetcher.indicators)
    f_scenario = features_from_panel(scenario_p, fetcher.indicators)
    last = training_data.end
    for key in overrides.overrides():
        feat = get_spec(key).feature_name
        if last not in f_scenario.index or last != as_of:
            continue  # the override month is not part of the training window at all
        actual_v, scenario_v = float(f_actual.loc[last, feat]), float(f_scenario.loc[last, feat])
        trained_v = float(training_data.frame.loc[last, feat])
        if not np.isclose(actual_v, scenario_v) and np.isclose(trained_v, scenario_v):
            raise ProvenanceError(
                f"override for {key!r} leaked into the training row {last.date()} ({feat}={trained_v})"
            )


def refit(
    *,
    fetcher: FredFetcher | None = None,
    engine_path: str | Path | None = None,
    labeler: StateLabeler | None = None,
    refresh: bool = True,
    n_states: int | None = None,
    window_years: int | None = None,
    training_start=None,
    overrides=None,
    n_restarts: int = 5,
) -> HMMEngine:
    """Refresh (optionally), rebuild the actual-only training set, fit, save.

    ``overrides`` is accepted only so :func:`assert_actual_only` can prove the
    hypothetical values were not used; it is never read for data.
    """
    fetcher = fetcher or FredFetcher()
    engine_path = Path(engine_path) if engine_path is not None else config.MODEL_PATH
    if refresh:
        fetcher.refresh()

    window = window_years if window_years is not None else config.TRAINING_WINDOW_YEARS
    training = fetcher.training_data(start=training_start, window_years=window)
    assert_actual_only(training, fetcher, overrides)

    engine = HMMEngine(n_states=n_states or config.N_STATES, n_restarts=n_restarts)
    engine.fit(training)
    engine.save(engine_path)

    labeler = labeler or StateLabeler()
    if labeler.labels_for(engine.fit_id) is None:
        log.warning(
            "fit %s has no confirmed labels; suggestion %s — confirm with `python -m model.labeler accept`",
            engine.fit_id, labeler.suggest(engine),
        )
    return engine


def next_run_time(now: datetime | None = None, day: int = config.REFIT_DAY_OF_MONTH, hour: int = config.REFIT_HOUR_UTC) -> datetime:
    """Next ``day``-of-month at ``hour`` UTC strictly after ``now``."""
    now = now or datetime.now(timezone.utc)
    candidate = now.replace(day=day, hour=hour, minute=0, second=0, microsecond=0)
    if candidate <= now:
        year, month = (now.year + 1, 1) if now.month == 12 else (now.year, now.month + 1)
        candidate = candidate.replace(year=year, month=month)
    return candidate


def run_daemon(**refit_kwargs) -> None:  # pragma: no cover - long-running loop
    while True:
        target = next_run_time()
        wait = (target - datetime.now(timezone.utc)).total_seconds()
        log.info("next refit at %s (in %.1f h)", target.isoformat(), wait / 3600)
        time.sleep(max(wait, 0))
        try:
            engine = refit(**refit_kwargs)
            log.info("refit done: %s", engine)
        except Exception:  # noqa: BLE001 - keep the daemon alive, log the failure
            log.exception("refit failed")
            time.sleep(timedelta(hours=1).total_seconds())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Monthly HMM refit (actual data only).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="refit now and exit (default)")
    mode.add_argument("--daemon", action="store_true", help="loop: sleep until the monthly slot, refit, repeat")
    mode.add_argument("--cron", action="store_true", help="print a crontab line")
    parser.add_argument("--no-refresh", action="store_true", help="fit on the existing cache without hitting FRED")
    parser.add_argument("--n-states", type=int, default=None)
    parser.add_argument("--window-years", type=int, default=None, help="rolling window; default expanding")
    parser.add_argument("--model", default=str(config.MODEL_PATH))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.cron:
        print(
            f"0 {config.REFIT_HOUR_UTC} {config.REFIT_DAY_OF_MONTH} * * cd {config.ROOT} && "
            f"{sys.executable} scheduler.py --once >> {config.CACHE_DIR}/refit.log 2>&1"
        )
        return 0

    kwargs = dict(
        refresh=not args.no_refresh,
        n_states=args.n_states,
        window_years=args.window_years,
        engine_path=args.model,
    )
    if args.daemon:
        run_daemon(**kwargs)
        return 0
    engine = refit(**kwargs)
    with pd.option_context("display.width", 160, "display.max_columns", 20, "display.precision", 3):
        print(engine)
        print(StateLabeler().describe(engine).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
