from datetime import datetime, timezone

import pytest

import scheduler
from model.hmm_engine import HMMEngine
from model.labeler import StateLabeler


def test_next_run_time():
    now = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)
    assert scheduler.next_run_time(now, day=2, hour=6) == datetime(2026, 10, 2, 6, 0, tzinfo=timezone.utc)
    assert scheduler.next_run_time(datetime(2026, 12, 5, tzinfo=timezone.utc), day=2, hour=6) == datetime(2027, 1, 2, 6, tzinfo=timezone.utc)
    assert scheduler.next_run_time(datetime(2026, 9, 1, tzinfo=timezone.utc), day=2, hour=6) == datetime(2026, 9, 2, 6, tzinfo=timezone.utc)


def test_refit_refreshes_fits_saves_and_flags_unlabelled(fetcher, fake_session, tmp_path, labeler, caplog):
    n_calls = len(fake_session.calls)
    path = tmp_path / "model" / "hmm.pkl"
    with caplog.at_level("WARNING", logger="scheduler"):
        engine = scheduler.refit(fetcher=fetcher, engine_path=path, labeler=labeler, refresh=True,
                                 n_states=2, n_restarts=1, window_years=10)
    assert len(fake_session.calls) > n_calls           # refresh hit "FRED"
    assert path.exists() and path.with_suffix(".json").exists()
    assert HMMEngine.load(path).fit_id == engine.fit_id
    assert engine.n_states == 2 and engine.training_start.year >= 2015
    assert "no confirmed labels" in caplog.text
    # labels confirmed -> no warning on the next refit of the same data
    labeler.accept_suggestion(engine)
    caplog.clear()
    with caplog.at_level("WARNING", logger="scheduler"):
        again = scheduler.refit(fetcher=fetcher, engine_path=path, labeler=labeler, refresh=False,
                                n_states=2, n_restarts=1, window_years=10)
    assert again.fit_id == engine.fit_id and "no confirmed labels" not in caplog.text


def test_cli_once_and_cron(fetcher, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(scheduler, "FredFetcher", lambda: fetcher)
    monkeypatch.setattr(scheduler, "StateLabeler", lambda: StateLabeler(tmp_path / "labels.json"))
    path = tmp_path / "hmm.pkl"
    assert scheduler.main(["--once", "--no-refresh", "--n-states", "2", "--window-years", "10", "--model", str(path)]) == 0
    out = capsys.readouterr().out
    assert "HMMEngine(n_states=2" in out and "suggested" in out and path.exists()
    assert scheduler.main(["--cron"]) == 0
    assert "scheduler.py --once" in capsys.readouterr().out
