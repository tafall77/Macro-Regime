"""The invariant that matters most: hypothetical override values never reach fit()."""
import numpy as np
import pandas as pd
import pytest

import scheduler
from data.overrides import OverrideStore
from model.hmm_engine import HMMEngine, ProvenanceError, TrainingData
from model.training_data import ACTUAL_PROVENANCE, SYNTHETIC_PROVENANCE
from scoring.live_score import scenario_panel


def test_refit_never_consults_the_override_store(fetcher, overrides, tmp_path, monkeypatch, labeler):
    overrides.set_override("ism_pmi", 30.0)      # wildly hypothetical
    overrides.set_override("yield_curve", -3.0)
    calls = []
    real_effective = OverrideStore.get_effective
    monkeypatch.setattr(OverrideStore, "get_effective", lambda self, *a, **k: calls.append(a) or real_effective(self, *a, **k))

    # scheduler.refit builds the training set from the fetcher only; give it the store for the guard
    engine = scheduler.refit(fetcher=fetcher, engine_path=tmp_path / "hmm.pkl", labeler=labeler,
                             refresh=False, n_states=2, n_restarts=1, overrides=overrides)
    assert engine.fitted and (tmp_path / "hmm.pkl").exists()
    # the guard's scenario cross-check reads the store, but fit() itself never did:
    # the training row equals the actual features, not the overridden ones
    actual_feats = fetcher.features().dropna()
    last = engine.training_end
    expected_pmi = actual_feats.loc[last, "pmi_vs_50"]
    assert expected_pmi != pytest.approx(30.0 - 50.0)
    # everything the engine saw is reproducible from the actual cache alone
    td = fetcher.training_data()
    pd.testing.assert_frame_equal(td.frame, actual_feats.loc[td.start:td.end])
    assert td.frame.loc[last, "pmi_vs_50"] == pytest.approx(expected_pmi)
    assert engine.training_provenance == ACTUAL_PROVENANCE


def test_training_data_ignores_overrides_entirely(fetcher, overrides):
    before = fetcher.training_data()
    overrides.set_override("ism_pmi", 30.0)
    overrides.set_override("fed_funds", 0.0)
    overrides.set_override("core_pce", 9.0)
    overrides.set_override("yield_curve", -2.0)
    after = fetcher.training_data()
    pd.testing.assert_frame_equal(before.frame, after.frame)
    # ... while the scenario path clearly differs on the latest row
    actual_p, scenario_p, as_of = scenario_panel(fetcher, overrides)
    assert (actual_p.loc[as_of] != scenario_p.loc[as_of]).all()


def test_fit_refuses_a_scenario_frame_even_when_dressed_up(fetcher, overrides, fitted_engine):
    overrides.set_override("ism_pmi", 30.0)
    _, scenario_p, _ = scenario_panel(fetcher, overrides)
    scenario_feats = fetcher.features(scenario_p).dropna()
    engine = HMMEngine(n_states=2, n_restarts=1, n_iter=5)
    with pytest.raises(ProvenanceError):
        engine.fit(scenario_feats)                                   # bare frame
    with pytest.raises(ProvenanceError):
        TrainingData(scenario_feats, "scenario")                    # unknown provenance


def test_assert_actual_only_catches_a_forged_training_set(fetcher, overrides):
    overrides.set_override("ism_pmi", 30.0)
    _, scenario_p, as_of = scenario_panel(fetcher, overrides)
    scenario_feats = fetcher.features(scenario_p).dropna()
    forged = TrainingData(scenario_feats, ACTUAL_PROVENANCE, feature_signs={})  # lies about provenance
    with pytest.raises(ProvenanceError, match="does not match|leaked"):
        scheduler.assert_actual_only(forged, fetcher, overrides)
    with pytest.raises(ProvenanceError, match="does not match"):
        scheduler.assert_actual_only(forged, fetcher)  # even without the store, the frame comparison catches it
    with pytest.raises(ProvenanceError, match="provenance"):
        scheduler.assert_actual_only(TrainingData(fetcher.features().dropna(), SYNTHETIC_PROVENANCE), fetcher)
    with pytest.raises(ProvenanceError):
        scheduler.assert_actual_only(fetcher.features().dropna(), fetcher)
    # the genuine article passes, with or without overrides in the store
    scheduler.assert_actual_only(fetcher.training_data(), fetcher, overrides)
    scheduler.assert_actual_only(fetcher.training_data(), fetcher)


def test_overrides_module_offers_no_training_frame():
    import data.overrides as ov
    assert not hasattr(ov.OverrideStore, "training_data")
    assert not any(name.startswith(("features", "frame", "panel")) for name in dir(ov.OverrideStore))
