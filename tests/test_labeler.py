import json

import pytest

from model.labeler import BEARISH, BULLISH, TRANSITION, LabelSet, StateLabeler, main


def test_suggest_follows_growth_order(fitted_engine):
    s = StateLabeler.suggest(fitted_engine)
    assert s == {0: BEARISH, 1: TRANSITION, 2: BULLISH}


def test_resolve_is_provisional_until_confirmed(fitted_engine, labeler):
    ls = labeler.resolve(fitted_engine)
    assert ls.provisional and ls.labels == StateLabeler.suggest(fitted_engine)
    assert ls.name(2) == "Bullish" and ls.display(2, 0.716) == "Bullish: 72"
    labeler.set_labels(fitted_engine.fit_id, {0: "bearish", 1: "transition", 2: "bullish"})
    ls = labeler.resolve(fitted_engine)
    assert not ls.provisional
    assert labeler.labels_for(fitted_engine.fit_id) == {0: BEARISH, 1: TRANSITION, 2: BULLISH}
    # a different fit id never inherits the mapping
    assert labeler.labels_for("deadbeef00") is None
    assert LabelSet("x", {}, True).name(1) == "State 1"


def test_partial_labels_stay_provisional(fitted_engine, labeler):
    labeler.set_label(fitted_engine.fit_id, 2, "Bullish")
    assert labeler.resolve(fitted_engine).provisional
    labeler.set_labels(fitted_engine.fit_id, {0: "BEARISH", 1: "transition"})
    assert not labeler.resolve(fitted_engine).provisional


def test_validation_and_persistence(fitted_engine, labeler, tmp_path):
    with pytest.raises(ValueError):
        labeler.set_label(fitted_engine.fit_id, 0, "sideways")
    labeler.accept_suggestion(fitted_engine)
    data = json.loads(labeler.path.read_text())
    assert data["fit_id"] == fitted_engine.fit_id and data["labels"] == {"0": "bearish", "1": "transition", "2": "bullish"}
    again = StateLabeler(labeler.path)
    assert again.confirmed_fit_id == fitted_engine.fit_id and not again.resolve(fitted_engine).provisional
    # confirming for a new fit replaces (not merges with) the old mapping
    again.set_labels("newfit0001", {0: "bullish"})
    assert again.labels_for(fitted_engine.fit_id) is None
    again.clear()
    assert again.confirmed_fit_id is None


def test_describe_has_suggested_and_confirmed_columns(fitted_engine, labeler):
    df = labeler.describe(fitted_engine)
    assert list(df.columns[:2]) == ["suggested", "confirmed"]
    assert list(df["suggested"]) == [BEARISH, TRANSITION, BULLISH]


def test_cli(fitted_engine, tmp_path, capsys):
    model = fitted_engine.save(tmp_path / "hmm.pkl")
    labels = tmp_path / "labels.json"
    assert main(["--model", str(model), "--labels", str(labels), "show"]) == 0
    assert "(provisional)" in capsys.readouterr().out
    assert main(["--model", str(model), "--labels", str(labels), "set", "0=bearish", "1=transition", "2=bullish"]) == 0
    assert "(confirmed)" in capsys.readouterr().out
    assert main(["--model", str(model), "--labels", str(labels), "accept"]) == 0
    assert StateLabeler(labels).labels_for(fitted_engine.fit_id) == {0: BEARISH, 1: TRANSITION, 2: BULLISH}
