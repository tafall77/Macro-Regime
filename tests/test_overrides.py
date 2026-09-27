import json

import pandas as pd
import pytest

from data import overrides as ov
from data.overrides import OverrideStore


def test_set_clear_and_effective(fetcher, overrides):
    latest = fetcher.latest_date("ism_pmi")
    actual = fetcher.get_actual("ism_pmi", latest)
    assert overrides.get_effective("ism_pmi", latest) == actual
    assert not overrides.is_overridden("ism_pmi")

    overrides.set_override("ism_pmi", 42)
    assert overrides.is_overridden("ism_pmi") and "ism_pmi" in overrides and len(overrides) == 1
    assert overrides.get_effective("ism_pmi", latest) == 42.0
    assert overrides.get_effective("ism_pmi", latest + pd.offsets.MonthEnd(1)) == 42.0  # later dates: still the override
    # earlier dates always fall back to the actual
    earlier = latest - pd.offsets.MonthEnd(1)
    assert overrides.get_effective("ism_pmi", earlier) == fetcher.get_actual("ism_pmi", earlier)
    # other indicators are untouched
    assert overrides.get_effective("fed_funds", latest) == fetcher.get_actual("fed_funds", latest)

    overrides.clear_override("ism_pmi")
    assert overrides.get_effective("ism_pmi", latest) == actual
    overrides.set_override("ism_pmi", 42)
    overrides.set_override("fed_funds", "5.5")  # numeric strings are accepted
    overrides.clear_all()
    assert overrides.overrides() == {}


def test_validation(fetcher, overrides):
    with pytest.raises(KeyError):
        overrides.set_override("gdp", 1.0)
    with pytest.raises(ValueError):
        overrides.set_override("ism_pmi", "abc")
    with pytest.raises(ValueError):
        overrides.set_override("ism_pmi", float("nan"))
    with pytest.raises(KeyError):
        overrides.get_effective("gdp", "2025-08-31")
    assert overrides.overrides() == {}


def test_session_file_round_trip(tmp_path, fetcher):
    path = tmp_path / "session" / "overrides.json"
    store = OverrideStore(fetcher, path)
    store.set_override("yield_curve", -0.3)
    store.set_override("core_pce", 3.1)
    assert json.loads(path.read_text()) == {"overrides": {"yield_curve": -0.3, "core_pce": 3.1}}
    reloaded = OverrideStore(fetcher, path)
    assert reloaded.overrides() == {"yield_curve": -0.3, "core_pce": 3.1}
    # junk in the file is ignored, not fatal
    path.write_text(json.dumps({"overrides": {"nope": 1, "ism_pmi": "x", "fed_funds": 4.5}}))
    assert OverrideStore(fetcher, path).overrides() == {"fed_funds": 4.5}


def test_in_memory_store_without_file(fetcher):
    store = OverrideStore(fetcher)
    store.set_override("ism_pmi", 55)
    assert store.get_override("ism_pmi") == 55.0 and store.get_override("fed_funds") is None


def test_store_without_actual_source_cannot_resolve_effective():
    store = OverrideStore()
    store.set_override("ism_pmi", 55)
    with pytest.raises(RuntimeError):
        store.get_effective("ism_pmi", "2025-08-31")


def test_module_level_functions_use_default_store(monkeypatch, fetcher, tmp_path):
    monkeypatch.setattr(ov, "_default_store", OverrideStore(fetcher, tmp_path / "o.json"))
    ov.set_override("ism_pmi", 40)
    assert ov.get_effective("ism_pmi", fetcher.latest_date()) == 40.0
    ov.clear_override("ism_pmi")
    ov.set_override("fed_funds", 1)
    ov.clear_all()
    assert ov.default_store().overrides() == {}
