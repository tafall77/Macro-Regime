import pytest

import dashboard.app as dash_app
from data.fred_fetcher import FredError
from data.transforms import INDICATOR_KEYS
from scoring.live_score import EngineLoader


@pytest.fixture
def services(fetcher, overrides, fitted_engine, labeler, tmp_path):
    path = fitted_engine.save(tmp_path / "hmm.pkl")
    return dash_app.Services(fetcher=fetcher, overrides=overrides, engine_loader=EngineLoader(path), labeler=labeler)


def test_app_builds_with_expected_components(services):
    app = dash_app.create_app(services)
    ids = set()

    def walk(c):
        if getattr(c, "id", None):
            ids.add(c.id)
        for child in (getattr(c, "children", None) or []) if isinstance(getattr(c, "children", None), list) else [getattr(c, "children", None)]:
            if child is not None and hasattr(child, "children") or getattr(child, "id", None):
                walk(child)

    walk(app.layout)
    for k in INDICATOR_KEYS:
        assert {f"override-{k}", f"actual-{k}", f"badge-{k}"} <= ids
    assert {"refresh-btn", "clear-btn", "gauge-actual", "gauge-scenario", "regime-select", "legend"} <= ids
    assert len(app.callback_map) == 2
    assert "{%app_entry%}" in app.index_string and "--actual" in app.index_string


def test_render_without_overrides(services):
    r = dash_app.render(services)
    assert r["banner_class"] == "banner" and r["score"] is not None
    assert r["regime_value"] == r["score"].top_actual
    assert len(r["regime_options"]) == 3
    assert r["gauge_actual"].data[0].value == pytest.approx(r["score"].prob(r["regime_value"]) * 100, abs=0.05)
    assert r["gauge_scenario"].data[0].delta.reference == pytest.approx(r["gauge_actual"].data[0].value)
    assert all(r["badge"][k] == ("FRED value", "badge") for k in INDICATOR_KEYS)
    assert "as of Aug 2025" in r["status"] and r["score"].fit_id in r["status"]
    assert len(r["distribution"].data) == 2 and r["distribution"].data[0].name == "Actual"
    assert "provisional" in str(r["legend"][-1].children)


def test_render_with_override_and_tracked_regime(services):
    services.overrides.set_override("ism_pmi", 41)
    r = dash_app.render(services, tracked_state=2)
    assert r["regime_value"] == 2 and r["banner_class"] == "banner info"
    text, cls = r["badge"]["ism_pmi"]
    assert cls == "badge on" and "41.0" in text
    assert r["badge"]["fed_funds"] == ("FRED value", "badge")
    assert "→" in r["feature"]["ism_pmi"] and "→" not in r["feature"]["fed_funds"]
    d = r["score"].delta(2) * 100
    assert f"{d:+.1f} pts" in r["delta"][0].children
    # an out-of-range tracked state falls back to the top actual state
    assert dash_app.render(services, tracked_state=99)["regime_value"] == r["score"].top_actual


def test_render_reports_missing_model_and_cache(services, tmp_path, fetcher):
    services.engine_loader = EngineLoader(tmp_path / "nope.pkl")
    r = dash_app.render(services)
    assert r["banner_class"] == "banner error" and "scheduler.py --once" in r["banner"]
    services.engine_loader = EngineLoader(tmp_path / "hmm.pkl")
    services.fetcher = type(fetcher)(api_key="k", cache_dir=tmp_path / "empty", manual_dir=tmp_path / "m", session=object())
    r = dash_app.render(services)
    assert r["banner_class"] == "banner error" and "Refresh from FRED" in r["banner"]


def test_sync_overrides_logic(services):
    store = services.overrides
    # initial load shows the persisted session values
    store.set_override("core_pce", 3.3)
    assert dash_app.sync_overrides(services, None, {}) == {"yield_curve": None, "core_pce": 3.3, "fed_funds": None, "ism_pmi": None}
    # typing a value sets it; blanking clears it; garbage clears it
    shown = dash_app.sync_overrides(services, "override-ism_pmi", {"ism_pmi": 47})
    assert shown["ism_pmi"] == 47.0 and store.get_override("ism_pmi") == 47.0
    shown = dash_app.sync_overrides(services, "override-ism_pmi", {"ism_pmi": None})
    assert shown["ism_pmi"] is None and not store.is_overridden("ism_pmi")
    store.set_override("ism_pmi", 47)
    dash_app.sync_overrides(services, "override-ism_pmi", {"ism_pmi": "abc"})
    assert not store.is_overridden("ism_pmi")
    # clear button wipes everything and blanks the inputs
    assert dash_app.sync_overrides(services, "clear-btn", {}) == {k: None for k in INDICATOR_KEYS}
    assert store.overrides() == {}


def test_refresh_keeps_overrides(services, fake_session):
    services.overrides.set_override("ism_pmi", 41)
    n = len(fake_session.calls)
    services.fetcher.refresh()
    assert len(fake_session.calls) > n
    assert services.overrides.overrides() == {"ism_pmi": 41.0}
    assert dash_app.render(services)["score"].has_overrides


def test_fmt():
    assert dash_app._fmt(0.456, "pp") == "+0.46 pp"
    assert dash_app._fmt(2.8, "%") == "2.80%"
    assert dash_app._fmt(47.26, "index") == "47.3"
