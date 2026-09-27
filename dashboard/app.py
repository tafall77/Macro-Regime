"""Dash dashboard: actual vs scenario regime confidence.

    python -m dashboard.app          # http://127.0.0.1:8050

Layout
------
* four indicator panels: current FRED headline value (read-only), an override
  input in the same units, and a "using override" badge;
* two confidence gauges side by side (Actual / Scenario) for the tracked regime,
  with the scenario-minus-actual delta highlighted;
* the persistence estimate (expected months in the top regime) under the gauges;
* the full state distribution for both runs and a state-label legend;
* "Refresh from FRED" (re-pulls actuals, keeps overrides) and "Clear overrides".

All heavy lifting is :func:`scoring.live_score.live_score`; nothing here refits.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from string import Template
from typing import Any

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html

import config
from data.fred_fetcher import CacheMissingError, FredError, FredFetcher, StaleDataError
from data.overrides import OverrideStore
from data.transforms import get_spec
from model.hmm_engine import NotFittedError
from model.labeler import StateLabeler
from reporting import figures as F
from reporting.figures import (ACTUAL, ACTUAL_TRACK, BORDER, FONT, GRID, INK, INK2, MUTED, PLANE, SCENARIO,
                               SCENARIO_TRACK, SURFACE, WARNING, gauge_figure)
from reporting.report import build_history
from scoring.live_score import EngineLoader, LiveScore, live_score

log = logging.getLogger(__name__)

INDEX_STRING = Template("""<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
  :root { color-scheme: light; --surface: $surface; --plane: $plane; --ink: $ink; --ink2: $ink2;
          --muted: $muted; --grid: $grid; --border: $border; --actual: $actual; --scenario: $scenario;
          --warning: $warning; }
  body { margin: 0; background: var(--plane); color: var(--ink); font-family: $font; }
  .page { max-width: 1240px; margin: 0 auto; padding: 20px 16px 40px; }
  .topbar { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; justify-content: space-between; }
  .topbar h1 { font-size: 20px; margin: 0; font-weight: 650; }
  .status { color: var(--ink2); font-size: 13px; }
  .btn { border: 1px solid var(--border); background: var(--surface); color: var(--ink); padding: 8px 14px;
         border-radius: 8px; font-size: 14px; cursor: pointer; }
  .btn:hover { border-color: var(--ink2); }
  .btn-primary { background: var(--actual); color: #fff; border-color: var(--actual); }
  .banner { margin: 14px 0 0; padding: 10px 14px; border-radius: 8px; font-size: 14px; display: none; }
  .banner.error { display: block; background: #fbe3e3; color: #7a1f1f; border: 1px solid #e9b7b7; }
  .banner.info { display: block; background: #e6f0fb; color: #14406f; border: 1px solid #b9d3f2; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; margin-top: 16px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }
  .card h3 { margin: 0 0 2px; font-size: 15px; font-weight: 650; }
  .card .sub { color: var(--muted); font-size: 12px; margin-bottom: 8px; }
  .card .value { font-size: 26px; font-weight: 600; }
  .card .asof { color: var(--ink2); font-size: 12px; }
  .card .feat { color: var(--ink2); font-size: 12px; margin-top: 4px; }
  .card label { display: block; margin-top: 10px; font-size: 12px; color: var(--ink2); }
  .card input { width: 100%; box-sizing: border-box; margin-top: 4px; padding: 7px 9px; font-size: 14px;
                border: 1px solid var(--border); border-radius: 6px; background: var(--plane); color: var(--ink); }
  .badge { display: inline-block; margin-top: 8px; font-size: 12px; padding: 3px 8px; border-radius: 999px;
           border: 1px solid var(--border); color: var(--ink2); }
  .badge.on { border-color: var(--warning); background: #fff5dc; color: #5a3d00; font-weight: 600; }
  .section { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; margin-top: 16px; }
  .section h2 { font-size: 15px; margin: 0 0 8px; font-weight: 650; }
  .gauges { display: grid; grid-template-columns: 1fr auto 1fr; gap: 12px; align-items: center; }
  .delta { text-align: center; min-width: 150px; }
  .delta .big { font-size: 30px; font-weight: 650; white-space: nowrap; }
  .delta .lbl { color: var(--ink2); font-size: 12px; }
  .persist { display: flex; flex-wrap: wrap; gap: 24px; margin-top: 6px; color: var(--ink2); font-size: 14px; }
  .persist b { color: var(--ink); }
  .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
  table.legend { border-collapse: collapse; width: 100%; font-size: 13px; }
  table.legend th, table.legend td { text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--grid); font-variant-numeric: tabular-nums; }
  table.legend th:first-child, table.legend td:first-child, table.legend th:nth-child(2), table.legend td:nth-child(2) { text-align: left; }
  table.legend th { color: var(--muted); font-weight: 500; }
  .note { color: var(--muted); font-size: 12px; margin-top: 8px; }
  .select { max-width: 320px; }
  @media (max-width: 760px) { .gauges { grid-template-columns: 1fr; } }
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>""").substitute(
    surface=SURFACE, plane=PLANE, ink=INK, ink2=INK2, muted=MUTED, grid=GRID, border=BORDER,
    actual=ACTUAL, scenario=SCENARIO, warning=WARNING, font=FONT,
)


@dataclass
class Services:
    fetcher: FredFetcher = field(default_factory=FredFetcher)
    overrides: OverrideStore | None = None
    engine_loader: EngineLoader = field(default_factory=EngineLoader)
    labeler: StateLabeler = field(default_factory=StateLabeler)

    def __post_init__(self) -> None:
        if self.overrides is None:
            self.overrides = OverrideStore(self.fetcher, config.OVERRIDES_PATH)


# ------------------------------------------------------------------ figures
def _fmt(value: float, unit: str) -> str:
    if unit == "index":
        return f"{value:.1f}"
    return f"{value:+.2f} {unit}" if unit == "pp" else f"{value:.2f}{'%' if unit == '%' else ' ' + unit}"


def distribution_figure(score: LiveScore) -> go.Figure:
    names = [score.label(k) for k in range(score.n_states)]
    return F.distribution_figure(names, [score.prob(k) for k in range(score.n_states)],
                                 [score.prob(k, True) for k in range(score.n_states)])


def legend_table(score: LiveScore) -> html.Table:
    feats = list(score.state_table.columns)
    header = html.Tr([html.Th("State"), html.Th("Label")] + [html.Th(f) for f in feats]
                     + [html.Th("Persistence (mo)"), html.Th("Actual"), html.Th("Scenario")])
    rows = []
    for k in range(score.n_states):
        means = score.state_table.loc[k]
        rows.append(html.Tr(
            [html.Td(str(k)), html.Td([html.Span(className="swatch", style={"background": F.state_color(k, score.labels.labels.get(k))}), score.label(k)])]
            + [html.Td(f"{means[f]:+.2f}") for f in feats]
            + [html.Td(f"{score.persistence[k]:.1f}"), html.Td(f"{score.prob(k) * 100:.0f}%"),
               html.Td(f"{score.prob(k, True) * 100:.0f}%")]
        ))
    return html.Table([html.Thead(header), html.Tbody(rows)], className="legend")


# ---------------------------------------------------------------- rendering
def render(services: Services, tracked_state: int | None = None) -> dict[str, Any]:
    """Compute every dynamic piece of the page. Pure apart from reading the services."""
    INDICATOR_KEYS = services.fetcher.keys
    out: dict[str, Any] = {
        "banner": "", "banner_class": "banner", "status": "", "regime_options": [], "regime_value": tracked_state,
        "gauge_actual": go.Figure(), "gauge_scenario": go.Figure(), "delta": [], "persistence": [],
        "distribution": go.Figure(), "history": go.Figure(), "legend": [], "score": None,
        "actual": {k: "—" for k in INDICATOR_KEYS}, "feature": {k: "" for k in INDICATOR_KEYS},
        "badge": {k: ("FRED value", "badge") for k in INDICATOR_KEYS},
    }
    try:
        engine = services.engine_loader.get()
        score = live_score(services.fetcher, services.overrides, engine, services.labeler)
    except NotFittedError as exc:
        out.update(banner=f"No fitted model yet: {exc}. Run `python scheduler.py --once`.", banner_class="banner error")
        return out
    except CacheMissingError as exc:
        out.update(banner=f"No FRED cache yet ({exc}). Click “Refresh from FRED” (needs FRED_API_KEY).",
                   banner_class="banner error")
        return out
    except (StaleDataError, FredError, ValueError, KeyError) as exc:
        out.update(banner=f"Cannot score: {exc}", banner_class="banner error")
        return out

    out["score"] = score
    if tracked_state is None or not (0 <= int(tracked_state) < score.n_states):
        tracked_state = score.top_actual
    tracked_state = int(tracked_state)
    out["regime_options"] = [{"label": f"{score.label(k)} (state {k})", "value": k} for k in range(score.n_states)]
    out["regime_value"] = tracked_state

    label = score.label(tracked_state)
    pa, ps = score.prob(tracked_state), score.prob(tracked_state, True)
    out["gauge_actual"] = gauge_figure(f"Actual · {label}", pa, ACTUAL, ACTUAL_TRACK)
    out["gauge_scenario"] = gauge_figure(f"Scenario · {label}", ps, SCENARIO, SCENARIO_TRACK, reference=pa)
    d = (ps - pa) * 100
    arrow = "▲" if d > 0.05 else ("▼" if d < -0.05 else "•")
    out["delta"] = [html.Div(f"{arrow} {d:+.1f} pts", className="big"),
                    html.Div(f"Scenario − Actual · {label}", className="lbl")]
    out["persistence"] = [
        html.Div([html.Span(className="swatch", style={"background": ACTUAL}), "Actual regime ",
                  html.B(score.label(score.top_actual)), f" · expected ≈ {score.persistence_actual:.1f} months"]),
        html.Div([html.Span(className="swatch", style={"background": SCENARIO}), "Scenario regime ",
                  html.B(score.label(score.top_scenario)), f" · expected ≈ {score.persistence_scenario:.1f} months"]),
    ]
    out["distribution"] = distribution_figure(score)
    try:
        out["history"] = F.history_figure(build_history(services.fetcher, engine, 15), score.labels.labels, engine.training_end)
    except Exception as exc:  # noqa: BLE001 - history is decorative; never block the score
        log.warning("history figure failed: %s", exc)
    legend_children: list[Any] = [legend_table(score)]
    if score.labels.provisional:
        legend_children.append(html.Div(
            f"Labels are provisional (heuristic) for fit {score.fit_id}. Confirm with "
            "`python -m model.labeler accept` or `python -m model.labeler set 0=bearish 1=transition 2=bullish`.",
            className="note"))
    out["legend"] = legend_children

    for snap in score.indicators:
        out["actual"][snap.key] = [html.Div(_fmt(snap.actual, snap.unit), className="value"),
                                   html.Div(f"FRED · as of {snap.actual_date:%b %Y}", className="asof")]
        feat = f"model input {snap.feature_name}: {snap.feature_actual:+.2f}"
        if snap.overridden:
            feat += f" → {snap.feature_scenario:+.2f}"
        out["feature"][snap.key] = feat
        out["badge"][snap.key] = (
            (f"⚠ Using override: {_fmt(snap.effective, snap.unit)}", "badge on") if snap.overridden
            else ("FRED value", "badge")
        )

    refreshed = f" · cache refreshed {score.refreshed_at[:16].replace('T', ' ')} UTC" if score.refreshed_at else ""
    out["status"] = (f"as of {score.as_of:%b %Y} · fit {score.fit_id} trained to {score.training_end:%b %Y}"
                     f" · {score.n_states} states{refreshed}")
    if score.has_overrides:
        n = sum(1 for i in score.indicators if i.overridden)
        out["banner"], out["banner_class"] = f"Scenario mode: {n} override(s) in effect on {score.as_of:%b %Y}.", "banner info"
    return out


def sync_overrides(services: Services, triggered: str | None, values: dict[str, Any]) -> dict[str, Any]:
    """Apply the input values to the store; return the values the inputs should show."""
    INDICATOR_KEYS = services.fetcher.keys
    store = services.overrides
    if triggered == "clear-btn":
        store.clear_all()
        return {k: None for k in INDICATOR_KEYS}
    if triggered is None:  # initial page load: inputs show what the session file holds
        return {k: store.get_override(k) for k in INDICATOR_KEYS}
    if triggered.startswith("override-"):
        key = triggered[len("override-"):]
        get_spec(key)
        value = values.get(key)
        if value is None or value == "":
            store.clear_override(key)
        else:
            try:
                store.set_override(key, value)
            except ValueError:
                store.clear_override(key)
    return {k: store.get_override(k) for k in INDICATOR_KEYS}


# ------------------------------------------------------------------- layout
def indicator_card(spec) -> html.Div:
    return html.Div(
        [
            html.H3(spec.name),
            html.Div(f"{spec.headline_label} ({spec.headline_unit})", className="sub"),
            html.Div(id=f"actual-{spec.key}", children="—"),
            html.Div(id=f"feature-{spec.key}", className="feat"),
            html.Label(f"Override ({spec.headline_unit})"),
            dcc.Input(id=f"override-{spec.key}", type="number", debounce=True, step="any",
                      placeholder="leave blank to use FRED"),
            html.Div(id=f"badge-{spec.key}", children="FRED value", className="badge"),
        ],
        className="card",
    )


def build_layout(indicators) -> html.Div:
    return html.Div(
        [
            dcc.Store(id="override-version", data=0),
            html.Div(
                [
                    html.Div([html.H1("Macro Regime Engine"), html.Div(id="status", className="status")]),
                    html.Div(
                        [html.Button("Refresh from FRED", id="refresh-btn", n_clicks=0, className="btn btn-primary"),
                         html.Button("Clear overrides", id="clear-btn", n_clicks=0, className="btn",
                                     style={"marginLeft": "8px"})]),
                ],
                className="topbar",
            ),
            html.Div(id="banner", className="banner"),
            html.Div([indicator_card(spec) for spec in indicators], className="cards"),
            html.Div(
                [
                    html.H2("Regime confidence"),
                    dcc.Dropdown(id="regime-select", options=[], value=None, clearable=False,
                                 placeholder="regime to track", className="select"),
                    html.Div(
                        [dcc.Graph(id="gauge-actual", config={"displayModeBar": False}),
                         html.Div(id="delta-tile", className="delta"),
                         dcc.Graph(id="gauge-scenario", config={"displayModeBar": False})],
                        className="gauges",
                    ),
                    html.Div(id="persistence", className="persist"),
                ],
                className="section",
            ),
            html.Div([html.H2("Full state distribution"),
                      dcc.Graph(id="distribution", config={"displayModeBar": False})], className="section"),
            html.Div([html.H2("Regime history · filtered probabilities, last 15 years"),
                      dcc.Graph(id="history", config={"displayModeBar": False})], className="section"),
            html.Div([html.H2("State legend"), html.Div(id="legend")], className="section"),
        ],
        className="page",
    )


def create_app(services: Services | None = None) -> Dash:
    services = services or Services()
    app = Dash(__name__, title="Macro Regime Engine")
    app.index_string = INDEX_STRING
    app.layout = build_layout(services.fetcher.indicators)
    app._services = services  # type: ignore[attr-defined]
    INDICATOR_KEYS = services.fetcher.keys

    override_inputs = [Input(f"override-{k}", "value") for k in INDICATOR_KEYS]
    override_outputs = [Output(f"override-{k}", "value") for k in INDICATOR_KEYS]

    @app.callback(
        Output("override-version", "data"), *override_outputs,
        Input("clear-btn", "n_clicks"), *override_inputs,
        State("override-version", "data"),
        prevent_initial_call=False,
    )
    def _sync(_clear, *args):
        values = dict(zip(INDICATOR_KEYS, args[: len(INDICATOR_KEYS)]))
        version = args[len(INDICATOR_KEYS)] or 0
        shown = sync_overrides(services, ctx.triggered_id, values)
        return (version + 1, *[shown[k] for k in INDICATOR_KEYS])

    outputs = [
        Output("banner", "children"), Output("banner", "className"), Output("status", "children"),
        Output("regime-select", "options"), Output("regime-select", "value"),
        Output("gauge-actual", "figure"), Output("gauge-scenario", "figure"), Output("delta-tile", "children"),
        Output("persistence", "children"), Output("distribution", "figure"), Output("history", "figure"),
        Output("legend", "children"),
        *[Output(f"actual-{k}", "children") for k in INDICATOR_KEYS],
        *[Output(f"feature-{k}", "children") for k in INDICATOR_KEYS],
        *[Output(f"badge-{k}", "children") for k in INDICATOR_KEYS],
        *[Output(f"badge-{k}", "className") for k in INDICATOR_KEYS],
    ]

    @app.callback(
        *outputs,
        Input("override-version", "data"), Input("refresh-btn", "n_clicks"), Input("regime-select", "value"),
    )
    def _render(_version, _refresh_clicks, tracked):
        if ctx.triggered_id == "refresh-btn":
            try:
                services.fetcher.refresh()  # actuals only; overrides are untouched
            except FredError as exc:
                log.error("refresh failed: %s", exc)
                r = render(services, tracked)
                r["banner"], r["banner_class"] = f"Refresh from FRED failed: {exc}", "banner error"
                return _flatten(r)
        return _flatten(render(services, tracked))

    return app


def _flatten(r: dict[str, Any]) -> tuple:
    INDICATOR_KEYS = tuple(r["actual"])
    return (
        r["banner"], r["banner_class"], r["status"], r["regime_options"], r["regime_value"],
        r["gauge_actual"], r["gauge_scenario"], r["delta"], r["persistence"], r["distribution"], r["history"], r["legend"],
        *[r["actual"][k] for k in INDICATOR_KEYS],
        *[r["feature"][k] for k in INDICATOR_KEYS],
        *[r["badge"][k][0] for k in INDICATOR_KEYS],
        *[r["badge"][k][1] for k in INDICATOR_KEYS],
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    app.run(host=config.DASH_HOST, port=config.DASH_PORT, debug=False)


if __name__ == "__main__":
    main()
