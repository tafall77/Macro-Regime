"""Self-contained HTML report of the current regime score.

In a notebook, the :class:`Report` object renders inline (``_repr_html_``); it
can also be saved as a single HTML file (``report.save("regime.html")``).
Figures are Plotly, so they stay interactive in both places.
"""
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.io as pio

from data.fred_fetcher import FredFetcher
from model.hmm_engine import HMMEngine
from reporting import figures as F
from scoring.live_score import LiveScore, actual_panel_asof

_CSS = f"""
.mr-report {{ font-family: {F.FONT}; color: {F.INK}; background: {F.PLANE}; padding: 16px; border-radius: 12px; max-width: 1200px; }}
.mr-report h1 {{ font-size: 20px; margin: 0 0 2px; }}
.mr-report h2 {{ font-size: 15px; margin: 0 0 8px; }}
.mr-report .status {{ color: {F.INK2}; font-size: 13px; margin-bottom: 12px; }}
.mr-report .banner {{ padding: 8px 12px; border-radius: 8px; font-size: 13px; margin-bottom: 12px;
  background: #e6f0fb; color: #14406f; border: 1px solid #b9d3f2; }}
.mr-report .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; }}
.mr-report .card, .mr-report .section {{ background: {F.SURFACE}; border: 1px solid {F.BORDER}; border-radius: 12px; padding: 12px 14px; }}
.mr-report .section {{ margin-top: 12px; }}
.mr-report .card h3 {{ margin: 0; font-size: 14px; }}
.mr-report .card .sub {{ color: {F.MUTED}; font-size: 12px; }}
.mr-report .card .value {{ font-size: 24px; font-weight: 600; margin-top: 4px; }}
.mr-report .card .meta {{ color: {F.INK2}; font-size: 12px; }}
.mr-report .badge {{ display: inline-block; margin-top: 6px; font-size: 12px; padding: 2px 8px; border-radius: 999px; border: 1px solid {F.BORDER}; color: {F.INK2}; }}
.mr-report .badge.on {{ border-color: {F.WARNING}; background: #fff5dc; color: #5a3d00; font-weight: 600; }}
.mr-report .gauges {{ display: grid; grid-template-columns: 1fr auto 1fr; gap: 8px; align-items: center; }}
.mr-report .delta {{ text-align: center; min-width: 140px; }}
.mr-report .delta .big {{ font-size: 28px; font-weight: 650; white-space: nowrap; }}
.mr-report .delta .lbl {{ color: {F.INK2}; font-size: 12px; }}
.mr-report .persist {{ display: flex; flex-wrap: wrap; gap: 20px; color: {F.INK2}; font-size: 13px; margin-top: 4px; }}
.mr-report .persist b {{ color: {F.INK}; }}
.mr-report .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }}
.mr-report table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
.mr-report th, .mr-report td {{ text-align: right; padding: 5px 8px; border-bottom: 1px solid {F.GRID}; font-variant-numeric: tabular-nums; }}
.mr-report th:first-child, .mr-report td:first-child, .mr-report th:nth-child(2), .mr-report td:nth-child(2) {{ text-align: left; }}
.mr-report th {{ color: {F.MUTED}; font-weight: 500; }}
.mr-report .note {{ color: {F.MUTED}; font-size: 12px; margin-top: 6px; }}
@media (max-width: 760px) {{ .mr-report .gauges {{ grid-template-columns: 1fr; }} }}
"""


def get_plotlyjs() -> str:
    """The bundled plotly.js source (its location moved between plotly.py releases)."""
    try:
        from plotly.offline import get_plotlyjs as _get
    except ImportError:  # pragma: no cover
        _get = pio.get_plotlyjs  # type: ignore[attr-defined]
    return _get()


def plotly_cdn_url() -> str:
    """The plotly.js build matching the installed plotly.py, as plotly itself would embed it."""
    import plotly.graph_objects as go

    m = re.search(r'src="([^"]+)"', pio.to_html(go.Figure(), include_plotlyjs="cdn", full_html=False))
    return m.group(1) if m else "https://cdn.plot.ly/plotly-latest.min.js"


def plotly_loader_script() -> str:
    """A loader that works in a plain page, JupyterLab (no AMD) and the classic notebook (require.js).

    plotly.js registers as an AMD module when ``define`` exists, so a bare <script src> tag never sets
    ``window.Plotly`` under require.js; this shim goes through ``require`` in that case.
    """
    url = plotly_cdn_url()
    return f"""<script>
(function () {{
  if (window.__mrPlotlyReady) return;
  var CDN = {json.dumps(url)}, queue = [], loading = false;
  function flush() {{ while (queue.length) {{ try {{ queue.shift()(); }} catch (e) {{ console.error(e); }} }} }}
  window.__mrPlotlyReady = function (cb) {{
    queue.push(cb);
    if (window.Plotly) return flush();
    if (loading) return;
    loading = true;
    if (typeof require === "function" && typeof requirejs !== "undefined") {{
      requirejs.config({{ paths: {{ "plotly-mr": [CDN.replace(/\.js$/, "")] }} }});
      require(["plotly-mr"], function (P) {{ window.Plotly = P; flush(); }});
    }} else {{
      var s = document.createElement("script"); s.src = CDN; s.charset = "utf-8";
      s.onload = flush; document.head.appendChild(s);
    }}
  }};
}})();
</script>"""


def _fmt(value: float, unit: str) -> str:
    if unit == "index":
        return f"{value:.1f}"
    if unit == "pp":
        return f"{value:+.2f} pp"
    return f"{value:.2f}%" if unit == "%" else f"{value:.2f} {unit}"


def _fig_html(fig, div_id: str) -> str:
    """A div plus a script that draws the figure once plotly.js is available (see plotly_loader_script)."""
    spec = pio.to_json(fig, validate=False)
    height = fig.layout.height or 300
    return (
        f'<div id="{div_id}" class="mr-plot" style="width:100%;height:{height}px"></div>'
        f'<script>window.__mrPlotlyReady(function () {{ var f = {spec}; '
        f'Plotly.newPlot("{div_id}", f.data, f.layout, {{displayModeBar: false, responsive: true}}); }});</script>'
    )


def build_history(fetcher: FredFetcher, engine: HMMEngine, years: int | None = 15) -> pd.DataFrame:
    """Filtered posteriors: training path plus any months after the last refit."""
    hist = engine.filtered_.copy()
    panel, as_of = actual_panel_asof(fetcher)
    feats = fetcher.features(panel).loc[:as_of]
    tail = feats.loc[feats.index > engine.training_end].dropna()
    if not tail.empty:
        hist = pd.concat([hist, engine.score_path(tail)])
    if years:
        hist = hist.loc[hist.index[-1] - pd.DateOffset(years=years):]
    return hist


@dataclass
class Report:
    html: str
    score: LiveScore
    generated_at: datetime

    def _repr_html_(self) -> str:
        return self.html

    def save(self, path: str | Path, inline_js: bool = False) -> Path:
        """Write a standalone HTML file. ``inline_js=True`` embeds plotly.js (~3 MB) for offline viewing."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        script = f"<script>{get_plotlyjs()}</script>" if inline_js else ""
        page = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Macro Regime report · {self.score.as_of:%b %Y}</title>{script}</head><body style='margin:0;background:{F.PLANE}'>"
            f"{self.html}</body></html>"
        )
        path.write_text(page, encoding="utf-8")
        return path


def build_report(score: LiveScore, fetcher: FredFetcher, engine: HMMEngine, *, history_years: int | None = 15,
                 title: str = "Macro Regime Engine", include_indicator_history: bool = True) -> Report:
    s = score
    e = html.escape
    tracked = s.top_actual
    label = s.label(tracked)
    pa, ps = s.prob(tracked), s.prob(tracked, True)
    d = (ps - pa) * 100
    arrow = "▲" if d > 0.05 else ("▼" if d < -0.05 else "•")
    labels = s.labels.labels

    cards = []
    for i in s.indicators:
        badge = (f'<span class="badge on">⚠ Using override: {e(_fmt(i.effective, i.unit))}</span>' if i.overridden
                 else '<span class="badge">FRED value</span>')
        feat = f"model input {e(i.feature_name)}: {i.feature_actual:+.2f}" + (f" → {i.feature_scenario:+.2f}" if i.overridden else "")
        cards.append(
            f'<div class="card"><h3>{e(i.name)}</h3><div class="sub">{e(i.headline_label)} ({e(i.unit)})</div>'
            f'<div class="value">{e(_fmt(i.actual, i.unit))}</div><div class="meta">FRED · as of {i.actual_date:%b %Y}</div>'
            f'<div class="meta">{feat}</div>{badge}</div>'
        )

    rows = []
    for k in range(s.n_states):
        means = s.state_table.loc[k]
        cells = "".join(f"<td>{means[f]:+.2f}</td>" for f in s.state_table.columns)
        rows.append(f"<tr><td>{k}</td><td><span class='swatch' style='background:{F.state_color(k, labels.get(k))}'></span>{e(s.label(k))}</td>"
                    f"{cells}<td>{s.persistence[k]:.1f}</td><td>{s.prob(k) * 100:.0f}%</td><td>{s.prob(k, True) * 100:.0f}%</td></tr>")
    head = "".join(f"<th>{e(f)}</th>" for f in s.state_table.columns)
    legend = (f"<table><thead><tr><th>State</th><th>Label</th>{head}<th>Persistence (mo)</th><th>Actual</th><th>Scenario</th></tr></thead>"
              f"<tbody>{''.join(rows)}</tbody></table>")
    if s.labels.provisional:
        legend += (f'<div class="note">Labels are provisional (heuristic) for fit {e(s.fit_id)}. Confirm with '
                   f'<code>python -m model.labeler accept</code> or <code>MacroRegime.label(...)</code>.</div>')

    names = [s.label(k) for k in range(s.n_states)]
    gauge_a = _fig_html(F.gauge_figure(f"Actual · {label}", pa, F.ACTUAL, F.ACTUAL_TRACK), "mr-gauge-actual")
    gauge_s = _fig_html(F.gauge_figure(f"Scenario · {label}", ps, F.SCENARIO, F.SCENARIO_TRACK, reference=pa), "mr-gauge-scenario")
    dist = _fig_html(F.distribution_figure(names, [s.prob(k) for k in range(s.n_states)],
                                           [s.prob(k, True) for k in range(s.n_states)]), "mr-dist")
    history = build_history(fetcher, engine, history_years)
    hist_html = _fig_html(F.history_figure(history, labels, engine.training_end), "mr-history")
    ind_html = ""
    if include_indicator_history:
        panel, _ = actual_panel_asof(fetcher)
        if history_years:
            panel = panel.loc[panel.index[-1] - pd.DateOffset(years=history_years):]
        ind_html = _fig_html(F.indicators_figure(panel, {i.key: f"{i.name} — {i.headline_label} ({i.unit})" for i in s.indicators}), "mr-ind")

    banner = ""
    if s.has_overrides:
        n = sum(1 for i in s.indicators if i.overridden)
        banner = f'<div class="banner">Scenario mode: {n} override(s) in effect on {s.as_of:%b %Y}.</div>'
    refreshed = f" · cache refreshed {e(s.refreshed_at[:16].replace('T', ' '))} UTC" if s.refreshed_at else ""
    generated = datetime.now(timezone.utc)

    body = f"""
<style>{_CSS}</style>
{plotly_loader_script()}
<div class="mr-report">
  <h1>{e(title)}</h1>
  <div class="status">as of {s.as_of:%b %Y} · fit {e(s.fit_id)} trained to {s.training_end:%b %Y} · {s.n_states} states{refreshed}
   · report {generated:%Y-%m-%d %H:%M} UTC</div>
  {banner}
  <div class="cards">{''.join(cards)}</div>
  <div class="section"><h2>Regime confidence · tracking the current actual regime ({e(label)})</h2>
    <div class="gauges">{gauge_a}
      <div class="delta"><div class="big">{arrow} {d:+.1f} pts</div><div class="lbl">Scenario − Actual · {e(label)}</div></div>
      {gauge_s}</div>
    <div class="persist">
      <div><span class="swatch" style="background:{F.ACTUAL}"></span>Actual regime <b>{e(s.label(s.top_actual))}</b> · expected ≈ {s.persistence_actual:.1f} months</div>
      <div><span class="swatch" style="background:{F.SCENARIO}"></span>Scenario regime <b>{e(s.label(s.top_scenario))}</b> · expected ≈ {s.persistence_scenario:.1f} months</div>
    </div></div>
  <div class="section"><h2>Full state distribution</h2>{dist}</div>
  <div class="section"><h2>Regime history · filtered probabilities{f', last {history_years} years' if history_years else ''}</h2>{hist_html}</div>
  {f'<div class="section"><h2>Indicators · headline series</h2>{ind_html}</div>' if ind_html else ''}
  <div class="section"><h2>State legend</h2>{legend}</div>
</div>
"""
    return Report(html=body, score=s, generated_at=generated)
