"""Indicator catalogue and the stationary transforms applied to raw FRED series.

There are two layers between FRED and the HMM:

    raw FRED series --headline_fn--> headline --feature_fn--> feature

* The **headline** is the number a human reads and overrides ("PMI = 47",
  "10Y-2Y spread = -0.30", "Core PCE YoY = 2.8%", "Unemployment = 4.1%").
* The **feature** is the stationary form the HMM is fitted on (a spread, a
  year-over-year change, a distance from 50, ...).

Keeping the override in headline units means history-dependent transforms are
recomputed against the actual history, so a scenario can never be internally
inconsistent.

``CATALOGUE`` holds every indicator the engine knows about; ``INDICATORS`` is the
active subset selected by ``MACRO_REGIME_INDICATORS`` (see :mod:`config`).
Indicators flagged ``manual_only`` are not published on FRED and are read from
``<manual_dir>/<SERIES_ID>.csv`` instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import pandas as pd

import config

MONTH_END = "ME"


def month_end(date) -> pd.Timestamp:
    """Normalise any date-like to the month-end timestamp of its month."""
    return pd.Timestamp(date).normalize() + pd.offsets.MonthEnd(0)


def months_between(earlier, later) -> int:
    a, b = pd.Timestamp(earlier), pd.Timestamp(later)
    return (b.year - a.year) * 12 + (b.month - a.month)


def to_month_end(series: pd.Series, how: str = "last") -> pd.Series:
    """Collapse any frequency to a complete month-end index.

    ``how`` is ``"last"`` (last observation in the month, right for levels and
    daily yields) or ``"mean"`` (right for weekly flows such as jobless claims).
    Months with no observation are kept as NaN so that row shifts equal month
    shifts for the transforms below.
    """
    s = series.dropna().sort_index()
    if s.empty:
        return s
    r = s.resample(MONTH_END)
    return r.mean() if how == "mean" else r.last()


def full_monthly_index(index: pd.Index) -> pd.DatetimeIndex:
    return pd.date_range(month_end(index.min()), month_end(index.max()), freq=MONTH_END)


# --- headline transforms: {series_id: monthly series} -> headline series -----

def _yoy(s: pd.Series) -> pd.Series:
    return (s / s.shift(12) - 1.0) * 100.0


def yield_spread(m):        # 10Y minus 2Y Treasury yield, pp
    return m["DGS10"] - m["DGS2"]


def core_pce_yoy(m):
    return _yoy(m["PCEPILFE"])


def fed_funds_level(m):
    return m["FEDFUNDS"]


def pmi_level(m):
    return m[config.PMI_SERIES_ID]


def credit_spread_level(m):  # Moody's Baa corporate yield minus 10Y Treasury, pp
    return m["BAA10Y"]


def unemployment_level(m):
    return m["UNRATE"]


def claims_thousands(m):     # monthly average of weekly initial claims, thousands
    return m["ICSA"] / 1000.0


def payrolls_3m_change(m):   # average monthly change in nonfarm payrolls over 3 months, thousands
    return m["PAYEMS"].diff(1).rolling(3).mean()


def cpi_yoy(m):
    return _yoy(m["CPIAUCSL"])


def core_cpi_yoy(m):
    return _yoy(m["CPILFESL"])


def ppi_mom(m):              # month-over-month % change in the producer price index
    return m["PPIACO"].pct_change(1) * 100.0


def sentiment_level(m):
    return m["UMCSENT"]


def services_pmi_level(m):
    return m["ISM_SERVICES_PMI"]


def chicago_pmi_level(m):
    return m["CHICAGO_PMI"]


def conf_board_level(m):
    return m["CONF_BOARD_CCI"]


# --- feature transforms: headline series -> feature series -------------------

def identity(s: pd.Series) -> pd.Series:
    return s.copy()


def diff_n(n: int) -> Callable[[pd.Series], pd.Series]:
    def _diff(s: pd.Series) -> pd.Series:
        return s - s.shift(n)

    _diff.__name__ = f"diff_{n}m"
    return _diff


def pct_change_n(n: int) -> Callable[[pd.Series], pd.Series]:
    def _pct(s: pd.Series) -> pd.Series:
        return (s / s.shift(n) - 1.0) * 100.0

    _pct.__name__ = f"pct_change_{n}m"
    return _pct


def rolling_mean_n(n: int) -> Callable[[pd.Series], pd.Series]:
    def _rm(s: pd.Series) -> pd.Series:
        return s.rolling(n).mean()

    _rm.__name__ = f"rolling_mean_{n}m"
    return _rm


def minus_50(s: pd.Series) -> pd.Series:
    return s - 50.0


@dataclass(frozen=True)
class IndicatorSpec:
    key: str
    name: str
    category: str
    series_ids: tuple[str, ...]
    headline_label: str
    headline_unit: str
    feature_name: str
    feature_unit: str
    headline_fn: Callable[[Mapping[str, pd.Series]], pd.Series]
    feature_fn: Callable[[pd.Series], pd.Series]
    bullish_sign: int  # +1 when a higher feature value reads pro-growth, -1 otherwise
    description: str = ""
    monthly_agg: str = "last"   # how sub-monthly raw data collapses to months: "last" or "mean"
    manual_only: bool = False   # not on FRED: read from <manual_dir>/<SERIES_ID>.csv

    def headline(self, raw: Mapping[str, pd.Series]) -> pd.Series:
        missing = [sid for sid in self.series_ids if sid not in raw]
        if missing:
            raise KeyError(f"{self.key}: missing raw series {missing}")
        monthly = {sid: to_month_end(raw[sid], self.monthly_agg) for sid in self.series_ids}
        out = self.headline_fn(monthly).dropna()
        out.name = self.key
        return out

    def feature(self, headline: pd.Series) -> pd.Series:
        out = self.feature_fn(headline)
        out.name = self.feature_name
        return out


_FF_N = config.FED_FUNDS_ROC_MONTHS

CATALOGUE: tuple[IndicatorSpec, ...] = (
    # --- interest rates / credit ------------------------------------------------
    IndicatorSpec("yield_curve", "Yield curve", "Interest rates / credit", ("DGS10", "DGS2"),
                  "10Y − 2Y spread", "pp", "yc_spread", "pp", yield_spread, identity, +1,
                  "Month-end 10-year minus 2-year Treasury yield. Inversions precede recessions."),
    IndicatorSpec("fed_funds", "Fed Funds", "Interest rates / credit", ("FEDFUNDS",),
                  "Effective Fed Funds", "%", f"fed_funds_chg_{_FF_N}m", f"pp / {_FF_N}m", fed_funds_level, diff_n(_FF_N), -1,
                  f"{_FF_N}-month change in the effective Fed Funds rate: hiking vs cutting."),
    IndicatorSpec("credit_spread", "Credit spread", "Interest rates / credit", ("BAA10Y",),
                  "Baa − 10Y spread", "pp", "credit_spread", "pp", credit_spread_level, identity, -1,
                  "Moody's Baa corporate yield over the 10-year Treasury: the price of credit risk."),
    # --- inflation --------------------------------------------------------------
    IndicatorSpec("core_pce", "Core PCE", "Inflation", ("PCEPILFE",),
                  "Core PCE YoY", "%", "core_pce_yoy", "% YoY", core_pce_yoy, identity, -1,
                  "Year-over-year change in the Core PCE price index, the Fed's preferred gauge."),
    IndicatorSpec("cpi", "CPI", "Inflation", ("CPIAUCSL",),
                  "CPI YoY", "%", "cpi_yoy", "% YoY", cpi_yoy, identity, -1,
                  "Headline consumer price inflation, year over year."),
    IndicatorSpec("core_cpi", "Core CPI", "Inflation", ("CPILFESL",),
                  "Core CPI YoY", "%", "core_cpi_yoy", "% YoY", core_cpi_yoy, identity, -1,
                  "CPI excluding food and energy, year over year. Highly collinear with Core PCE."),
    IndicatorSpec("ppi", "PPI", "Inflation", ("PPIACO",),
                  "PPI MoM", "%", "ppi_mom_3m", "% MoM, 3m avg", ppi_mom, rolling_mean_n(3), -1,
                  "Producer prices (all commodities), month over month, smoothed over 3 months."),
    # --- labour market ----------------------------------------------------------
    IndicatorSpec("unemployment", "Unemployment rate", "Labour market", ("UNRATE",),
                  "Unemployment rate", "%", "unemployment_chg_12m", "pp / 12m", unemployment_level, diff_n(12), -1,
                  "12-month change in the unemployment rate (rising unemployment is the recession signal)."),
    IndicatorSpec("jobless_claims", "Initial jobless claims", "Labour market", ("ICSA",),
                  "Initial claims (monthly avg)", "k", "claims_yoy", "% YoY", claims_thousands, pct_change_n(12), -1,
                  "Weekly initial claims averaged per month, year-over-year % change.", monthly_agg="mean"),
    IndicatorSpec("payrolls", "Nonfarm payrolls", "Labour market", ("PAYEMS",),
                  "Payroll change (3m avg)", "k", "payrolls_3m", "k / month", payrolls_3m_change, identity, +1,
                  "Average monthly change in nonfarm payrolls over the last 3 months."),
    # --- activity / sentiment ---------------------------------------------------
    IndicatorSpec("ism_pmi", "ISM Manufacturing PMI", "Activity / sentiment", (config.PMI_SERIES_ID,),
                  "PMI", "index", "pmi_vs_50", "pts vs 50", pmi_level, minus_50, +1,
                  "PMI level relative to the 50 expansion/contraction line."),
    IndicatorSpec("consumer_sentiment", "Consumer sentiment (U. Michigan)", "Activity / sentiment", ("UMCSENT",),
                  "Sentiment index", "index", "sentiment_chg_12m", "pts / 12m", sentiment_level, diff_n(12), +1,
                  "University of Michigan consumer sentiment, 12-month change. FRED proxy for confidence surveys."),
    IndicatorSpec("ism_services_pmi", "ISM Services PMI", "Activity / sentiment", ("ISM_SERVICES_PMI",),
                  "Services PMI", "index", "services_pmi_vs_50", "pts vs 50", services_pmi_level, minus_50, +1,
                  "ISM Non-Manufacturing PMI. Not on FRED: supply manual/ISM_SERVICES_PMI.csv.", manual_only=True),
    IndicatorSpec("chicago_pmi", "Chicago PMI", "Activity / sentiment", ("CHICAGO_PMI",),
                  "Chicago PMI", "index", "chicago_pmi_vs_50", "pts vs 50", chicago_pmi_level, minus_50, +1,
                  "MNI Chicago Business Barometer. Not on FRED: supply manual/CHICAGO_PMI.csv.", manual_only=True),
    IndicatorSpec("consumer_confidence", "Consumer confidence (Conference Board)", "Activity / sentiment", ("CONF_BOARD_CCI",),
                  "Confidence index", "index", "confidence_chg_12m", "pts / 12m", conf_board_level, diff_n(12), +1,
                  "Conference Board Consumer Confidence, 12-month change. Not on FRED: supply manual/CONF_BOARD_CCI.csv.",
                  manual_only=True),
)

CATALOGUE_MAP: dict[str, IndicatorSpec] = {spec.key: spec for spec in CATALOGUE}
CATALOGUE_KEYS: tuple[str, ...] = tuple(CATALOGUE_MAP)
MANUAL_ONLY_SERIES: frozenset[str] = frozenset(sid for s in CATALOGUE if s.manual_only for sid in s.series_ids)


def get_spec(indicator: str) -> IndicatorSpec:
    try:
        return CATALOGUE_MAP[indicator]
    except KeyError:
        raise KeyError(f"unknown indicator {indicator!r}; expected one of {list(CATALOGUE_MAP)}") from None


def select(keys: Iterable[str]) -> tuple[IndicatorSpec, ...]:
    """Catalogue specs for ``keys``, in the given order (KeyError for unknown keys)."""
    return tuple(get_spec(k) for k in keys)


def series_ids_for(indicators: Iterable[IndicatorSpec]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(sid for spec in indicators for sid in spec.series_ids))


# The active set: what the fetcher, model, dashboard and report use by default.
INDICATORS: tuple[IndicatorSpec, ...] = select(config.ACTIVE_INDICATORS)
INDICATOR_MAP: dict[str, IndicatorSpec] = {spec.key: spec for spec in INDICATORS}
INDICATOR_KEYS: tuple[str, ...] = tuple(INDICATOR_MAP)
FEATURE_NAMES: tuple[str, ...] = tuple(spec.feature_name for spec in INDICATORS)
FEATURE_SIGNS: dict[str, int] = {spec.feature_name: spec.bullish_sign for spec in INDICATORS}
ALL_SERIES_IDS: tuple[str, ...] = series_ids_for(INDICATORS)


def headline_panel(raw: Mapping[str, pd.Series], indicators: Iterable[IndicatorSpec] = INDICATORS) -> pd.DataFrame:
    """Build the monthly headline panel (columns = indicator keys) from raw series."""
    cols = [spec.headline(raw) for spec in indicators]
    panel = pd.concat(cols, axis=1).sort_index()
    panel = panel.reindex(full_monthly_index(panel.index))
    panel.index.name = "date"
    return panel


def features_from_panel(panel: pd.DataFrame, indicators: Iterable[IndicatorSpec] | None = None) -> pd.DataFrame:
    """Apply each indicator's feature transform to a headline panel.

    The panel is reindexed onto a complete monthly range first so that row-based
    shifts (``diff``) are month-based shifts. Without ``indicators`` the panel's
    own columns select the specs.
    """
    specs = tuple(indicators) if indicators is not None else select(panel.columns)
    panel = panel.reindex(full_monthly_index(panel.index))
    cols = {spec.feature_name: spec.feature(panel[spec.key]) for spec in specs}
    feats = pd.DataFrame(cols, index=panel.index)
    feats.index.name = "date"
    return feats
