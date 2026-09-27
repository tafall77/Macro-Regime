"""Indicator registry and the stationary transforms applied to raw FRED series.

There are two layers between FRED and the HMM:

    raw FRED series --headline_fn--> headline --feature_fn--> feature

* The **headline** is the number a human reads and overrides ("PMI = 47",
  "10Y-2Y spread = -0.30", "Core PCE YoY = 2.8%", "Fed Funds = 5.25%").
* The **feature** is the stationary form the HMM is fitted on. For the yield
  spread and Core PCE YoY the headline already is the feature; the Fed Funds
  rate becomes an N-month rate-of-change and the PMI becomes its distance
  from the 50 expansion/contraction line.

Keeping the override in headline units means history-dependent transforms
(the Fed Funds rate-of-change) are recomputed against the actual history, so
a scenario can never be internally inconsistent.
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


def to_month_end(series: pd.Series) -> pd.Series:
    """Collapse any frequency to a complete month-end index (last obs per month).

    Months with no observation are kept as NaN so that row shifts equal month
    shifts for the transforms below.
    """
    s = series.dropna().sort_index()
    if s.empty:
        return s
    return s.resample(MONTH_END).last()


def full_monthly_index(index: pd.Index) -> pd.DatetimeIndex:
    return pd.date_range(month_end(index.min()), month_end(index.max()), freq=MONTH_END)


# --- headline transforms: {series_id: monthly series} -> headline series -----

def yield_spread(m: Mapping[str, pd.Series]) -> pd.Series:
    """10Y minus 2Y Treasury yield, percentage points."""
    return m["DGS10"] - m["DGS2"]


def core_pce_yoy(m: Mapping[str, pd.Series]) -> pd.Series:
    """Year-over-year % change of the Core PCE price index."""
    s = m["PCEPILFE"]
    return (s / s.shift(12) - 1.0) * 100.0


def fed_funds_level(m: Mapping[str, pd.Series]) -> pd.Series:
    return m["FEDFUNDS"]


def pmi_level(m: Mapping[str, pd.Series]) -> pd.Series:
    return m[config.PMI_SERIES_ID]


# --- feature transforms: headline series -> feature series -------------------

def identity(s: pd.Series) -> pd.Series:
    return s.copy()


def diff_n(n: int) -> Callable[[pd.Series], pd.Series]:
    def _diff(s: pd.Series) -> pd.Series:
        return s - s.shift(n)

    _diff.__name__ = f"diff_{n}m"
    return _diff


def minus_50(s: pd.Series) -> pd.Series:
    return s - 50.0


@dataclass(frozen=True)
class IndicatorSpec:
    key: str
    name: str
    series_ids: tuple[str, ...]
    headline_label: str
    headline_unit: str
    feature_name: str
    feature_unit: str
    headline_fn: Callable[[Mapping[str, pd.Series]], pd.Series]
    feature_fn: Callable[[pd.Series], pd.Series]
    bullish_sign: int  # +1 when a higher feature value reads pro-growth, -1 otherwise
    description: str = ""

    def headline(self, raw: Mapping[str, pd.Series]) -> pd.Series:
        missing = [sid for sid in self.series_ids if sid not in raw]
        if missing:
            raise KeyError(f"{self.key}: missing raw series {missing}")
        monthly = {sid: to_month_end(raw[sid]) for sid in self.series_ids}
        out = self.headline_fn(monthly).dropna()
        out.name = self.key
        return out

    def feature(self, headline: pd.Series) -> pd.Series:
        out = self.feature_fn(headline)
        out.name = self.feature_name
        return out


INDICATORS: tuple[IndicatorSpec, ...] = (
    IndicatorSpec(
        key="yield_curve",
        name="Yield curve",
        series_ids=("DGS10", "DGS2"),
        headline_label="10Y − 2Y spread",
        headline_unit="pp",
        feature_name="yc_spread",
        feature_unit="pp",
        headline_fn=yield_spread,
        feature_fn=identity,
        bullish_sign=+1,
        description="Month-end 10-year minus 2-year Treasury yield.",
    ),
    IndicatorSpec(
        key="core_pce",
        name="Core PCE",
        series_ids=("PCEPILFE",),
        headline_label="Core PCE YoY",
        headline_unit="%",
        feature_name="core_pce_yoy",
        feature_unit="% YoY",
        headline_fn=core_pce_yoy,
        feature_fn=identity,
        bullish_sign=-1,
        description="Year-over-year change in the Core PCE price index.",
    ),
    IndicatorSpec(
        key="fed_funds",
        name="Fed Funds",
        series_ids=("FEDFUNDS",),
        headline_label="Effective Fed Funds",
        headline_unit="%",
        feature_name=f"fed_funds_chg_{config.FED_FUNDS_ROC_MONTHS}m",
        feature_unit=f"pp / {config.FED_FUNDS_ROC_MONTHS}m",
        headline_fn=fed_funds_level,
        feature_fn=diff_n(config.FED_FUNDS_ROC_MONTHS),
        bullish_sign=-1,
        description=f"{config.FED_FUNDS_ROC_MONTHS}-month change in the effective Fed Funds rate.",
    ),
    IndicatorSpec(
        key="ism_pmi",
        name="ISM Manufacturing PMI",
        series_ids=(config.PMI_SERIES_ID,),
        headline_label="PMI",
        headline_unit="index",
        feature_name="pmi_vs_50",
        feature_unit="pts vs 50",
        headline_fn=pmi_level,
        feature_fn=minus_50,
        bullish_sign=+1,
        description="PMI level relative to the 50 expansion/contraction line.",
    ),
)

INDICATOR_MAP: dict[str, IndicatorSpec] = {spec.key: spec for spec in INDICATORS}
INDICATOR_KEYS: tuple[str, ...] = tuple(INDICATOR_MAP)
FEATURE_NAMES: tuple[str, ...] = tuple(spec.feature_name for spec in INDICATORS)
FEATURE_SIGNS: dict[str, int] = {spec.feature_name: spec.bullish_sign for spec in INDICATORS}
ALL_SERIES_IDS: tuple[str, ...] = tuple(
    dict.fromkeys(sid for spec in INDICATORS for sid in spec.series_ids)
)


def get_spec(indicator: str) -> IndicatorSpec:
    try:
        return INDICATOR_MAP[indicator]
    except KeyError:
        raise KeyError(
            f"unknown indicator {indicator!r}; expected one of {list(INDICATOR_MAP)}"
        ) from None


def headline_panel(
    raw: Mapping[str, pd.Series], indicators: Iterable[IndicatorSpec] = INDICATORS
) -> pd.DataFrame:
    """Build the monthly headline panel (columns = indicator keys) from raw series."""
    cols = [spec.headline(raw) for spec in indicators]
    panel = pd.concat(cols, axis=1).sort_index()
    panel = panel.reindex(full_monthly_index(panel.index))
    panel.index.name = "date"
    return panel


def features_from_panel(
    panel: pd.DataFrame, indicators: Iterable[IndicatorSpec] = INDICATORS
) -> pd.DataFrame:
    """Apply each indicator's feature transform to a headline panel.

    The panel is reindexed onto a complete monthly range first so that row-based
    shifts (``diff``) are month-based shifts.
    """
    panel = panel.reindex(full_monthly_index(panel.index))
    cols = {spec.feature_name: spec.feature(panel[spec.key]) for spec in indicators}
    feats = pd.DataFrame(cols, index=panel.index)
    feats.index.name = "date"
    return feats
