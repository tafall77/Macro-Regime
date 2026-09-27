"""Pull the macro series from the FRED API, transform, and cache them locally.

Cache layout (Parquet, keyed by date)::

    <cache_dir>/raw/<SERIES_ID>.parquet   one file per raw FRED series
    <cache_dir>/panel.parquet             monthly headline panel, one column per indicator
    <cache_dir>/meta.json                 refresh timestamp + latest observation per series

Optional hand-maintained supplements live in ``<manual_dir>/<SERIES_ID>.csv``
(columns ``date,value``). They are merged on top of the FRED pull, which is the
supported way to feed ISM PMI prints now that ISM no longer licenses them to FRED.

Public surface used by the rest of the engine:

* :meth:`FredFetcher.refresh` – re-pull everything from FRED and rewrite the cache.
* :meth:`FredFetcher.get_actual` – headline value of an indicator as of a date.
* :meth:`FredFetcher.training_data` – the *only* factory for :class:`TrainingData`,
  built from the actual cache and nothing else.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
import requests

import config
from data.transforms import (
    INDICATORS,
    IndicatorSpec,
    features_from_panel,
    get_spec,
    headline_panel,
    month_end,
    months_between,
    select,
)
from model.training_data import ACTUAL_PROVENANCE, TrainingData

log = logging.getLogger(__name__)

FRED_PAGE_LIMIT = 100_000


class FredError(RuntimeError):
    """Raised when the FRED API refuses or fails a request."""


class StaleDataError(LookupError):
    """Raised when the latest observation is too old for the requested date."""


class CacheMissingError(FileNotFoundError):
    """Raised when the local cache has not been populated yet."""


def _parse_observations(series_id: str, observations: Iterable[Mapping]) -> pd.Series:
    obs = list(observations)
    if not obs:
        return pd.Series(dtype="float64", name=series_id)
    idx = pd.to_datetime([o["date"] for o in obs])
    vals = pd.to_numeric([o["value"] for o in obs], errors="coerce")  # FRED uses "." for missing
    s = pd.Series(vals, index=idx, name=series_id).dropna().sort_index()
    s.index.name = "date"
    return s[~s.index.duplicated(keep="last")]


class FredFetcher:
    """Fetches, transforms and caches the indicator series."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str | Path | None = None,
        manual_dir: str | Path | None = None,
        indicators: Iterable[IndicatorSpec | str] = INDICATORS,
        session: requests.Session | None = None,
        observation_start: str | None = None,
        base_url: str | None = None,
        stale_limit_months: int | None = None,
        timeout: float = config.REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self.api_key = api_key if api_key is not None else config.FRED_API_KEY
        self.cache_dir = Path(cache_dir) if cache_dir is not None else config.FRED_CACHE_DIR
        self.manual_dir = Path(manual_dir) if manual_dir is not None else config.MANUAL_DIR
        self.indicators = tuple(get_spec(i) if isinstance(i, str) else i for i in indicators)
        self.session = session or requests.Session()
        self.observation_start = observation_start or config.OBSERVATION_START
        self.base_url = base_url or config.FRED_BASE_URL
        self.stale_limit_months = (
            stale_limit_months if stale_limit_months is not None else config.STALE_LIMIT_MONTHS
        )
        self.timeout = timeout
        self._panel: pd.DataFrame | None = None

    # ------------------------------------------------------------------ paths
    @property
    def raw_dir(self) -> Path:
        return self.cache_dir / "raw"

    @property
    def panel_path(self) -> Path:
        return self.cache_dir / "panel.parquet"

    @property
    def meta_path(self) -> Path:
        return self.cache_dir / "meta.json"

    def raw_path(self, series_id: str) -> Path:
        return self.raw_dir / f"{series_id}.parquet"

    @property
    def series_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(sid for spec in self.indicators for sid in spec.series_ids))

    @property
    def manual_only_ids(self) -> frozenset[str]:
        return frozenset(sid for spec in self.indicators if spec.manual_only for sid in spec.series_ids)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(spec.key for spec in self.indicators)

    # --------------------------------------------------------------- network
    def fetch_series(self, series_id: str) -> pd.Series:
        """Pull one raw series from FRED (all pages), returned as a float Series."""
        if not self.api_key:
            raise FredError(
                "FRED_API_KEY is not set. Get a free key at "
                "https://fred.stlouisfed.org/docs/api/api_key.html and export it."
            )
        observations: list[Mapping] = []
        offset = 0
        while True:
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "observation_start": self.observation_start,
                "limit": FRED_PAGE_LIMIT,
                "offset": offset,
            }
            try:
                resp = self.session.get(self.base_url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:  # pragma: no cover - network
                raise FredError(f"{series_id}: request failed: {exc}") from exc
            if resp.status_code != 200:
                detail = ""
                try:
                    detail = resp.json().get("error_message", "")
                except Exception:  # noqa: BLE001 - best-effort error detail
                    detail = getattr(resp, "text", "")
                raise FredError(f"{series_id}: FRED returned HTTP {resp.status_code}: {detail}")
            body = resp.json()
            page = body.get("observations", [])
            observations.extend(page)
            count = int(body.get("count", len(observations)))
            offset += len(page)
            if not page or offset >= count:
                break
        return _parse_observations(series_id, observations)

    # ---------------------------------------------------------------- manual
    def manual_series(self, series_id: str) -> pd.Series | None:
        path = self.manual_dir / f"{series_id}.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        if "date" not in cols or "value" not in cols:
            raise ValueError(f"{path}: expected columns 'date' and 'value'")
        s = pd.Series(
            pd.to_numeric(df[cols["value"]], errors="coerce").to_numpy(),
            index=pd.to_datetime(df[cols["date"]]),
            name=series_id,
        ).dropna().sort_index()
        s.index.name = "date"
        return s[~s.index.duplicated(keep="last")]

    def _pull_with_supplement(self, series_id: str) -> pd.Series:
        manual = self.manual_series(series_id)
        if series_id in self.manual_only_ids:
            if manual is None or manual.empty:
                raise FredError(
                    f"{series_id} is not published on FRED; supply {self.manual_dir / (series_id + '.csv')} "
                    "(columns: date,value) or remove the indicator from MACRO_REGIME_INDICATORS"
                )
            return manual
        try:
            fred = self.fetch_series(series_id)
        except FredError:
            if manual is None or manual.empty:
                raise
            log.warning("%s: FRED pull failed, using manual CSV only", series_id)
            return manual
        if manual is None or manual.empty:
            return fred
        merged = pd.concat([fred, manual]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]  # manual wins on overlap
        merged.name = series_id
        return merged

    # ----------------------------------------------------------------- cache
    def refresh(self) -> pd.DataFrame:
        """Re-pull every raw series, rebuild the headline panel, rewrite the cache."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        raw: dict[str, pd.Series] = {}
        for sid in self.series_ids:
            s = self._pull_with_supplement(sid)
            if s.empty:
                raise FredError(f"{sid}: no observations returned")
            raw[sid] = s
            s.to_frame("value").to_parquet(self.raw_path(sid))
        panel = headline_panel(raw, self.indicators)
        panel.to_parquet(self.panel_path)
        meta = {
            "refreshed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "observation_start": self.observation_start,
            "series": {
                sid: {"first": str(s.index[0].date()), "last": str(s.index[-1].date()), "n": int(len(s))}
                for sid, s in raw.items()
            },
            "indicators": {
                key: {
                    "first": str(panel[key].first_valid_index().date()),
                    "last": str(panel[key].last_valid_index().date()),
                }
                for key in panel.columns
            },
        }
        self.meta_path.write_text(json.dumps(meta, indent=2))
        self._panel = panel
        log.info("FRED cache refreshed: %s", {k: v["last"] for k, v in meta["indicators"].items()})
        return panel.copy()

    def load_raw(self, series_id: str) -> pd.Series:
        path = self.raw_path(series_id)
        if not path.exists():
            raise CacheMissingError(f"{path} missing; run FredFetcher.refresh() first")
        s = pd.read_parquet(path)["value"]
        s.name = series_id
        return s

    def meta(self) -> dict:
        if not self.meta_path.exists():
            return {}
        return json.loads(self.meta_path.read_text())

    def has_cache(self) -> bool:
        return self.panel_path.exists()

    def panel(self) -> pd.DataFrame:
        """Monthly headline panel (columns = indicator keys, month-end index)."""
        if self._panel is None:
            if not self.panel_path.exists():
                raise CacheMissingError(
                    f"{self.panel_path} missing; run FredFetcher.refresh() first"
                )
            self._panel = pd.read_parquet(self.panel_path)
        return self._panel.copy()

    def features(self, panel: pd.DataFrame | None = None) -> pd.DataFrame:
        """Feature matrix (columns = feature names) from the cached or given panel."""
        return features_from_panel(self.panel() if panel is None else panel, self.indicators)

    # ------------------------------------------------------------- accessors
    def latest_date(self, indicator: str | None = None) -> pd.Timestamp:
        """Latest month with a value for ``indicator`` (or across all indicators)."""
        panel = self.panel()
        if indicator is None:
            return pd.Timestamp(panel.dropna(how="all").index.max())
        get_spec(indicator)
        last = panel[indicator].last_valid_index()
        if last is None:
            raise StaleDataError(f"{indicator}: no observations in cache")
        return pd.Timestamp(last)

    def get_actual(self, indicator: str, date) -> float:
        """Headline value of ``indicator`` as of ``date`` (month granularity).

        Returns the last observation on or before the month of ``date``; raises
        :class:`StaleDataError` when that observation is older than the stale limit.
        """
        get_spec(indicator)
        target = month_end(date)
        s = self.panel()[indicator].dropna()
        s = s.loc[:target]
        if s.empty:
            raise StaleDataError(f"{indicator}: no observation on or before {target.date()}")
        last = pd.Timestamp(s.index[-1])
        if months_between(last, target) > self.stale_limit_months:
            raise StaleDataError(
                f"{indicator}: latest observation {last.date()} is more than "
                f"{self.stale_limit_months} months before {target.date()}"
            )
        return float(s.iloc[-1])

    def latest_values(self) -> dict[str, float]:
        return {spec.key: self.get_actual(spec.key, self.latest_date(spec.key)) for spec in self.indicators}

    # -------------------------------------------------------------- training
    def training_data(
        self,
        start=None,
        end=None,
        window_years: int | None = None,
    ) -> TrainingData:
        """Build the HMM training set from the **actual** cache only.

        This is the only place in the codebase that constructs
        :class:`TrainingData` with the ``fred_actual`` provenance. It never
        consults the override store, by construction.
        """
        feats = self.features().dropna()
        if feats.empty:
            raise CacheMissingError("no complete feature rows; refresh the cache first")
        start_ts = pd.Timestamp(start) if start is not None else pd.Timestamp(config.TRAINING_START)
        if window_years:
            start_ts = max(start_ts, feats.index[-1] - pd.DateOffset(years=window_years))
        feats = feats.loc[start_ts:]
        if end is not None:
            feats = feats.loc[: month_end(end)]
        signs = {spec.feature_name: spec.bullish_sign for spec in self.indicators}
        return TrainingData(
            frame=feats,
            provenance=ACTUAL_PROVENANCE,
            source=f"FRED cache {self.cache_dir}",
            feature_signs=signs,
        )


_default_fetcher: FredFetcher | None = None


def default_fetcher() -> FredFetcher:
    """Process-wide fetcher using the paths in :mod:`config`."""
    global _default_fetcher
    if _default_fetcher is None:
        _default_fetcher = FredFetcher()
    return _default_fetcher


def get_actual(indicator: str, date) -> float:
    return default_fetcher().get_actual(indicator, date)


def refresh() -> pd.DataFrame:
    return default_fetcher().refresh()
