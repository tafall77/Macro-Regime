"""Central configuration for the Macro Regime Engine.

Everything here can be overridden through environment variables so the same code
runs from the dashboard process, the monthly refit job and the test-suite without
edits. Paths are resolved once at import time.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from a .env file into os.environ (existing variables win)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ROOT / ".env")  # git-ignored; the simplest place for FRED_API_KEY


def _refuse_committed_secret(example: Path) -> None:
    """Stop early if a real-looking key sits in the committed example file."""
    import re

    if example.exists() and re.search(r"^\s*FRED_API_KEY\s*=\s*[0-9a-f]{32}\s*$", example.read_text(), re.M):
        raise RuntimeError(
            f"{example.name} contains what looks like a real FRED API key. That file is committed to git: "
            "move the key to a file named .env (git-ignored), restore the placeholder in "
            f"{example.name}, and regenerate the key at https://fred.stlouisfed.org/docs/api/api_key.html."
        )


_refuse_committed_secret(ROOT / ".env.example")

# --- Local state -------------------------------------------------------------
CACHE_DIR = Path(os.environ.get("MACRO_REGIME_CACHE_DIR", ROOT / "cache")).expanduser()
FRED_CACHE_DIR = CACHE_DIR / "fred"          # raw series + headline panel (Parquet)
MANUAL_DIR = CACHE_DIR / "manual"            # optional hand-maintained CSV supplements
MODEL_DIR = CACHE_DIR / "model"
MODEL_PATH = MODEL_DIR / "hmm_engine.pkl"
LABELS_PATH = MODEL_DIR / "labels.json"
OVERRIDES_PATH = CACHE_DIR / "overrides.json"  # session file for scenario overrides

# --- FRED --------------------------------------------------------------------
FRED_API_KEY = os.environ.get("FRED_API_KEY")
FRED_BASE_URL = os.environ.get(
    "FRED_BASE_URL", "https://api.stlouisfed.org/fred/series/observations"
)
OBSERVATION_START = os.environ.get("MACRO_REGIME_OBSERVATION_START", "1985-01-01")
# ISM stopped licensing its PMI to FRED in 2016; NAPM is the historical series id.
# Point this at a proxy series, or drop a CSV in MANUAL_DIR (see README).
PMI_SERIES_ID = os.environ.get("MACRO_REGIME_PMI_SERIES", "NAPM")
REQUEST_TIMEOUT_SECONDS = 30

# --- Indicator set -----------------------------------------------------------
# Comma-separated keys from data.transforms.CATALOGUE. The default is the core set the
# model ships with; run `MacroRegime.evaluate_indicators()` on real data before widening it.
DEFAULT_INDICATORS = "yield_curve,core_pce,fed_funds,ism_pmi,credit_spread,unemployment,jobless_claims"
ACTIVE_INDICATORS = tuple(
    k.strip() for k in os.environ.get("MACRO_REGIME_INDICATORS", DEFAULT_INDICATORS).split(",") if k.strip()
)

# --- Feature construction ----------------------------------------------------
FED_FUNDS_ROC_MONTHS = int(os.environ.get("MACRO_REGIME_FF_ROC_MONTHS", "3"))
# How many months a lagging indicator may be carried forward before it is "stale".
STALE_LIMIT_MONTHS = int(os.environ.get("MACRO_REGIME_STALE_LIMIT_MONTHS", "3"))

# --- Model -------------------------------------------------------------------
N_STATES = int(os.environ.get("MACRO_REGIME_N_STATES", "3"))
TRAINING_START = os.environ.get("MACRO_REGIME_TRAINING_START", "1990-01-01")
_window = os.environ.get("MACRO_REGIME_TRAINING_WINDOW_YEARS")
TRAINING_WINDOW_YEARS: int | None = int(_window) if _window else None  # None = expanding window
RANDOM_SEED = 7

# --- Scheduler ---------------------------------------------------------------
REFIT_DAY_OF_MONTH = int(os.environ.get("MACRO_REGIME_REFIT_DAY", "2"))
REFIT_HOUR_UTC = int(os.environ.get("MACRO_REGIME_REFIT_HOUR_UTC", "6"))

# --- Dashboard ---------------------------------------------------------------
DASH_HOST = os.environ.get("MACRO_REGIME_HOST", "127.0.0.1")
DASH_PORT = int(os.environ.get("MACRO_REGIME_PORT", "8050"))
