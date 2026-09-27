"""Human-in-the-loop labelling of hidden states.

An HMM only knows the data structure; it has no idea which state is "bullish".
After each refit, look at the per-state feature means (``show``), then record
the mapping (``set`` / ``accept``). The mapping is stored against the engine's
``fit_id`` so a refit that reshuffles states can never silently inherit stale
labels: until you confirm again, the dashboard shows the heuristic suggestion
marked *provisional*.

CLI::

    python -m model.labeler show
    python -m model.labeler accept                       # accept the suggestion
    python -m model.labeler set 0=bearish 1=transition 2=bullish
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

import config
from model.hmm_engine import HMMEngine

BEARISH, TRANSITION, BULLISH = "bearish", "transition", "bullish"
VALID_LABELS: tuple[str, ...] = (BEARISH, TRANSITION, BULLISH)


@dataclass(frozen=True)
class LabelSet:
    fit_id: str
    labels: dict[int, str]
    provisional: bool  # True when these are heuristic suggestions, not confirmed by a human

    def name(self, state: int) -> str:
        label = self.labels.get(int(state))
        if label is None:
            return f"State {state}"
        return label.capitalize()

    def display(self, state: int, prob: float) -> str:
        """'Bullish: 72' instead of 'State 2: 0.72'."""
        return f"{self.name(state)}: {round(prob * 100):.0f}"


def _normalise_label(label: str) -> str:
    lab = str(label).strip().lower()
    if lab not in VALID_LABELS:
        raise ValueError(f"label must be one of {VALID_LABELS}, got {label!r}")
    return lab


class StateLabeler:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else config.LABELS_PATH
        self._fit_id: str | None = None
        self._labels: dict[int, str] = {}
        self._confirmed_at: str | None = None
        self._load()

    # ------------------------------------------------------------ persistence
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return
        self._fit_id = data.get("fit_id")
        self._confirmed_at = data.get("confirmed_at")
        self._labels = {}
        for k, v in (data.get("labels") or {}).items():
            try:
                self._labels[int(k)] = _normalise_label(v)
            except ValueError:
                continue

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "fit_id": self._fit_id,
                    "confirmed_at": self._confirmed_at,
                    "labels": {str(k): v for k, v in sorted(self._labels.items())},
                },
                indent=2,
            )
        )

    # --------------------------------------------------------------- setting
    def set_labels(self, fit_id: str, mapping: Mapping[int, str]) -> dict[int, str]:
        labels = {int(k): _normalise_label(v) for k, v in mapping.items()}
        if fit_id != self._fit_id:
            self._labels = {}
        self._fit_id = fit_id
        self._labels.update(labels)
        self._confirmed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._save()
        return dict(self._labels)

    def set_label(self, fit_id: str, state: int, label: str) -> dict[int, str]:
        return self.set_labels(fit_id, {state: label})

    def clear(self) -> None:
        self._fit_id, self._labels, self._confirmed_at = None, {}, None
        self._save()

    # --------------------------------------------------------------- reading
    def labels_for(self, fit_id: str) -> dict[int, str] | None:
        """Confirmed labels for this fit, or None if the stored mapping is for another fit."""
        if fit_id is None or fit_id != self._fit_id or not self._labels:
            return None
        return dict(self._labels)

    @property
    def confirmed_fit_id(self) -> str | None:
        return self._fit_id

    @staticmethod
    def suggest(engine: HMMEngine) -> dict[int, str]:
        """Heuristic labels from the growth-score ordering: lowest = bearish, highest = bullish."""
        scores = np.asarray(engine.state_scores_, dtype="float64")
        order = list(np.argsort(scores, kind="stable"))
        k = len(order)
        out: dict[int, str] = {}
        for rank, state in enumerate(order):
            if rank == 0:
                out[int(state)] = BEARISH
            elif rank == k - 1:
                out[int(state)] = BULLISH
            else:
                out[int(state)] = TRANSITION
        return out

    def resolve(self, engine: HMMEngine) -> LabelSet:
        """Confirmed labels if they match ``engine.fit_id``; otherwise a provisional suggestion."""
        confirmed = self.labels_for(engine.fit_id)
        if confirmed is not None and len(confirmed) == engine.n_states:
            return LabelSet(engine.fit_id, confirmed, provisional=False)
        return LabelSet(engine.fit_id, self.suggest(engine), provisional=True)

    def accept_suggestion(self, engine: HMMEngine) -> dict[int, str]:
        return self.set_labels(engine.fit_id, self.suggest(engine))

    def describe(self, engine: HMMEngine) -> pd.DataFrame:
        """Engine's state table plus the suggested and confirmed labels."""
        df = engine.describe()
        suggestion = self.suggest(engine)
        confirmed = self.labels_for(engine.fit_id) or {}
        df.insert(0, "suggested", [suggestion[k] for k in df.index])
        df.insert(1, "confirmed", [confirmed.get(k, "") for k in df.index])
        return df


# -------------------------------------------------------------------- CLI
def _parse_assignments(items: list[str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"expected STATE=LABEL, got {item!r}")
        state, label = item.split("=", 1)
        out[int(state)] = label
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Label HMM hidden states (human in the loop).")
    parser.add_argument("--model", default=str(config.MODEL_PATH))
    parser.add_argument("--labels", default=str(config.LABELS_PATH))
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("show", help="print state means, suggestion and confirmed labels")
    sub.add_parser("accept", help="confirm the heuristic suggestion for the current fit")
    p_set = sub.add_parser("set", help="confirm labels explicitly, e.g. 0=bearish 1=transition 2=bullish")
    p_set.add_argument("assignments", nargs="+")
    args = parser.parse_args(argv)

    engine = HMMEngine.load(args.model)
    labeler = StateLabeler(args.labels)
    cmd = args.cmd or "show"
    if cmd == "accept":
        labeler.accept_suggestion(engine)
    elif cmd == "set":
        labeler.set_labels(engine.fit_id, _parse_assignments(args.assignments))

    with pd.option_context("display.width", 160, "display.max_columns", 20, "display.precision", 3):
        print(f"model: {engine}")
        print(labeler.describe(engine).to_string())
    ls = labeler.resolve(engine)
    print("labels:", {k: ls.name(k) for k in sorted(ls.labels)}, "(provisional)" if ls.provisional else "(confirmed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
