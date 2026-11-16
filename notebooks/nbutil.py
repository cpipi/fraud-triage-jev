"""Shared helpers for the analysis notebooks: data loading and a consistent chart style."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Validated default categorical palette (fixed order; colour follows the entity).
MODEL_COLORS: Dict[str, str] = {
    "jev": "#2a78d6",
    "gpt-4o-mini": "#eb6834",
    "gpt-4o-mini-reasoning": "#e87ba4",
    "reference_model": "#1baf7a",
    "rules": "#eda100",
}
MODEL_TITLES: Dict[str, str] = {
    "jev": "Jev",
    "gpt-4o-mini": "GPT-4o-mini (label only)",
    "gpt-4o-mini-reasoning": "GPT-4o-mini (reasoning)",
    "reference_model": "Supervised ref. (ceiling)",
    "rules": "Analyst rules (floor)",
}
TYPOLOGY_COLORS: Dict[str, str] = {
    "NORMAL": "#2a78d6",
    "MONEY_MULE": "#eb6834",
    "ACCOUNT_TAKEOVER": "#1baf7a",
    "SYNTHETIC_ID": "#eda100",
    "STRUCTURING_AML": "#e87ba4",
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "font.size": 9.5,
    })


def data_dir() -> Path:
    """PaySim-seeded data when present, otherwise the stand-in (clearly flagged)."""
    real = ROOT / "data" / "alerts_test.jsonl"
    if real.exists():
        return ROOT / "data"
    print("NOTE: PaySim-seeded data not found; using synthetic stand-in data (not publishable).")
    return ROOT / "data" / "stand_in"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def latest_run(prefer_live: bool = True, name: str = "main") -> Optional[Path]:
    """The named live run (default `main`) if it exists, else the most recent run."""
    named = ROOT / "results" / "live" / name / "report.json"
    if prefer_live and named.exists():
        return named.parent
    for kind in (["live", "dry_run"] if prefer_live else ["dry_run"]):
        runs = sorted((ROOT / "results" / kind).glob("*/report.json"), key=lambda p: p.stat().st_mtime)
        runs = [r for r in runs if r.parent.name not in {"smoke"}]
        if runs:
            return runs[-1].parent
    return None
