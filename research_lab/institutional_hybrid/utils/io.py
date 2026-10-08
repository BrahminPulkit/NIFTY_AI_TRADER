"""Explicit read-only loaders; all outputs must remain inside LAB_ROOT."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

LAB_ROOT = Path(__file__).resolve().parents[1]


def assert_lab_output(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != LAB_ROOT and LAB_ROOT not in resolved.parents:
        raise ValueError(f"Refusing write outside research lab: {resolved}")
    return resolved


def read_market_csv(path: Path) -> pd.DataFrame:
    """Read any user-supplied market snapshot without mutating its source."""
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    return frame.set_index("timestamp").sort_index()
