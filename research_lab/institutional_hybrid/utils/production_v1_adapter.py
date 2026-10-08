"""Read-only adapter and suitability audit for Production Dataset V1.

The adapter deliberately fails closed: it never fills absent market history
from model estimates, derived outcomes, or a different production artifact.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd


LAB_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = LAB_ROOT.parents[1]
PRODUCTION_V1 = WORKSPACE_ROOT / "data" / "prediction" / "prediction_dataset_v1.parquet"

REQUIRED_CONTINUOUS_FIELDS = {
    "timestamp",
    "index_open", "index_high", "index_low", "index_close",
    "futures_volume",
    "option_open", "option_high", "option_low", "option_close", "option_volume",
}
REQUIRED_OPTION_LIQUIDITY_FIELDS = {
    "option_bid", "option_ask", "option_open_interest",
    "option_delta", "option_gamma", "option_theta", "option_iv",
}
FORBIDDEN_OUTCOME_PROXIES = {
    "premium_expansion_probability", "expected_premium_expansion",
    "expected_drawdown", "expected_holding_minutes", "expected_terminal_return",
    "terminal_return_standard_error", "terminal_return_95_lower",
    "expansion_drawdown_ratio", "recommendation",
}


class DatasetNotSuitableError(RuntimeError):
    """Raised when real historical trades cannot be reconstructed."""


@dataclass(frozen=True)
class DatasetAudit:
    source: str
    sha256: str
    rows: int
    columns: int
    first_timestamp: str
    last_timestamp: str
    duplicate_timestamps: int
    median_interval_minutes: float
    intervals_over_one_minute_pct: float
    missing_continuous_fields: list[str]
    missing_option_liquidity_fields: list[str]
    forbidden_proxy_fields_present: list[str]
    suitable: bool
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ProductionV1ReadOnlyAdapter:
    """Load V1 without any write, mutation, fallback join, or proxy outcome."""

    def __init__(self, source: Path = PRODUCTION_V1):
        self.source = source.resolve()
        expected = PRODUCTION_V1.resolve()
        if self.source != expected:
            raise ValueError(f"Adapter is pinned to Production Dataset V1: {expected}")

    def read(self) -> pd.DataFrame:
        if not self.source.is_file():
            raise FileNotFoundError(f"Production Dataset V1 not found: {self.source}")
        before = self.source.stat()
        frame = pd.read_parquet(self.source)
        after = self.source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("Production Dataset V1 changed during the read; aborting.")
        return frame

    def audit(self, frame: pd.DataFrame | None = None) -> DatasetAudit:
        data = self.read() if frame is None else frame
        if "timestamp" not in data:
            timestamps = pd.Series(dtype="datetime64[ns]")
        else:
            timestamps = pd.to_datetime(data["timestamp"], errors="coerce").sort_values()
        intervals = timestamps.diff().dt.total_seconds().div(60).dropna()
        missing_market = sorted(REQUIRED_CONTINUOUS_FIELDS.difference(data.columns))
        missing_liquidity = sorted(REQUIRED_OPTION_LIQUIDITY_FIELDS.difference(data.columns))
        reasons: list[str] = []
        if missing_market:
            reasons.append("Missing continuous futures/index and realized ATM premium fields.")
        if missing_liquidity:
            reasons.append("Missing historical option liquidity and Greek fields.")
        if not intervals.empty and float((intervals > 1).mean()) > 0.01:
            reasons.append("Rows are sparse setup observations, not a continuous one-minute path.")
        proxies = sorted(FORBIDDEN_OUTCOME_PROXIES.intersection(data.columns))
        if proxies:
            reasons.append("Model-derived proxy fields are present and prohibited as realized outcomes.")
        return DatasetAudit(
            source=str(self.source), sha256=_file_sha256(self.source),
            rows=len(data), columns=len(data.columns),
            first_timestamp=str(timestamps.min()), last_timestamp=str(timestamps.max()),
            duplicate_timestamps=int(timestamps.duplicated().sum()),
            median_interval_minutes=float(intervals.median()) if not intervals.empty else float("nan"),
            intervals_over_one_minute_pct=float((intervals > 1).mean()) if not intervals.empty else 1.0,
            missing_continuous_fields=missing_market,
            missing_option_liquidity_fields=missing_liquidity,
            forbidden_proxy_fields_present=proxies,
            suitable=not reasons, reasons=reasons,
        )

    def load_for_phase1(self) -> pd.DataFrame:
        frame = self.read()
        audit = self.audit(frame)
        if not audit.suitable:
            details = "; ".join(audit.reasons)
            raise DatasetNotSuitableError(
                f"Production Dataset V1 cannot support Phase-1 realized-trade research: {details}"
            )
        return frame
