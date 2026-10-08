"""Strict point-in-time NIFTY option contract resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.historical_option_contract_resolver import (
    build_snapshot_catalog,
)


DEFAULT_ARCHIVE = Path("backup/reconstruction/data/security_master_archive")
DEFAULT_INDEX = Path("data/aligned/canonical_index_option_aligned.parquet")
DEFAULT_VALIDATED = Path(
    "data/normalized/historical_options/validated_one_day/manifest.json")


@dataclass(frozen=True)
class ContractResolution:
    status: str
    date: str
    timestamp: str | None
    underlying: str
    expiry_policy: str
    strike_policy: str
    option_type: str
    expiry: str | None = None
    strike: float | None = None
    security_id: str | None = None
    source_snapshot: str | None = None
    contract_segment_id: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class HistoricalContractResolver:
    """Resolve contracts only from timestamped masters or validated evidence."""

    def __init__(self, *, archive: str | Path = DEFAULT_ARCHIVE,
                 index_path: str | Path = DEFAULT_INDEX,
                 validated_manifest: str | Path = DEFAULT_VALIDATED):
        paths = sorted(Path(archive).glob("dhan_master_*.csv"))
        self.catalog = build_snapshot_catalog(paths)
        self.index = self._load_index(Path(index_path))
        self._session_last = {} if self.index.empty else {
            day: group.iloc[-1]
            for day, group in self.index.groupby(
                self.index.timestamp.dt.strftime("%Y-%m-%d"), sort=False)
        }
        self.validated = self._load_validated(Path(validated_manifest))

    @staticmethod
    def _load_index(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame({
                "timestamp": pd.Series(dtype="datetime64[ns, Asia/Kolkata]"),
                "index_close": pd.Series(dtype="float64"),
            })
        frame = pd.read_parquet(path, columns=["timestamp", "index_close"])
        frame["timestamp"] = pd.to_datetime(
            frame.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
        return frame.sort_values("timestamp", kind="stable")

    @staticmethod
    def _load_validated(path: Path) -> dict:
        if not path.exists():
            return {}
        body = json.loads(path.read_text(encoding="utf-8"))
        if not body.get("complete_session"):
            return {}
        return body

    def resolve_contract(self, *, underlying: str = "NIFTY", date,
                         expiry_policy: str = "WEEK/CURRENT_NEAR",
                         strike_policy: str = "ATM", option_type: str,
                         timestamp=None) -> ContractResolution:
        day = pd.Timestamp(date).date().isoformat()
        side = option_type.upper().replace("CALL", "CE").replace("PUT", "PE")
        base = dict(date=day, timestamp=None, underlying=underlying.upper(),
                    expiry_policy=expiry_policy, strike_policy=strike_policy,
                    option_type=side)
        if underlying.upper() != "NIFTY":
            return ContractResolution("UNRESOLVED", **base, reason="UNSUPPORTED_UNDERLYING")
        if expiry_policy != "WEEK/CURRENT_NEAR":
            return ContractResolution("UNRESOLVED", **base, reason="UNSUPPORTED_EXPIRY_POLICY")
        if strike_policy != "ATM":
            return ContractResolution("UNRESOLVED", **base, reason="UNSUPPORTED_STRIKE_POLICY")
        if side not in {"CE", "PE"}:
            return ContractResolution("UNRESOLVED", **base, reason="UNSUPPORTED_OPTION_TYPE")

        verified = self._validated_contract(day, side)
        if verified is not None:
            return verified
        point, spot = self._spot(day, timestamp)
        if point is None:
            return ContractResolution("UNRESOLVED", **base, reason="NIFTY_SPOT_UNAVAILABLE")
        base["timestamp"] = point.isoformat()
        strike = float(round(float(spot) / 50.0) * 50.0)
        result, failure = self._catalog_contract(point, strike, side)
        if result is None:
            return ContractResolution(
                "UNRESOLVED", **base, strike=strike,
                reason=failure)
        identity = self._segment_id(
            day, result.expiry.strftime("%Y-%m-%d"), strike, side,
            str(result.security_id))
        return ContractResolution(
            "RESOLVED", **base, expiry=result.expiry.strftime("%Y-%m-%d"), strike=strike,
            security_id=str(result.security_id),
            source_snapshot=str(result.master_source),
            contract_segment_id=identity,
            reason="TIMESTAMPED_MASTER_NEAREST_WEEKLY_EXPIRY")

    def _catalog_contract(self, point: pd.Timestamp, strike: float, side: str):
        eligible = self.catalog.loc[self.catalog.snapshot_timestamp.le(point)]
        if eligible.empty:
            return None, "NO_ELIGIBLE_TIMESTAMPED_MASTER"
        captured = eligible.snapshot_timestamp.max()
        master = eligible.loc[
            eligible.snapshot_timestamp.eq(captured) & eligible.expiry_flag.eq("W")]
        future = master.loc[master.expiry.dt.date >= captured.date()]
        if future.empty or point.date() > future.expiry.min().date():
            return None, "NO_ELIGIBLE_TIMESTAMPED_MASTER"
        expiry = future.expiry.min()
        match = future.loc[
            future.expiry.eq(expiry) & future.strike.eq(strike)
            & future.option_type.eq(side)]
        if match.empty:
            return None, "CONTRACT_NOT_IN_ELIGIBLE_MASTER"
        if match.security_id.nunique() != 1:
            return None, "AMBIGUOUS_SECURITY_ID"
        return match.iloc[0], None

    def _spot(self, day: str, timestamp) -> tuple[pd.Timestamp | None, float | None]:
        selected = self._session_last.get(day)
        if selected is None:
            return None, None
        if timestamp is None:
            pass
        else:
            session = self.index.loc[
                self.index.timestamp.dt.strftime("%Y-%m-%d").eq(day)]
            point = pd.Timestamp(timestamp)
            point = (point.tz_localize("Asia/Kolkata") if point.tzinfo is None
                     else point.tz_convert("Asia/Kolkata"))
            eligible = session.loc[session.timestamp.le(point)]
            if eligible.empty:
                return None, None
            selected = eligible.iloc[-1]
        return pd.Timestamp(selected.timestamp), float(selected.index_close)

    def _validated_contract(self, day: str, side: str) -> ContractResolution | None:
        if self.validated.get("session_date") != day:
            return None
        item = self.validated.get("contracts", {}).get(side)
        if not item:
            return None
        security_id = str(item["security_id"])
        strike = float(item["strike"])
        expiry = str(item["expiry"])
        timestamp = item.get("last_timestamp")
        return ContractResolution(
            "RESOLVED", day, timestamp, "NIFTY", "WEEK/CURRENT_NEAR", "ATM",
            side, expiry, strike, security_id, str(DEFAULT_VALIDATED),
            self._segment_id(day, expiry, strike, side, security_id),
            "VALIDATED_EXACT_CONTRACT_SESSION")

    @staticmethod
    def _segment_id(day: str, expiry: str, strike: float,
                    side: str, security_id: str) -> str:
        identity = f"NIFTY|{day}|{expiry}|{strike:g}|{side}|{security_id}"
        return hashlib.sha256(identity.encode()).hexdigest()[:20]
