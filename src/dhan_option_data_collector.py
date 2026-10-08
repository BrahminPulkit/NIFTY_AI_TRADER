"""Contract-aware persistence for Dhan data already held in the shared cache."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SCHEMA_VERSION = 1
DATA_COLUMNS = [
    "timestamp", "underlying", "market", "option_type", "security_id",
    "exchange_segment", "instrument", "display_name", "strike", "expiry",
    "open", "high", "low", "close", "volume", "open_interest", "ltp",
    "bid", "ask", "spread", "spot", "collected_at", "source",
    "schema_version",
]


class OptionMarketDataCollector:
    """Persist one latest completed candle per resolved CE/PE contract.

    The collector deliberately has no broker client. It can only consume a
    defensive snapshot produced by ``DhanConnectionManager``.
    """

    def __init__(self, root: str | Path = "data/collected/dhan_options"):
        self.root = Path(root)

    def collect_snapshot(self, snapshot: dict, *, collected_at=None) -> dict:
        now = self._timestamp(collected_at)
        rows = []
        candles = snapshot.get("candles", {})
        options = snapshot.get("options", {})
        chain = {
            str(row.get("security_id")): row
            for row in snapshot.get("option_chain", [])
            if row.get("security_id") is not None
        }
        for market in ("NIFTY_ATM_CE", "NIFTY_ATM_PE"):
            metadata = options.get(market)
            frame = candles.get(market)
            if not metadata or not isinstance(frame, pd.DataFrame) or frame.empty:
                continue
            candle = frame.sort_values("timestamp", kind="stable").iloc[-1]
            quote = chain.get(str(metadata.get("security_id")), {})
            rows.append(self._option_row(
                market, metadata, candle, quote, snapshot, now))
        if not rows:
            return {"status": "WAITING", "rows_written": 0, "last_write": None}

        incoming = pd.DataFrame(rows, columns=DATA_COLUMNS)
        # Partition by exchange candle date, not collection date. A weekend
        # heartbeat must not duplicate Friday's final candle under Saturday.
        latest_candle = pd.to_datetime(incoming["timestamp"], utc=True).max()
        day = latest_candle.tz_convert("Asia/Kolkata").strftime("%Y-%m-%d")
        directory = self.root / day
        path = directory / "nifty_atm_options.parquet"
        directory.mkdir(parents=True, exist_ok=True)
        existing = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=DATA_COLUMNS)
        combined = pd.concat([existing, incoming], ignore_index=True)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
        combined = combined.drop_duplicates(
            ["timestamp", "security_id"], keep="last"
        ).sort_values(["timestamp", "option_type"], kind="stable")
        temporary = path.with_suffix(".tmp.parquet")
        combined.to_parquet(temporary, index=False)
        temporary.replace(path)
        self._write_manifest(directory, path, combined, now)
        return {
            "status": "READY",
            "rows_written": int(len(incoming)),
            "total_rows": int(len(combined)),
            "last_write": now.isoformat(),
            "file": str(path),
        }

    @staticmethod
    def _timestamp(value=None) -> pd.Timestamp:
        current = pd.Timestamp.now(tz="Asia/Kolkata") if value is None else pd.Timestamp(value)
        return (current.tz_localize("Asia/Kolkata") if current.tzinfo is None
                else current.tz_convert("Asia/Kolkata"))

    @staticmethod
    def _option_row(market, metadata, candle, quote, snapshot, now) -> dict:
        option_type = str(metadata.get("option_type", "")).upper()
        if option_type not in {"CE", "PE"}:
            raise ValueError(f"Unsupported option type for collection: {option_type!r}")
        candle_timestamp = pd.Timestamp(candle["timestamp"])
        if candle_timestamp.tzinfo is None:
            candle_timestamp = candle_timestamp.tz_localize("Asia/Kolkata")
        return {
            "timestamp": candle_timestamp,
            "underlying": "NIFTY",
            "market": market,
            "option_type": option_type,
            "security_id": str(metadata["security_id"]),
            "exchange_segment": metadata.get("exchange_segment"),
            "instrument": metadata.get("instrument"),
            "display_name": metadata.get("display_name"),
            "strike": metadata.get("strike"),
            "expiry": metadata.get("expiry"),
            "open": candle.get("open"), "high": candle.get("high"),
            "low": candle.get("low"), "close": candle.get("close"),
            "volume": candle.get("volume"),
            "open_interest": candle.get("open_interest"),
            "ltp": quote.get("ltp"), "bid": quote.get("bid"),
            "ask": quote.get("ask"), "spread": quote.get("spread"),
            "spot": snapshot.get("quotes", {}).get("NIFTY"),
            "collected_at": now,
            "source": "DHAN_SHARED_CACHE",
            "schema_version": SCHEMA_VERSION,
        }

    @staticmethod
    def _write_manifest(directory: Path, data_path: Path,
                        frame: pd.DataFrame, now: pd.Timestamp) -> None:
        counts = frame["option_type"].value_counts().to_dict()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "file": data_path.name,
            "rows": int(len(frame)),
            "ce_rows": int(counts.get("CE", 0)),
            "pe_rows": int(counts.get("PE", 0)),
            "first_timestamp": pd.Timestamp(frame["timestamp"].min()).isoformat(),
            "last_timestamp": pd.Timestamp(frame["timestamp"].max()).isoformat(),
            "updated_at": now.isoformat(),
        }
        target = directory / "manifest.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(target)
