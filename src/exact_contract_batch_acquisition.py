"""Evidence-bound exact-contract batch acquisition for research only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
import json
from pathlib import Path

import pandas as pd

from src.dhan_live_broker import Instrument
from src.historical_contract_resolution_service import HistoricalContractResolver


EXPECTED_MINUTES = 375
OUTPUT_COLUMNS = [
    "timestamp", "underlying", "expiry", "strike", "option_type",
    "security_id", "contract_segment_id", "open", "high", "low", "close",
    "volume", "spot",
]


@dataclass(frozen=True)
class SessionCoverage:
    date: str
    ce_contract: str | None
    pe_contract: str | None
    expiry: str | None
    strike: float | None
    ce_security_id: str | None
    pe_security_id: str | None
    ce_candles: int
    pe_candles: int
    overlap: int
    missing_ce: int
    missing_pe: int
    duplicate_timestamps: int
    status: str
    reason: str | None


class ExactContractBatchAcquisition:
    def __init__(self, client, *, resolver=None,
                 output_root: str | Path = "data/normalized/historical_options/validated_batch",
                 report_root: str | Path = "reports/phase5"):
        self.client = client
        self.resolver = resolver or HistoricalContractResolver()
        self.output_root = Path(output_root)
        self.report_root = Path(report_root)

    def acquire(self, dates: list[str]) -> dict:
        dates = sorted(dict.fromkeys(str(pd.Timestamp(value).date()) for value in dates))
        coverage, ready_frames, api_failures = [], [], []
        for session_date in dates:
            try:
                result, frame = self._acquire_session(session_date)
            except Exception as exc:
                result, frame = SessionCoverage(
                    session_date, None, None, None, None, None, None,
                    0, 0, 0, EXPECTED_MINUTES, EXPECTED_MINUTES, 0,
                    "PARTIAL", f"DHAN_API_ERROR: {exc}"), None
                api_failures.append({"date": session_date, "error": str(exc)})
            coverage.append(asdict(result))
            if result.status == "TRAINING_READY" and frame is not None:
                ready_frames.append(frame)
        table = pd.DataFrame(coverage)
        self._write_reports(table, api_failures)
        training_path = None
        if len(ready_frames) >= 20:
            combined = pd.concat(ready_frames, ignore_index=True).sort_values(
                ["timestamp", "option_type"], kind="stable")
            training_path = self.output_root / "training_ready.parquet"
            self._atomic_parquet(combined, training_path)
        return {
            "sessions_attempted": len(dates),
            "training_ready_sessions": int(table.status.eq("TRAINING_READY").sum()),
            "partial_sessions": int(table.status.eq("PARTIAL").sum()),
            "unresolved_sessions": int(table.status.eq("UNRESOLVED").sum()),
            "api_failures": api_failures,
            "training_ready_file": str(training_path) if training_path else None,
            "standalone_validation_eligible": len(ready_frames) >= 20,
        }

    def _acquire_session(self, session_date: str):
        ce = self.resolver.resolve_contract(date=session_date, option_type="CE")
        pe = self.resolver.resolve_contract(date=session_date, option_type="PE")
        if ce.status != "RESOLVED" or pe.status != "RESOLVED":
            reasons = sorted({item.reason for item in (ce, pe) if item.reason})
            return SessionCoverage(
                session_date, None, None, ce.expiry or pe.expiry,
                ce.strike or pe.strike, ce.security_id, pe.security_id,
                0, 0, 0, EXPECTED_MINUTES, EXPECTED_MINUTES, 0,
                "UNRESOLVED", " | ".join(reasons)), None
        if ce.expiry != pe.expiry or ce.strike != pe.strike:
            return SessionCoverage(
                session_date, self._contract(ce), self._contract(pe), None, None,
                ce.security_id, pe.security_id, 0, 0, 0, EXPECTED_MINUTES,
                EXPECTED_MINUTES, 0, "UNRESOLVED", "CE_PE_CONTRACT_POLICY_MISMATCH"), None
        frames = {}
        for resolution in (ce, pe):
            instrument = Instrument(
                f"NIFTY_ATM_{resolution.option_type}", resolution.security_id,
                "NSE_FNO", "OPTIDX",
                f"NIFTY {resolution.expiry} {resolution.strike:g} {resolution.option_type}",
                resolution.expiry, resolution.strike, resolution.option_type)
            start = f"{session_date} 09:15:00"
            end = f"{(date.fromisoformat(session_date) + timedelta(days=1))} 00:00:00"
            raw = self.client.completed_candles(instrument, start, end)
            frames[resolution.option_type] = self._normalize(
                raw, resolution, session_date)
        ce_frame, pe_frame = frames["CE"], frames["PE"]
        ce_index, pe_index = set(ce_frame.timestamp), set(pe_frame.timestamp)
        duplicate_count = int(ce_frame.timestamp.duplicated().sum() + pe_frame.timestamp.duplicated().sum())
        expected = pd.date_range(
            f"{session_date} 09:15", f"{session_date} 15:29",
            freq="min", tz="Asia/Kolkata")
        missing_ce = len(expected.difference(pd.DatetimeIndex(ce_frame.timestamp)))
        missing_pe = len(expected.difference(pd.DatetimeIndex(pe_frame.timestamp)))
        complete = (
            len(ce_frame) == len(pe_frame) == EXPECTED_MINUTES
            and len(ce_index & pe_index) == EXPECTED_MINUTES
            and missing_ce == missing_pe == duplicate_count == 0
            and ce_frame.contract_segment_id.nunique() == 1
            and pe_frame.contract_segment_id.nunique() == 1)
        status = "TRAINING_READY" if complete else "PARTIAL"
        reason = None if complete else "INCOMPLETE_OR_MISALIGNED_SESSION"
        combined = pd.concat([ce_frame, pe_frame], ignore_index=True).sort_values(
            ["timestamp", "option_type"], kind="stable")
        directory = self.output_root / session_date
        if not combined.empty:
            self._atomic_parquet(ce_frame, directory / "ce.parquet")
            self._atomic_parquet(pe_frame, directory / "pe.parquet")
            self._atomic_parquet(combined, directory / "ce_pe.parquet")
        result = SessionCoverage(
            session_date, self._contract(ce), self._contract(pe), ce.expiry,
            ce.strike, ce.security_id, pe.security_id, len(ce_frame), len(pe_frame),
            len(ce_index & pe_index), missing_ce, missing_pe, duplicate_count,
            status, reason)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text(
            json.dumps(asdict(result), indent=2), encoding="utf-8")
        return result, combined if complete else None

    def _normalize(self, frame, resolution, session_date: str) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        data = frame.copy()
        data["timestamp"] = pd.to_datetime(
            data.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
        data = data.loc[
            data.timestamp.dt.strftime("%Y-%m-%d").eq(session_date)].copy()
        data["underlying"] = "NIFTY"
        data["expiry"] = resolution.expiry
        data["strike"] = resolution.strike
        data["option_type"] = resolution.option_type
        data["security_id"] = resolution.security_id
        data["contract_segment_id"] = resolution.contract_segment_id
        data["spot"] = self._spot(data.timestamp)
        return data[OUTPUT_COLUMNS].sort_values("timestamp", kind="stable")

    def _spot(self, timestamps: pd.Series) -> pd.Series:
        if len(timestamps) == 0:
            return pd.Series(dtype=float)
        lookup = self.resolver.index.set_index("timestamp").index_close
        return timestamps.map(lookup)

    @staticmethod
    def _contract(resolution) -> str:
        return (f"NIFTY {resolution.expiry} {resolution.strike:g} "
                f"{resolution.option_type} [{resolution.security_id}]")

    def _write_reports(self, table: pd.DataFrame, api_failures: list[dict]) -> None:
        self.report_root.mkdir(parents=True, exist_ok=True)
        table.to_csv(self.report_root / "exact_contract_data_coverage.csv", index=False)
        payload = {
            "sessions_attempted": len(table),
            "status_counts": table.status.value_counts().to_dict(),
            "rows": table.to_dict("records"), "api_failures": api_failures,
            "training_ready_created": bool(table.status.eq("TRAINING_READY").sum() >= 20),
        }
        (self.report_root / "exact_contract_data_coverage.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.parquet")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
