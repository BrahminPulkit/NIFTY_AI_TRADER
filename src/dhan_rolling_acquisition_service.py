"""Authenticated, read-only Dhan rolling-option acquisition and policy segmentation."""

from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.dhan_rolling_option_downloader import date_chunks, response_frame


FIELDS = ["open", "high", "low", "close", "volume", "oi", "iv", "strike", "spot"]


class DhanRollingAcquisitionService:
    def __init__(self, client, root: str | Path = "data/raw/dhan_rolling_options_weekly_current_v1"):
        self.client = client
        self.root = Path(root)

    @staticmethod
    def payload(start: date, end: date, side: str, *, expiry_code: int = 0) -> dict:
        return {
            "exchangeSegment": "NSE_FNO", "interval": "1", "securityId": 13,
            "instrument": "OPTIDX", "expiryFlag": "WEEK", "expiryCode": expiry_code,
            "strike": "ATM", "drvOptionType": side, "requiredData": FIELDS,
            "fromDate": str(start), "toDate": str(end),
        }

    def acquire(self, start: date, end: date, *, expiry_code: int = 0) -> dict:
        if start >= end:
            raise ValueError("start must be earlier than exclusive end")
        frames, chunks = [], []
        for left, right in date_chunks(start, end, 30):
            for side in ("CALL", "PUT"):
                payload = self.payload(left, right, side, expiry_code=expiry_code)
                body = self.client.rolling_options(payload)
                raw_path = self.root / "raw" / side.lower() / f"{left}_{right}.json"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                self._atomic_json(raw_path, body)
                frame = response_frame(
                    body, side, expiry_flag="WEEK", expiry_code=expiry_code,
                    relative_strike="ATM")
                frame["underlying"] = "NIFTY"
                frame["option_type"] = "CE" if side == "CALL" else "PE"
                frame["source_chunk"] = f"{left}_{right}"
                frames.append(frame)
                chunks.append({"side": side, "from": str(left), "to": str(right),
                               "rows": len(frame), "raw_file": str(raw_path)})
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        normalized = self.segment(combined)
        normalized_path = self.root / "normalized" / f"{start}_{end}.parquet"
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = normalized_path.with_suffix(".tmp.parquet")
        normalized.to_parquet(temporary, index=False); temporary.replace(normalized_path)
        quality = self.quality(normalized)
        parameters = self.payload(start, end, "CALL", expiry_code=expiry_code)
        parameters["drvOptionType"] = ["CALL", "PUT"]
        manifest = {
            "version": f"dhan_weekly_expiry_{expiry_code}_atm_v1", "endpoint": "/v2/charts/rollingoption",
            "from": str(start), "to_exclusive": str(end), "parameters": parameters,
            "chunks": chunks, "normalized_file": str(normalized_path), "quality": quality,
            "segment_rule": "reset on IST session, option_type, or strike change",
            "contains_credentials": False,
        }
        self._atomic_json(self.root / "manifest.json", manifest)
        return manifest

    @staticmethod
    def segment(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        data = frame.copy()
        data["timestamp"] = pd.to_datetime(data.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
        data = data.sort_values(["option_type", "timestamp"], kind="stable").drop_duplicates(
            ["option_type", "timestamp"], keep="last")
        data["session_date"] = data.timestamp.dt.strftime("%Y-%m-%d")
        identity = data.session_date + "|" + data.option_type.astype(str) + "|" + data.strike.astype(str)
        transition = identity.ne(identity.shift())
        sequence = transition.groupby(data.option_type, sort=False).cumsum().astype(int)
        data["segment_id"] = [hashlib.sha256(
            f"{side}|{session}|{strike}|{number}".encode()).hexdigest()[:20]
            for side, session, strike, number in zip(
                data.option_type, data.session_date, data.strike, sequence)]
        return data.sort_values(["timestamp", "option_type"], kind="stable").reset_index(drop=True)

    @staticmethod
    def quality(frame: pd.DataFrame) -> dict:
        if frame.empty:
            return {"raw_rows": 0, "clean_rows": 0}
        ce, pe = frame.loc[frame.option_type.eq("CE")], frame.loc[frame.option_type.eq("PE")]
        sessions = frame.groupby(["session_date", "option_type"]).timestamp.nunique()
        table = sessions.eq(375).unstack(fill_value=False)
        complete_both = int((table["CE"] & table["PE"]).sum()) \
            if {"CE", "PE"}.issubset(table.columns) else 0
        return {
            "raw_rows": len(frame), "clean_rows": len(frame), "ce_rows": len(ce), "pe_rows": len(pe),
            "trading_days": int(frame.session_date.nunique()), "complete_sessions": complete_both,
            "partial_side_sessions": int((~sessions.eq(375)).sum()),
            "missing_minutes": int((375 - sessions).clip(lower=0).sum()),
            "duplicate_timestamps": int(frame.duplicated(["option_type", "timestamp"]).sum()),
            "strike_changes": int(frame.groupby(["session_date", "option_type"]).strike.apply(
                lambda value: max(0, int(value.ne(value.shift()).sum()) - 1)).sum()),
            "contract_segments": int(frame.segment_id.nunique()),
            "missing_oi": int(frame.oi.isna().sum()), "missing_iv": int(frame.iv.isna().sum()),
            "missing_spot": int(frame.spot.isna().sum()),
            "ce_pe_timestamp_overlap": len(set(ce.timestamp) & set(pe.timestamp)),
        }

    @staticmethod
    def _atomic_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)
