"""Resumable Dhan rolling expired-options acquisition."""

from __future__ import annotations

from datetime import date, timedelta
import json
import os
from pathlib import Path
import time

import pandas as pd
import requests


API_URL = "https://api.dhan.co/v2/charts/rollingoption"
FIELDS = ["open", "high", "low", "close", "iv", "volume", "strike", "oi", "spot"]


def date_chunks(start: date, end: date, days: int = 30):
    if not 1 <= days <= 30:
        raise ValueError("Dhan rolling-options chunks must be between 1 and 30 days")
    cursor = start
    while cursor < end:
        boundary = min(cursor + timedelta(days=days), end)
        yield cursor, boundary
        cursor = boundary


def request_plan(start: date, end: date, *, option_type: str = "PUT",
                 expiry_flag: str = "WEEK", expiry_code: int = 1,
                 relative_strike: str = "ATM") -> list[dict]:
    option_type = option_type.upper()
    expiry_flag = expiry_flag.upper()
    if option_type not in {"CALL", "PUT"}:
        raise ValueError("option_type must be CALL or PUT")
    if expiry_flag not in {"WEEK", "MONTH"}:
        raise ValueError("expiry_flag must be WEEK or MONTH")
    if expiry_code not in {0, 1, 2}:
        raise ValueError("expiry_code must be 0, 1 or 2")
    return [{
        "exchangeSegment": "NSE_FNO", "interval": "1", "securityId": 13,
        "instrument": "OPTIDX", "expiryFlag": expiry_flag, "expiryCode": expiry_code,
        "strike": relative_strike, "drvOptionType": option_type,
        "requiredData": FIELDS, "fromDate": str(left), "toDate": str(right),
    } for left, right in date_chunks(start, end)]


def response_frame(body: dict, option_type: str, *, expiry_flag: str = "WEEK",
                   expiry_code: int = 1, relative_strike: str = "ATM") -> pd.DataFrame:
    if body.get("status") == "failure" or body.get("errorCode"):
        raise RuntimeError(f"Dhan rolling-options request rejected: {body.get('errorCode', 'unknown')}")
    side = "pe" if option_type == "PUT" else "ce"
    payload = body.get("data", {}).get(side)
    if payload is None:
        return pd.DataFrame(columns=["timestamp", *FIELDS])
    required = {"timestamp", "open", "high", "low", "close"}
    if missing := required.difference(payload):
        raise ValueError(f"Rolling-options response missing fields: {sorted(missing)}")
    count = len(payload["timestamp"])
    values = {key: payload.get(key, [None] * count) for key in FIELDS}
    if any(len(value) != count for value in values.values()):
        raise ValueError("Rolling-options response arrays have inconsistent lengths")
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(payload["timestamp"], unit="s", utc=True).tz_convert("Asia/Kolkata"),
        **values,
    })
    frame["option_type"] = option_type
    frame["relative_strike"] = relative_strike
    frame["expiry_flag"] = expiry_flag
    frame["expiry_code"] = expiry_code
    if frame.timestamp.duplicated().any() or not frame.timestamp.is_monotonic_increasing:
        raise ValueError("Rolling-options timestamps must be unique and sorted")
    numeric = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or (numeric <= 0).any().any():
        raise ValueError("Rolling-options response contains invalid OHLC")
    return frame


class RollingOptionDownloader:
    def __init__(self, root: str | Path = "data/raw/dhan_rolling_options", *, requester=requests.post):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.requester = requester
        self.client_id = os.getenv("DHAN_CLIENT_ID")
        self.token = os.getenv("DHAN_ACCESS_TOKEN")

    def download(self, start: date, end: date, option_type: str = "PUT", *,
                 expiry_flag: str = "WEEK", expiry_code: int = 1,
                 relative_strike: str = "ATM") -> list[Path]:
        if not self.client_id or not self.token:
            raise EnvironmentError("DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN are required")
        manifest = self._manifest()
        files = []
        for payload in request_plan(
            start, end, option_type=option_type, expiry_flag=expiry_flag,
            expiry_code=expiry_code, relative_strike=relative_strike,
        ):
            key = (f"{option_type}_{expiry_flag}_{expiry_code}_{relative_strike}_"
                   f"{payload['fromDate']}_{payload['toDate']}")
            path = self.root / option_type.lower() / f"{payload['fromDate']}_{payload['toDate']}.parquet"
            if manifest["chunks"].get(key, {}).get("status") == "complete" and path.exists():
                files.append(path)
                continue
            response = self.requester(API_URL, headers={
                "Accept": "application/json", "Content-Type": "application/json",
                "client-id": self.client_id, "access-token": self.token,
            }, json=payload, timeout=60)
            if not response.ok:
                raise RuntimeError(f"Dhan rolling-options HTTP {response.status_code}")
            frame = response_frame(
                response.json(), option_type, expiry_flag=expiry_flag,
                expiry_code=expiry_code, relative_strike=relative_strike)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp.parquet")
            frame.to_parquet(temporary, index=False)
            temporary.replace(path)
            manifest["chunks"][key] = {
                "status": "complete", "file": str(path), "rows": len(frame),
                "from": payload["fromDate"], "to": payload["toDate"],
                "option_type": option_type, "expiry_flag": expiry_flag,
                "expiry_code": expiry_code, "relative_strike": relative_strike,
            }
            self._save(manifest)
            files.append(path)
            time.sleep(.22)
        return files

    def _manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {"version": 1, "source": API_URL, "chunks": {}}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _save(self, manifest: dict) -> None:
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(self.manifest_path)
