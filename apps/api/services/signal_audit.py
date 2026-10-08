"""Read-only live inference coverage and retrospective move audit."""

from __future__ import annotations

from collections import Counter
from datetime import date
import json
from pathlib import Path

import pandas as pd

from src.dhan_connection_manager import get_dhan_manager


ROOT = Path(__file__).resolve().parents[3]
JOURNAL = ROOT / "logs/live_inference/predictions.jsonl"
FORWARD_MINUTES = 15
REVIEW_MOVE_PCT = 0.15


def _records(session_date: str) -> list[dict]:
    if not JOURNAL.exists():
        return []
    rows: list[dict] = []
    for line in JOURNAL.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            timestamp = pd.Timestamp(row["timestamp"])
            if timestamp.strftime("%Y-%m-%d") == session_date:
                rows.append(row)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    # A reconnect can evaluate the same completed candle again. Keep the last result.
    return list({str(row["timestamp"]): row for row in rows}.values())


def _session_candles(session_date: str) -> pd.DataFrame:
    frame = get_dhan_manager().snapshot().get("candles", {}).get("NIFTY", pd.DataFrame())
    if frame.empty or "timestamp" not in frame:
        return pd.DataFrame()
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    return result[result.timestamp.dt.strftime("%Y-%m-%d") == session_date].sort_values("timestamp")


def _event(row: dict, candles: pd.DataFrame) -> dict:
    timestamp = pd.Timestamp(row["timestamp"])
    if timestamp.tzinfo is None and not candles.empty:
        timestamp = timestamp.tz_localize(candles.timestamp.dt.tz)
    current = (
        candles[candles.timestamp == timestamp]
        if not candles.empty else pd.DataFrame())
    future = (
        candles[candles.timestamp > timestamp].head(FORWARD_MINUTES)
        if not candles.empty else pd.DataFrame())
    close = float(current.close.iloc[0]) if len(current) else None
    up = down = None
    if close and len(future):
        up = round((float(future.high.max()) / close - 1) * 100, 3)
        down = round((float(future.low.min()) / close - 1) * 100, 3)
    decision = str(row.get("decision", "SKIP"))
    review = decision != "TRADE" and max(up or 0, abs(down or 0)) >= REVIEW_MOVE_PCT
    probability = row.get("probability")
    model_ran = probability is not None
    return {
        "timestamp": timestamp.isoformat(),
        "setup": str(row.get("strategy", "NONE")) if model_ran else "NONE",
        "stage1": "PASS" if model_ran else "BLOCKED",
        "model": "EVALUATED" if model_ran else "NOT_RUN",
        "probability": probability,
        "threshold": 0.90,
        "decision": decision,
        "action": "BUY CE" if decision == "TRADE" else "NO TRADE",
        "reason": str(row.get("reason", "UNKNOWN")),
        "forward_up_pct": up,
        "forward_down_pct": down,
        "review_candidate": review,
        "outcome_window_complete": len(future) >= FORWARD_MINUTES,
    }


def signal_audit_payload(session_date: str | None = None) -> dict:
    selected = session_date or date.today().isoformat()
    try:
        pd.Timestamp(selected)
    except ValueError as exc:
        raise ValueError("date must use YYYY-MM-DD") from exc
    candles = _session_candles(selected)
    records = _records(selected)
    events = [_event(row, candles) for row in records]
    evaluated = sum(event["model"] == "EVALUATED" for event in events)
    approved = sum(event["decision"] == "TRADE" for event in events)
    review = sum(event["review_candidate"] for event in events)
    total_candles = len(candles)
    audited = len(events)
    reasons = Counter(event["reason"] for event in events)
    return {
        "session_date": selected,
        "scope": "ATM_CALL_ONLY",
        "threshold": 0.90,
        "methodology": {
            "forward_minutes": FORWARD_MINUTES,
            "review_move_pct": REVIEW_MOVE_PCT,
            "note": "Retrospective review candidates are not live signals or proven trades.",
        },
        "summary": {
            "market_candles": total_candles,
            "audited_candles": audited,
            "coverage_pct": round(audited / total_candles * 100, 1) if total_candles else 0.0,
            "model_evaluations": evaluated,
            "approved_signals": approved,
            "review_candidates": review,
            "data_gaps": max(total_candles - audited, 0),
        },
        "reason_breakdown": [
            {"reason": reason, "count": count}
            for reason, count in reasons.most_common(8)
        ],
        "events": list(reversed(events[-180:])),
    }
