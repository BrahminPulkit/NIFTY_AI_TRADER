"""JSON-safe presentation contracts built from existing runtime state."""

from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
from typing import Any

import pandas as pd

from apps.research_ui.components.trader_ui import (
    data_freshness,
    safe_action,
    strategy_guide,
    trade_direction,
)

ROOT = Path(__file__).resolve().parents[2]
JOINED_PREDICTIONS = ROOT / "data/research/probability_threshold/joined_probability_dataset.parquet"


def _clean(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item"):
        return _clean(value.item())
    return value


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _clean(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def latest_prediction() -> tuple[dict[str, Any], bool, str]:
    snapshot: dict[str, Any] = {}
    if JOINED_PREDICTIONS.exists():
        joined = pd.read_parquet(JOINED_PREDICTIONS)
        if len(joined):
            joined["timestamp"] = pd.to_datetime(joined["timestamp"])
            snapshot = joined.sort_values("timestamp").iloc[-1].to_dict()
    journal = ROOT / "logs/live_inference/predictions.jsonl"
    if not journal.exists() or not journal.stat().st_size:
        return snapshot, False, "Frozen research record"
    try:
        import json

        record = json.loads(journal.read_text(encoding="utf-8").splitlines()[-1])
    except (OSError, ValueError, IndexError):
        return snapshot, False, "Frozen research record"
    return {
        **snapshot,
        "timestamp": record.get("timestamp"),
        "catboost_predicted_probability": record.get("probability") or 0,
        "strategy_confidence_score": record.get("confidence") or 0,
        "recommendation": record.get("decision", "SKIP"),
        "decision_reason": record.get("reason", "UNAVAILABLE"),
        "current_strategy": record.get("strategy", "UNAVAILABLE"),
        "market_state": record.get("regime", "UNAVAILABLE"),
        "live_model_version": record.get("model_version"),
        "live_latency_ms": record.get("latency_ms"),
        "risk_status": record.get("risk_status"),
    }, True, "Native inference journal"


def desk_contract(cache: dict[str, Any]) -> dict[str, Any]:
    snapshot, live, source = latest_prediction()
    cached_signal = cache.get("signal")
    if cached_signal:
        snapshot = {
            **snapshot,
            "timestamp": cached_signal.get("timestamp"),
            "catboost_predicted_probability": cached_signal.get("probability") or 0,
            "strategy_confidence_score": cached_signal.get("confidence") or 0,
            "recommendation": cached_signal.get("decision", "SKIP"),
            "decision_reason": cached_signal.get("reason", "UNAVAILABLE"),
            "current_strategy": cached_signal.get("strategy", "UNAVAILABLE"),
            "market_state": cached_signal.get("regime", "UNAVAILABLE"),
            "direction": "BULLISH" if cached_signal.get("action") == "BUY CE" else "UNKNOWN",
        }
        live, source = True, "Shared live inference worker"
    if not snapshot:
        return {"available": False, "reason": "No verified prediction is available"}
    probability = float(snapshot.get("catboost_predicted_probability", 0) or 0)
    action, tone, reason = safe_action(snapshot)
    if cached_signal:
        action = cached_signal.get("action", "NO TRADE")
        tone = "bull" if action == "BUY CE" else "warning"
        reason = str(cached_signal.get("reason", reason)).replace("_", " ").title()
    direction = trade_direction(snapshot)
    guide = strategy_guide(snapshot.get("current_strategy"))
    freshness = data_freshness(snapshot.get("timestamp"), live)
    holding = _clean(snapshot.get("expected_holding_minutes"))
    if holding is not None:
        holding = float(holding)
        if holding <= 0 or holding > 1440:
            holding = None
    raw_regime = snapshot.get("market_state", snapshot.get("market_regime"))
    regime = (
        None if raw_regime is None or str(raw_regime).lower() in {"", "none", "nan", "<na>"}
        else str(raw_regime).replace("_", " ").title()
    )
    call_only = bool(
        cached_signal
        and cached_signal.get("instrument_scope") == "ATM_CALL_ONLY"
    )
    option_key = (
        "NIFTY_ATM_CE" if call_only or direction != "BEARISH"
        else "NIFTY_ATM_PE"
    )
    contract = cache.get("options", {}).get(option_key)
    checks = [
        {"label": "Trend direction", "ready": direction != "UNKNOWN"},
        {"label": "Strategy selected", "ready": str(snapshot.get("current_strategy", "")).upper() in {
            "PULLBACK_BREAKOUT", "MOMENTUM_BREAKOUT", "EXTENDED_BREAKOUT",
            "CONTINUATION_BREAKOUT",
        }},
        {"label": "Decision gate", "ready": str(snapshot.get("recommendation", "SKIP")).upper() == "TRADE"},
        {"label": "Probability threshold", "ready": probability >= .90},
        {"label": "Risk context", "ready": snapshot.get("expected_drawdown") is not None},
        {"label": "Option contract", "ready": contract is not None},
        {"label": "Fresh market data", "ready": freshness["label"] == "LIVE" and not freshness["stale"]},
    ]
    chain = cache.get("option_chain", [])
    writer_rows = []
    for row in chain:
        try:
            oi = float(row.get("oi"))
            strike = float(row.get("strike"))
        except (TypeError, ValueError):
            continue
        option_type = str(row.get("option_type", "")).upper()
        if option_type in {"CE", "PE"} and oi >= 0:
            writer_rows.append({
                "option_type": option_type, "strike": strike, "oi": oi,
                "ltp": _clean(row.get("ltp")), "expiry": _clean(row.get("expiry")),
            })
    ce_rows = [row for row in writer_rows if row["option_type"] == "CE"]
    pe_rows = [row for row in writer_rows if row["option_type"] == "PE"]
    strongest_ce = max(ce_rows, key=lambda row: row["oi"], default=None)
    strongest_pe = max(pe_rows, key=lambda row: row["oi"], default=None)
    ce_total = sum(row["oi"] for row in ce_rows)
    pe_total = sum(row["oi"] for row in pe_rows)
    dominant = (
        "CALL WRITERS" if ce_total > pe_total
        else "PUT WRITERS" if pe_total > ce_total
        else "BALANCED"
    )
    return {
        "available": True,
        "action": action,
        "tone": tone,
        "reason": reason,
        "probability": probability,
        "confidence": float(snapshot.get("strategy_confidence_score", probability) or 0),
        "strategy": guide,
        "direction": direction,
        "regime": regime or "Unavailable",
        "holding_minutes": holding,
        "expected_drawdown": _clean(snapshot.get("expected_drawdown")),
        "expected_expansion": _clean(snapshot.get("expected_premium_expansion")),
        "timestamp": _clean(snapshot.get("timestamp")),
        "source": source,
        "freshness": freshness,
        "contract": {key: _clean(value) for key, value in contract.items()} if contract else None,
        "checks": checks,
        "writer_activity": {
            "method": "OPEN_INTEREST_PROXY",
            "dominant": dominant,
            "call_total_oi": ce_total,
            "put_total_oi": pe_total,
            "strongest_call": strongest_ce,
            "strongest_put": strongest_pe,
        },
    }


def candle_contract(cache: dict[str, Any], symbol: str = "NIFTY") -> list[dict[str, Any]]:
    frame = cache.get("candles", {}).get(symbol, pd.DataFrame())
    if frame is None or frame.empty:
        return []
    view = frame.tail(600).copy()
    if "timestamp" not in view.columns:
        view = view.reset_index()
        view = view.rename(columns={view.columns[0]: "timestamp"})
    return records(view)


def paper_summary() -> dict[str, Any]:
    stats_path = ROOT / "reports/paper_trading/risk_statistics.csv"
    journal_path = ROOT / "reports/paper_trading/trade_journal.csv"
    stats = pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()
    journal = pd.read_csv(journal_path) if journal_path.exists() else pd.DataFrame()
    return {
        "statistics": records(stats.head(1))[0] if len(stats) else {},
        "journal": records(journal.tail(50).iloc[::-1]) if len(journal) else [],
        "updated_at": datetime.now().astimezone().isoformat(),
    }
