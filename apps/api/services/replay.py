"""Chronological intraday replay using frozen OOF and decision artifacts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import json

import pandas as pd

from src.live_prediction_engine import FrozenDecisionReplay, FrozenOOFReplay, LivePredictionEngine
from src.live_feature_generator import build_live_contract_row
from src.feature_pipeline import build_features
from src.option_feature_pipeline import OPTION_FEATURES, OPTION_FEATURE_VERSION, build_option_features
from src.setup_engine import SetupContract, build_setups
from src.production_predictor import ProductionPredictor
from src.dhan_connection_manager import get_dhan_manager
from src.put_shadow_inference import PutShadowInferenceEngine
from src.multistrategy_shadow_inference import MultiStrategyShadowEngine

ROOT = Path(__file__).resolve().parents[3]
ALIGNED = ROOT / "data/aligned/canonical_index_option_aligned.parquet"
PREDICTIONS = ROOT / "data/prediction/prediction_dataset_v1.parquet"
OOF = ROOT / "data/prediction/oof/oof_predictions_catboost.parquet"
THRESHOLD = 0.9
DECISIONS = ROOT / "data/runtime/decision_observations.parquet"
BROKER_REPLAY_CACHE = ROOT / "data/runtime/replay_cache"
ROLLING_PUT = ROOT / "data/processed/nifty_atm_put_rolling_1min.parquet"
ROLLING_REPLAY_CACHE = ROOT / "data/runtime/rolling_replay_cache"


class _ReadOnlyLogger:
    def append(self, _record) -> None:
        return None


@lru_cache(maxsize=1)
def _engine() -> LivePredictionEngine:
    return LivePredictionEngine(
        FrozenOOFReplay(OOF), FrozenDecisionReplay(PREDICTIONS),
        _ReadOnlyLogger(), THRESHOLD,
    )


@lru_cache(maxsize=8)
def _session_candles(session_date: str) -> pd.DataFrame:
    start = pd.Timestamp(f"{session_date} 00:00:00", tz="Asia/Kolkata")
    end = start + pd.Timedelta(days=1)
    frame = pd.read_parquet(ALIGNED, columns=[
        "timestamp", "index_open", "index_high", "index_low", "index_close",
        "index_volume", "option_open", "option_high", "option_low", "option_close",
        "option_volume",
    ], filters=[("timestamp", ">=", start), ("timestamp", "<", end)])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
    return frame.sort_values("timestamp").reset_index(drop=True)


def available_sessions() -> list[str]:
    timestamps = _engine().predictor.frame.index
    return sorted(pd.Index(timestamps.strftime("%Y-%m-%d")).unique(), reverse=True)


def session_payload(session_date: str) -> dict:
    try:
        selected = pd.Timestamp(session_date)
    except ValueError as exc:
        raise ValueError("Replay date must use YYYY-MM-DD") from exc
    if selected.strftime("%Y-%m-%d") != session_date:
        raise ValueError("Replay date must use YYYY-MM-DD")
    sessions = available_sessions()
    if session_date not in sessions:
        raise LookupError("No frozen OOF replay session exists for this date")
    day = _session_candles(session_date)
    if day.empty:
        raise LookupError("No synchronized candle data exists for this date")
    start, end = day.timestamp.iloc[0], day.timestamp.iloc[-1]
    timestamps = _engine().predictor.frame.loc[start:end].index
    events = []
    for timestamp in timestamps:
        result = _engine().process_replay_timestamp(timestamp)
        events.append({
            "timestamp": result.timestamp.isoformat(),
            "action": "BUY CE" if result.decision == "TRADE" else "NO TRADE",
            "decision": result.decision,
            "reason": result.reason,
            "probability": result.probability,
            "confidence": result.confidence,
            "strategy": result.strategy,
            "regime": result.regime,
            "source": result.source,
        })
    rows = []
    for row in day.itertuples(index=False):
        rows.append({
            "timestamp": row.timestamp.isoformat(),
            "open": float(row.index_open), "high": float(row.index_high),
            "low": float(row.index_low), "close": float(row.index_close),
            "volume": None if pd.isna(row.index_volume) else float(row.index_volume),
            "option_open": float(row.option_open), "option_high": float(row.option_high),
            "option_low": float(row.option_low), "option_close": float(row.option_close),
            "option_volume": None if pd.isna(row.option_volume) else float(row.option_volume),
        })
    return {
        "session_date": session_date,
        "source": "FROZEN_OOF_REPLAY",
        "threshold": THRESHOLD,
        "candles": rows,
        "events": events,
        "summary": {
            "candle_count": len(rows), "setup_count": len(events),
            "approved_signal_count": sum(event["decision"] == "TRADE" for event in events),
            "session_start": rows[0]["timestamp"], "session_end": rows[-1]["timestamp"],
        },
    }


def broker_sessions() -> list[str]:
    snapshot = get_dhan_manager().snapshot()
    index = snapshot.get("candles", {}).get("NIFTY", pd.DataFrame())
    option = snapshot.get("candles", {}).get("NIFTY_ATM_CE", pd.DataFrame())
    put = snapshot.get("candles", {}).get("NIFTY_ATM_PE", pd.DataFrame())
    if index.empty or option.empty or put.empty:
        return []
    common = pd.Index(pd.to_datetime(index.timestamp).dt.strftime("%Y-%m-%d")).intersection(
        pd.Index(pd.to_datetime(option.timestamp).dt.strftime("%Y-%m-%d"))).intersection(
        pd.Index(pd.to_datetime(put.timestamp).dt.strftime("%Y-%m-%d")))
    return sorted(common.unique(), reverse=True)


@lru_cache(maxsize=1)
def rolling_sessions() -> list[str]:
    cache_path = ROOT / "data/runtime/rolling_replay_sessions.json"
    fingerprint = f"{ALIGNED.stat().st_size}:{ALIGNED.stat().st_mtime_ns}|{ROLLING_PUT.stat().st_size}:{ROLLING_PUT.stat().st_mtime_ns}"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint:
            return cached["sessions"]
    call_dates = pd.to_datetime(pd.read_parquet(ALIGNED, columns=["timestamp"]).timestamp, utc=True).dt.tz_convert(
        "Asia/Kolkata").dt.strftime("%Y-%m-%d")
    put_dates = pd.to_datetime(pd.read_parquet(ROLLING_PUT, columns=["timestamp"]).timestamp, utc=True).dt.tz_convert(
        "Asia/Kolkata").dt.strftime("%Y-%m-%d")
    values = sorted(set(call_dates).intersection(set(put_dates)), reverse=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"fingerprint": fingerprint, "sessions": values}), encoding="utf-8")
    return values


def rolling_session_payload(session_date: str) -> dict:
    if session_date not in rolling_sessions():
        raise LookupError("Selected date is unavailable in the rolling ATM CALL/PUT dataset")
    fingerprint = f"v2|{OPTION_FEATURE_VERSION}|{ALIGNED.stat().st_size}:{ALIGNED.stat().st_mtime_ns}|{ROLLING_PUT.stat().st_size}:{ROLLING_PUT.stat().st_mtime_ns}"
    cache_path = ROLLING_REPLAY_CACHE / f"{session_date}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.pop("_fingerprint", None) == fingerprint:
            return cached
    day_start = pd.Timestamp(session_date, tz="Asia/Kolkata")
    history_start, day_end = day_start - pd.Timedelta(days=10), day_start + pd.Timedelta(days=1)
    aligned = pd.read_parquet(ALIGNED, columns=[
        "timestamp", "index_open", "index_high", "index_low", "index_close", "index_volume",
        "option_open", "option_high", "option_low", "option_close", "option_volume",
    ], filters=[("timestamp", ">=", history_start), ("timestamp", "<", day_end)])
    index = aligned.rename(columns={
        "index_open": "open", "index_high": "high", "index_low": "low",
        "index_close": "close", "index_volume": "volume",
    })[["timestamp", "open", "high", "low", "close", "volume"]]
    call = aligned.rename(columns={
        "option_open": "open", "option_high": "high", "option_low": "low",
        "option_close": "close", "option_volume": "volume",
    })[["timestamp", "open", "high", "low", "close", "volume"]]
    put = pd.read_parquet(ROLLING_PUT, filters=[
        ("timestamp", ">=", history_start), ("timestamp", "<", day_end)])
    for frame in (index, call, put):
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    strategy_engine = MultiStrategyShadowEngine()
    strategy_events = strategy_engine.replay(index, call, put, session_date)
    model_status = "READY" if strategy_engine.packages else "MODEL_VALIDATION_BLOCKED"
    index_day = index.loc[index.timestamp.dt.strftime("%Y-%m-%d").eq(session_date)].set_index("timestamp")
    call_day = call.loc[call.timestamp.dt.strftime("%Y-%m-%d").eq(session_date)].set_index("timestamp")
    put_day = put.loc[put.timestamp.dt.strftime("%Y-%m-%d").eq(session_date)].set_index("timestamp")
    common = index_day.index.intersection(call_day.index).intersection(put_day.index).sort_values()
    rows = []
    for timestamp in common:
        spot, ce, pe = index_day.loc[timestamp], call_day.loc[timestamp], put_day.loc[timestamp]
        call_strike = round(float(spot.close) / 50) * 50
        put_strike = float(pe.strike) if pd.notna(pe.get("strike")) else call_strike
        rows.append({
            "timestamp": timestamp.isoformat(), "open": float(spot.open), "high": float(spot.high),
            "low": float(spot.low), "close": float(spot.close),
            "volume": None if pd.isna(spot.volume) else float(spot.volume),
            "option_open": float(ce.open), "option_high": float(ce.high), "option_low": float(ce.low),
            "option_close": float(ce.close), "option_volume": None if pd.isna(ce.volume) else float(ce.volume),
            "put_option_open": float(pe.open), "put_option_high": float(pe.high),
            "put_option_low": float(pe.low), "put_option_close": float(pe.close),
            "put_option_volume": None if pd.isna(pe.volume) else float(pe.volume),
            "call_strike": call_strike, "put_strike": put_strike,
        })
    row_lookup = {row["timestamp"]: row for row in rows}
    for event in strategy_events:
        candle = row_lookup.get(event["timestamp"], {})
        is_call = event["side"] == "CALL"
        event.update({"option_type": "CE" if is_call else "PE",
                      "strike": candle.get("call_strike" if is_call else "put_strike"),
                      "regime": "ROLLING_ATM"})
    payload = {
        "session_date": session_date, "source": "ROLLING_ATM_STRATEGY_REPLAY", "threshold": .90,
        "candles": rows, "events": [], "put_events": [], "strategy_events": strategy_events,
        "put_threshold": .85, "summary": {
            "model_status": model_status,
            "model_reason": (None if strategy_engine.packages else
                             "No CALL/PUT model passed the current option contract validation"),
            "candle_count": len(rows), "setup_count": sum(event["side"] == "CALL" for event in strategy_events),
            "approved_signal_count": sum(event["side"] == "CALL" and event["decision"] == "PAPER_SIGNAL" for event in strategy_events),
            "put_setup_count": sum(event["side"] == "PUT" for event in strategy_events),
            "put_approved_signal_count": sum(event["side"] == "PUT" and event["decision"] == "PAPER_SIGNAL" for event in strategy_events),
            "strategy_setup_count": len(strategy_events),
            "strategy_approved_signal_count": sum(event["decision"] == "PAPER_SIGNAL" for event in strategy_events),
            "session_start": rows[0]["timestamp"], "session_end": rows[-1]["timestamp"],
            "missing_candles": 375 - len(rows), "data_valid": len(rows) == 375,
        },
    }
    ROLLING_REPLAY_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"_fingerprint": fingerprint, **payload}), encoding="utf-8")
    return payload


def broker_session_payload(session_date: str) -> dict:
    if session_date not in broker_sessions():
        raise LookupError("Selected date is not available in the shared Dhan candle cache")
    snapshot = get_dhan_manager().snapshot()
    index = snapshot["candles"]["NIFTY"].copy()
    option = snapshot["candles"]["NIFTY_ATM_CE"].copy()
    put = snapshot["candles"]["NIFTY_ATM_PE"].copy()
    instruments = snapshot.get("options", {})
    ce_contract = instruments.get("NIFTY_ATM_CE") or {}
    pe_contract = instruments.get("NIFTY_ATM_PE") or {}
    fingerprint = f"v4|{OPTION_FEATURE_VERSION}|" + "|".join([
        str(len(frame)) + ":" + str(pd.to_datetime(frame.timestamp.iloc[-1]))
        for frame in (index, option, put)
    ]) + f"|{ce_contract.get('security_id')}|{pe_contract.get('security_id')}"
    cache_path = BROKER_REPLAY_CACHE / f"{session_date}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.pop("_fingerprint", None) == fingerprint:
            for candle in cached.get("candles", []):
                candle["call_strike"] = ce_contract.get("strike")
                candle["put_strike"] = pe_contract.get("strike")
            return cached
    for frame in (index, option, put):
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    index_day = index.loc[index.timestamp.dt.strftime("%Y-%m-%d").eq(session_date)]
    option_day = option.loc[option.timestamp.dt.strftime("%Y-%m-%d").eq(session_date)]
    put_day = put.loc[put.timestamp.dt.strftime("%Y-%m-%d").eq(session_date)]
    common = index_day.timestamp[
        index_day.timestamp.isin(option_day.timestamp)
        & index_day.timestamp.isin(put_day.timestamp)]
    index_day = index_day.loc[index_day.timestamp.isin(common)].sort_values("timestamp")
    option_day = option_day.loc[option_day.timestamp.isin(common)].sort_values("timestamp")
    put_day = put_day.loc[put_day.timestamp.isin(common)].sort_values("timestamp")
    if index_day.empty:
        raise LookupError("Dhan index and ATM Call candles are not aligned for this date")
    features = build_features(index.loc[index.timestamp.le(index_day.timestamp.iloc[-1])])
    setups = build_setups(features, SetupContract())
    allowed = set(pd.to_datetime(setups.loc[setups.trade_allowed.eq(1), "timestamp"], utc=True))
    predictor = ProductionPredictor(ROOT / "production_model")
    decisions = pd.read_parquet(DECISIONS)
    events = []
    put_events = []
    put_predictor = PutShadowInferenceEngine(ROOT / "models/stage2_put_candidate_v1")
    put_features = build_option_features(put).set_index("timestamp")
    index_lookup = index_day.set_index("timestamp")
    option_lookup = option_day.set_index("timestamp")
    put_lookup = put_day.set_index("timestamp")
    for timestamp in index_day.timestamp:
        if pd.Timestamp(timestamp).tz_convert("UTC") not in allowed:
            continue
        row, audit = build_live_contract_row(
            index.loc[index.timestamp.le(timestamp)], option.loc[option.timestamp.le(timestamp)],
            decisions, predictor.contract["feature_order"],
        )
        if row is None:
            continue
        output = predictor.predict(row)
        probability = float(output["probability"])
        decision_gate = str(row.iloc[0]["recommendation"])
        trade = decision_gate == "TRADE" and probability >= THRESHOLD
        reason = ("ALL_FROZEN_GATES_PASSED" if trade else
                  f"DECISION_ENGINE_{row.iloc[0]['decision_reason']}" if decision_gate != "TRADE"
                  else "PROBABILITY_BELOW_APPROVED_THRESHOLD")
        events.append({
            "timestamp": pd.Timestamp(timestamp).isoformat(), "action": "BUY CE" if trade else "NO TRADE",
            "decision": "TRADE" if trade else "SKIP", "reason": reason,
            "probability": probability, "confidence": probability,
            "strategy": str(row.iloc[0]["current_strategy"]),
            "regime": str(row.iloc[0]["market_state"]), "source": "NATIVE_MODEL_AUDIT",
            "option_type": "CE", "strike": ce_contract.get("strike"),
            "expiry": ce_contract.get("expiry"),
            "entry_premium": float(option_lookup.loc[timestamp, "close"]),
            "index_price": float(index_lookup.loc[timestamp, "close"]),
        })
        put_row = row.copy()
        put_feature = put_features.loc[timestamp]
        for name in OPTION_FEATURES:
            put_row.loc[put_row.index[0], f"option_{name}"] = put_feature[name]
        put_result = put_predictor.infer_contract_row(put_row, timestamp, audit)
        if put_result["probability"] is not None:
            put_events.append({
                "timestamp": put_result["timestamp"], "action": put_result["action"],
                "decision": put_result["decision"], "reason": put_result["reason"],
                "probability": put_result["probability"], "confidence": put_result["probability"],
                "strategy": put_result["strategy"], "regime": "BEARISH_STAGE1",
                "source": "PUT_FORWARD_PAPER_AUDIT",
                "option_type": "PE", "strike": pe_contract.get("strike"),
                "expiry": pe_contract.get("expiry"),
                "entry_premium": float(put_lookup.loc[timestamp, "close"]),
                "index_price": float(index_lookup.loc[timestamp, "close"]),
            })
    strategy_engine = MultiStrategyShadowEngine()
    strategy_events = strategy_engine.replay(index, option, put, session_date)
    model_status = "READY" if strategy_engine.packages else "MODEL_VALIDATION_BLOCKED"
    for event in strategy_events:
        contract = ce_contract if event["side"] == "CALL" else pe_contract
        event.update({"option_type": "CE" if event["side"] == "CALL" else "PE",
                      "strike": contract.get("strike"), "expiry": contract.get("expiry")})
    rows = [{
        "timestamp": row.timestamp.isoformat(), "open": float(row.open), "high": float(row.high),
        "low": float(row.low), "close": float(row.close),
        "volume": None if pd.isna(row.volume) else float(row.volume),
        "option_open": float(option_lookup.loc[row.timestamp, "open"]),
        "option_high": float(option_lookup.loc[row.timestamp, "high"]),
        "option_low": float(option_lookup.loc[row.timestamp, "low"]),
        "option_close": float(option_lookup.loc[row.timestamp, "close"]),
        "option_volume": None if pd.isna(option_lookup.loc[row.timestamp, "volume"]) else float(option_lookup.loc[row.timestamp, "volume"]),
        "put_option_open": float(put_lookup.loc[row.timestamp, "open"]),
        "put_option_high": float(put_lookup.loc[row.timestamp, "high"]),
        "put_option_low": float(put_lookup.loc[row.timestamp, "low"]),
        "put_option_close": float(put_lookup.loc[row.timestamp, "close"]),
        "put_option_volume": None if pd.isna(put_lookup.loc[row.timestamp, "volume"]) else float(put_lookup.loc[row.timestamp, "volume"]),
        "call_strike": ce_contract.get("strike"), "put_strike": pe_contract.get("strike"),
    } for row in index_day.itertuples(index=False)]
    expected = pd.date_range(f"{session_date} 09:15", f"{session_date} 15:29", freq="min", tz="Asia/Kolkata")
    missing = len(expected.difference(pd.DatetimeIndex(index_day.timestamp)))
    payload = {
        "session_date": session_date, "source": "NATIVE_MODEL_AUDIT", "threshold": THRESHOLD,
        "candles": rows, "events": events, "put_events": put_events,
        "strategy_events": strategy_events, "put_threshold": .85,
        "summary": {"candle_count": len(rows), "setup_count": len(events),
                    "model_status": model_status,
                    "model_reason": (None if strategy_engine.packages else
                                     "No CALL/PUT model passed the current option contract validation"),
                    "approved_signal_count": sum(event["decision"] == "TRADE" for event in events),
                    "put_setup_count": len(put_events),
                    "put_approved_signal_count": sum(event["action"] == "BUY PE" for event in put_events),
                    "strategy_setup_count": len(strategy_events),
                    "strategy_approved_signal_count": sum(
                        event["decision"] == "PAPER_SIGNAL" for event in strategy_events),
                    "session_start": rows[0]["timestamp"], "session_end": rows[-1]["timestamp"],
                    "missing_candles": missing, "data_valid": missing == 0},
    }
    BROKER_REPLAY_CACHE.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"_fingerprint": fingerprint, **payload}), encoding="utf-8")
    temporary.replace(cache_path)
    return payload
