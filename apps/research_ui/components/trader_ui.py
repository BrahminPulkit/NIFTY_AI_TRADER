"""Novice-facing, fail-closed presentation helpers."""

from __future__ import annotations

from datetime import datetime

import pandas as pd


STRATEGY_GUIDE = {
    "PULLBACK_BREAKOUT": {
        "name": "Pullback Breakout",
        "summary": "Price resumed its trend after a temporary pullback.",
        "watch": "Wait for the breakout level to hold after the pullback.",
    },
    "MOMENTUM_BREAKOUT": {
        "name": "Momentum Breakout",
        "summary": "Price broke a key level with trend and momentum aligned.",
        "watch": "Avoid chasing if price moves too far beyond the breakout.",
    },
    "EXTENDED_BREAKOUT": {
        "name": "Extended Breakout",
        "summary": "The breakout is valid, but price is already extended.",
        "watch": "Risk is higher because the move may be late.",
    },
    "CONTINUATION_BREAKOUT": {
        "name": "Continuation Breakout",
        "summary": "An established trend is attempting another continuation leg.",
        "watch": "The prior trend must remain intact.",
    },
}


def strategy_guide(value: object) -> dict[str, str]:
    key = str(value or "").upper()
    if key in {"", "NONE", "NO_SETUP"}:
        return {
            "name": "No Active Setup",
            "summary": "The live scanner is running, but no approved strategy pattern is present now.",
            "watch": "Continue scanning; the model runs only after a complete setup forms.",
        }
    return STRATEGY_GUIDE.get(key, {
        "name": "Strategy unavailable",
        "summary": "No verified strategy classification is available.",
        "watch": "Wait for a completed setup.",
    })


def trade_direction(snapshot: dict) -> str:
    """Infer direction only from explicit directional fields."""
    values = (
        snapshot.get("direction"),
        snapshot.get("entry_signal"),
        snapshot.get("trend_regime"),
        snapshot.get("market_regime"),
    )
    for value in values:
        text = str(value or "").upper()
        if text in {"1", "LONG", "BULLISH", "BUY_CE"} or "UP" in text:
            return "BULLISH"
        if text in {"-1", "SHORT", "BEARISH", "BUY_PE"} or "DOWN" in text:
            return "BEARISH"
    return "UNKNOWN"


def safe_action(snapshot: dict) -> tuple[str, str, str]:
    """Return action, tone and plain-language reason without guessing."""
    if str(snapshot.get("recommendation", "SKIP")).upper() != "TRADE":
        return "NO TRADE", "warning", "The complete setup is not approved."
    direction = trade_direction(snapshot)
    if direction == "BULLISH":
        return "BUY CE", "bull", "A bullish setup passed the decision gate."
    if direction == "BEARISH":
        return "BUY PE", "bear", "A bearish setup passed the decision gate."
    return "NO TRADE", "warning", "Direction is not explicit, so the signal is blocked."


def data_freshness(timestamp: object, live: bool = False,
                   now: datetime | None = None) -> dict[str, object]:
    parsed = pd.to_datetime(timestamp, errors="coerce")
    if pd.isna(parsed):
        return {"label": "UNAVAILABLE", "age": "Unknown age", "stale": True}
    current = pd.Timestamp(now or datetime.now().astimezone())
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize(current.tzinfo)
    else:
        current = current.tz_convert(parsed.tzinfo)
    seconds = max(0, int((current - parsed).total_seconds()))
    if seconds < 60:
        age = f"{seconds}s old"
    elif seconds < 3600:
        age = f"{seconds // 60}m old"
    elif seconds < 86400:
        age = f"{seconds // 3600}h old"
    else:
        age = f"{seconds // 86400}d old"
    stale_after = 180 if live else 86400
    return {
        "label": "LIVE" if live and seconds <= stale_after else
                 "STALE" if live else "RESEARCH REPLAY",
        "age": age,
        "stale": seconds > stale_after,
    }
