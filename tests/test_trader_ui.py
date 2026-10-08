from datetime import datetime, timedelta, timezone

from apps.research_ui.components.trader_ui import (
    data_freshness, safe_action, strategy_guide, trade_direction,
)


def test_trade_action_maps_direction_without_call_bias():
    bullish = {"recommendation": "TRADE", "trend_regime": "TREND_UP"}
    bearish = {"recommendation": "TRADE", "trend_regime": "TREND_DOWN"}
    assert safe_action(bullish)[0] == "BUY CE"
    assert safe_action(bearish)[0] == "BUY PE"
    assert trade_direction(bearish) == "BEARISH"


def test_trade_action_fails_closed_when_direction_is_unknown():
    action, tone, reason = safe_action({"recommendation": "TRADE"})
    assert action == "NO TRADE"
    assert tone == "warning"
    assert "blocked" in reason.lower()


def test_skip_never_becomes_an_option_order():
    assert safe_action({
        "recommendation": "SKIP", "trend_regime": "TREND_UP",
    })[0] == "NO TRADE"


def test_freshness_distinguishes_live_and_research():
    now = datetime(2026, 7, 27, 10, tzinfo=timezone.utc)
    fresh = data_freshness(now - timedelta(seconds=30), live=True, now=now)
    stale = data_freshness(now - timedelta(minutes=10), live=True, now=now)
    research = data_freshness(now - timedelta(days=5), live=False, now=now)
    assert fresh["label"] == "LIVE" and not fresh["stale"]
    assert stale["label"] == "STALE" and stale["stale"]
    assert research["label"] == "RESEARCH REPLAY"


def test_strategy_has_plain_language_copy():
    guide = strategy_guide("MOMENTUM_BREAKOUT")
    assert guide["name"] == "Momentum Breakout"
    assert guide["summary"]
    assert guide["watch"]
