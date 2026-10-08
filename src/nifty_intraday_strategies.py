"""Causal NIFTY intraday research strategies for 1-minute OHLCV data.

Signals are calculated on the NIFTY index while volume/VWAP may use the
matched near-month NIFTY future.  This module never sends broker orders.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


IST = "Asia/Kolkata"
STRATEGIES = ("GOLDEN_RULE", "ORB_RETEST", "VWAP_MEAN_REVERSION")


@dataclass(frozen=True)
class IntradayConfig:
    target_r: float = 2.0
    atr_period: int = 14
    stop_buffer_atr: float = 0.08
    minimum_stop_atr: float = 0.35
    maximum_stop_atr: float = 1.50
    mean_reversion_distance_atr: float = 0.60
    trend_adx_minimum: float = 20.0
    sideways_adx_maximum: float = 20.0
    minimum_ema_gap_atr: float = 0.08
    maximum_vwap_crosses_15m: int = 4
    maximum_pullback_volume_ratio: float = 0.90
    minimum_trigger_volume_ratio: float = 1.05
    maximum_pullback_range_atr: float = 3.0
    scalp_partial_r: float = 1.0
    scalp_final_r: float = 1.8
    scalp_time_stop_minutes: int = 20
    flat_vwap_slope_atr: float = 0.12
    orb_retest_tolerance_atr: float = 0.12
    orb_retest_bars: int = 20
    maximum_hold_minutes: int = 60
    entry_start: str = "09:30"
    entry_end: str = "14:45"
    square_off: str = "15:15"
    cooldown_minutes: int = 5
    maximum_trades_per_day: int = 5
    maximum_losses_per_day: int = 2
    cost_points_per_trade: float = 1.0


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _wilder(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _completed_timeframe(frame: pd.DataFrame, rule: str, suffix: str) -> pd.DataFrame:
    """Aggregate bars and timestamp them when the interval is fully known."""
    grouped = (
        frame.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        .resample(rule, label="right", closed="left", origin="start_day")
        .agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        })
        .dropna()
        .reset_index()
    )
    grouped[f"ema9_{suffix}"] = _ema(grouped["close"], 9)
    grouped[f"ema20_{suffix}"] = _ema(grouped["close"], 20)
    grouped[f"ema20_slope_{suffix}"] = (
        grouped[f"ema20_{suffix}"] - grouped[f"ema20_{suffix}"].shift(2)
    )
    previous_close = grouped["close"].shift()
    true_range = pd.concat(
        [
            grouped["high"] - grouped["low"],
            (grouped["high"] - previous_close).abs(),
            (grouped["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    grouped[f"atr_{suffix}"] = _wilder(true_range, 14)
    grouped[f"volume_average_{suffix}"] = grouped["volume"].shift(1).rolling(10).mean()
    grouped[f"volume_ratio_{suffix}"] = (
        grouped["volume"] / grouped[f"volume_average_{suffix}"].replace(0, np.nan)
    )
    return grouped.rename(
        columns={
            name: f"{name}_{suffix}"
            for name in ("open", "high", "low", "close", "volume")
        }
    )


def prepare_intraday_data(
    index_bars: pd.DataFrame,
    futures_bars: pd.DataFrame | None = None,
    config: IntradayConfig = IntradayConfig(),
) -> pd.DataFrame:
    """Build causal 1m/5m/15m features from completed candles only."""
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required.difference(index_bars.columns)
    if missing:
        raise ValueError(f"index_bars missing columns: {sorted(missing)}")
    data = index_bars.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    if data["timestamp"].dt.tz is None:
        data["timestamp"] = data["timestamp"].dt.tz_localize(IST)
    else:
        data["timestamp"] = data["timestamp"].dt.tz_convert(IST)
    data = data.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    data["session"] = data["timestamp"].dt.date

    if futures_bars is not None and not futures_bars.empty:
        if "volume" not in futures_bars.columns:
            raise ValueError("futures_bars missing column: volume")
        # NIFTY index responses may also contain a zero/placeholder volume
        # column. Futures volume is the intentional analytical source here.
        data = data.drop(columns=["volume"], errors="ignore")
        volume = futures_bars[["timestamp", "volume"]].copy()
        volume["timestamp"] = pd.to_datetime(volume["timestamp"])
        if volume["timestamp"].dt.tz is None:
            volume["timestamp"] = volume["timestamp"].dt.tz_localize(IST)
        else:
            volume["timestamp"] = volume["timestamp"].dt.tz_convert(IST)
        volume = volume.sort_values("timestamp").drop_duplicates("timestamp")
        data = data.merge(volume, on="timestamp", how="left", validate="one_to_one")
    elif "volume" not in data:
        data["volume"] = 0.0
    data["volume"] = pd.to_numeric(data["volume"], errors="coerce").fillna(0.0)

    previous_close = data["close"].shift()
    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["atr"] = _wilder(true_range, config.atr_period)
    up = data["high"].diff()
    down = -data["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=data.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=data.index)
    plus_di = 100 * _wilder(plus_dm, 14) / data["atr"].replace(0, np.nan)
    minus_di = 100 * _wilder(minus_dm, 14) / data["atr"].replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    data["adx"] = _wilder(dx, 14)

    typical = (data["high"] + data["low"] + data["close"]) / 3
    data["cum_volume"] = data.groupby("session")["volume"].cumsum()
    data["cum_value"] = (typical * data["volume"]).groupby(data["session"]).cumsum()
    data["vwap"] = data["cum_value"] / data["cum_volume"].replace(0, np.nan)
    # Index volume can be zero. A causal expanding typical-price mean is an
    # explicit fallback, not a claim that index volume exists.
    fallback = typical.groupby(data["session"]).expanding().mean().reset_index(level=0, drop=True)
    data["vwap"] = data["vwap"].fillna(fallback)
    data["vwap_slope_5"] = data["vwap"] - data["vwap"].shift(5)
    above_vwap = data["close"] > data["vwap"]
    data["vwap_cross"] = above_vwap.ne(above_vwap.shift()).astype(int)
    data["vwap_crosses_15m"] = data["vwap_cross"].rolling(15).sum()
    data["rolling_high_20"] = data["high"].shift(1).rolling(20).max()
    data["rolling_low_20"] = data["low"].shift(1).rolling(20).min()
    data["micro_high_5"] = data["high"].shift(1).rolling(5).max()
    data["micro_low_5"] = data["low"].shift(1).rolling(5).min()
    data["volume_average_20"] = data["volume"].shift(1).rolling(20).mean()
    data["volume_ratio_1m"] = data["volume"] / data["volume_average_20"].replace(0, np.nan)

    five = _completed_timeframe(data, "5min", "5m")
    fifteen = _completed_timeframe(data, "15min", "15m")
    for context in (five, fifteen):
        data = pd.merge_asof(
            data.sort_values("timestamp"),
            context.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
    return data.reset_index(drop=True)


def _rejection(row: pd.Series, direction: int) -> bool:
    candle_range = max(float(row["high"] - row["low"]), 1e-9)
    body = abs(float(row["close"] - row["open"]))
    lower_wick = float(min(row["open"], row["close"]) - row["low"])
    upper_wick = float(row["high"] - max(row["open"], row["close"]))
    if direction == 1:
        return row["close"] > row["open"] and lower_wick >= max(body, 0.35 * candle_range)
    return row["close"] < row["open"] and upper_wick >= max(body, 0.35 * candle_range)


def generate_candidates(
    data: pd.DataFrame,
    selected: tuple[str, ...] | list[str] = STRATEGIES,
    config: IntradayConfig = IntradayConfig(),
) -> pd.DataFrame:
    """Return objective candidates. Entries occur at the next 1m bar open."""
    unknown = set(selected).difference(STRATEGIES)
    if unknown:
        raise ValueError(f"Unknown strategies: {sorted(unknown)}")
    rows: list[dict] = []
    orb_state: dict[object, dict] = {}
    for i in range(2, len(data) - 1):
        row, prev = data.iloc[i], data.iloc[i - 1]
        if pd.isna(row["atr"]) or row["atr"] <= 0:
            continue
        clock = row["timestamp"].strftime("%H:%M")
        if not config.entry_start <= clock <= config.entry_end:
            continue
        session_data = data[data["session"] == row["session"]]
        opening = session_data[
            (session_data["timestamp"].dt.strftime("%H:%M") >= "09:15")
            & (session_data["timestamp"].dt.strftime("%H:%M") < "09:30")
        ]

        signals: list[tuple[str, int, float]] = []
        ema_gap_5m_atr = abs(row["ema9_5m"] - row["ema20_5m"]) / row["atr"]
        ema_gap_15m_atr = abs(row["ema9_15m"] - row["ema20_15m"]) / row["atr"]
        orderly_trend = (
            row["adx"] >= config.trend_adx_minimum
            and row["vwap_crosses_15m"] <= config.maximum_vwap_crosses_15m
            and ema_gap_5m_atr >= config.minimum_ema_gap_atr
            and ema_gap_15m_atr >= config.minimum_ema_gap_atr
        )
        if "GOLDEN_RULE" in selected:
            bull15 = (
                row["close_15m"] > row["ema9_15m"] > row["ema20_15m"]
                and row["ema20_slope_15m"] > 0
            )
            bear15 = (
                row["close_15m"] < row["ema9_15m"] < row["ema20_15m"]
                and row["ema20_slope_15m"] < 0
            )
            pullback_range_ok = (
                row["high_5m"] - row["low_5m"]
                <= config.maximum_pullback_range_atr * row["atr"]
            )
            quiet_pullback = (
                row["volume_ratio_5m"] <= config.maximum_pullback_volume_ratio
                and pullback_range_ok
            )
            bull5 = (
                row["ema9_5m"] > row["ema20_5m"]
                and row["low_5m"] <= row["ema9_5m"] + 0.15 * row["atr"]
                and row["close_5m"] >= row["ema20_5m"]
                and row["close"] > row["vwap"]
            )
            bear5 = (
                row["ema9_5m"] < row["ema20_5m"]
                and row["high_5m"] >= row["ema9_5m"] - 0.15 * row["atr"]
                and row["close_5m"] <= row["ema20_5m"]
                and row["close"] < row["vwap"]
            )
            bull_sweep = prev["low"] < prev["micro_low_5"] and prev["close"] > prev["micro_low_5"]
            bear_sweep = prev["high"] > prev["micro_high_5"] and prev["close"] < prev["micro_high_5"]
            higher_low = row["low"] > prev["low"]
            lower_high = row["high"] < prev["high"]
            bull_trigger = row["close"] > prev["high"] and row["close"] > row["open"]
            bear_trigger = row["close"] < prev["low"] and row["close"] < row["open"]
            trigger_volume = (
                row["volume_ratio_1m"] >= config.minimum_trigger_volume_ratio
                and row["volume"] > prev["volume"]
            )
            if (
                orderly_trend and bull15 and bull5 and quiet_pullback
                and (bull_sweep or higher_low) and bull_trigger and trigger_volume
            ):
                signals.append(("GOLDEN_RULE", 1, float(min(prev["low"], row["low"]))))
            if (
                orderly_trend and bear15 and bear5 and quiet_pullback
                and (bear_sweep or lower_high) and bear_trigger and trigger_volume
            ):
                signals.append(("GOLDEN_RULE", -1, float(max(prev["high"], row["high"]))))

        if "ORB_RETEST" in selected and len(opening) == 15:
            high, low = float(opening["high"].max()), float(opening["low"].min())
            state = orb_state.setdefault(row["session"], {"direction": 0, "bar": -1})
            orb_bull_filter = (
                orderly_trend
                and row["ema9_5m"] > row["ema20_5m"]
                and row["close"] > row["vwap"]
            )
            orb_bear_filter = (
                orderly_trend
                and row["ema9_5m"] < row["ema20_5m"]
                and row["close"] < row["vwap"]
            )
            if state["direction"] == 0:
                if orb_bull_filter and prev["close"] <= high < row["close"]:
                    state.update(direction=1, bar=i, level=high)
                elif orb_bear_filter and prev["close"] >= low > row["close"]:
                    state.update(direction=-1, bar=i, level=low)
            elif i - state["bar"] <= config.orb_retest_bars:
                tolerance = config.orb_retest_tolerance_atr * row["atr"]
                touched = (
                    row["low"] <= state["level"] + tolerance
                    if state["direction"] == 1
                    else row["high"] >= state["level"] - tolerance
                )
                held = row["close"] > state["level"] if state["direction"] == 1 else row["close"] < state["level"]
                if touched and held and _rejection(row, state["direction"]):
                    stop = float(row["low"] if state["direction"] == 1 else row["high"])
                    signals.append(("ORB_RETEST", state["direction"], stop))
                    state["direction"] = 0
            else:
                state["direction"] = 0

        if "VWAP_MEAN_REVERSION" in selected:
            flat = abs(row["vwap_slope_5"]) <= config.flat_vwap_slope_atr * row["atr"]
            sideways = row["adx"] <= config.sideways_adx_maximum and flat
            distance = (row["close"] - row["vwap"]) / row["atr"]
            at_resistance = row["high"] >= row["rolling_high_20"]
            at_support = row["low"] <= row["rolling_low_20"]
            if sideways and distance >= config.mean_reversion_distance_atr and at_resistance and _rejection(row, -1):
                signals.append(("VWAP_MEAN_REVERSION", -1, float(row["high"])))
            if sideways and distance <= -config.mean_reversion_distance_atr and at_support and _rejection(row, 1):
                signals.append(("VWAP_MEAN_REVERSION", 1, float(row["low"])))

        for strategy, direction, raw_stop in signals:
            entry = float(data.iloc[i + 1]["open"])
            buffered_stop = raw_stop - direction * config.stop_buffer_atr * row["atr"]
            risk = direction * (entry - buffered_stop)
            risk = max(risk, config.minimum_stop_atr * row["atr"])
            if risk > config.maximum_stop_atr * row["atr"]:
                continue
            stop = entry - direction * risk
            if strategy == "VWAP_MEAN_REVERSION":
                target = float(row["vwap"])
            elif strategy == "GOLDEN_RULE":
                target = entry + direction * config.scalp_final_r * risk
            else:
                target = entry + direction * config.target_r * risk
            if direction * (target - entry) <= 0:
                continue
            rows.append(
                {
                    "strategy": strategy,
                    "signal_time": row["timestamp"],
                    "entry_time": data.iloc[i + 1]["timestamp"],
                    "direction": direction,
                    "side": "BUY CE" if direction == 1 else "BUY PE",
                    "entry": entry,
                    "stop": stop,
                    "target": target,
                    "partial_target": (
                        entry + direction * config.scalp_partial_r * risk
                        if strategy == "GOLDEN_RULE" else target
                    ),
                    "risk_points": risk,
                    "atr": float(row["atr"]),
                    "adx": float(row["adx"]),
                    "quality_score": float(
                        np.clip(
                            (
                                45
                                + min(max(float(row["adx"]) - config.trend_adx_minimum, 0), 20)
                                + min(float(row["volume_ratio_1m"]), 2.0) * 10
                                + min(ema_gap_5m_atr + ema_gap_15m_atr, 0.50) * 30
                            )
                            if strategy != "VWAP_MEAN_REVERSION"
                            else (
                                55
                                + min(abs(float(row["close"] - row["vwap"])) / row["atr"], 1.5) * 15
                                + max(config.sideways_adx_maximum - float(row["adx"]), 0)
                            ),
                            0,
                            100,
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def latest_live_candidates(
    completed_features: pd.DataFrame,
    current_price: float,
    now=None,
    selected: tuple[str, ...] | list[str] = STRATEGIES,
    config: IntradayConfig = IntradayConfig(),
) -> pd.DataFrame:
    """Evaluate the latest completed candle with current LTP as paper entry.

    The synthetic execution row is used only to supply the prospective
    next-bar entry price. It is never used to calculate indicators/signals.
    """
    if completed_features.empty:
        return pd.DataFrame()
    current = pd.Timestamp.now(tz=IST) if now is None else pd.Timestamp(now)
    current = (
        current.tz_localize(IST) if current.tzinfo is None
        else current.tz_convert(IST)
    )
    latest = completed_features.iloc[-1]
    if current.date() != latest["timestamp"].date():
        return pd.DataFrame()
    synthetic = latest.copy()
    synthetic["timestamp"] = current.floor("min")
    synthetic["open"] = float(current_price)
    synthetic["high"] = float(current_price)
    synthetic["low"] = float(current_price)
    synthetic["close"] = float(current_price)
    augmented = pd.concat(
        [completed_features, synthetic.to_frame().T],
        ignore_index=True,
    )
    augmented["timestamp"] = pd.to_datetime(augmented["timestamp"])
    candidates = generate_candidates(augmented, selected, config)
    if candidates.empty:
        return candidates
    candidates = candidates[
        candidates["signal_time"].eq(latest["timestamp"])
    ].copy()
    return candidates.sort_values(
        ["quality_score", "strategy"], ascending=[False, True]
    ).reset_index(drop=True)


def backtest_candidates(
    data: pd.DataFrame,
    candidates: pd.DataFrame,
    config: IntradayConfig = IntradayConfig(),
) -> pd.DataFrame:
    """One-position-at-a-time, stop-first, next-open execution backtest."""
    if candidates.empty:
        return pd.DataFrame()
    trades: list[dict] = []
    occupied_until = None
    daily_trades: dict[object, int] = {}
    daily_losses: dict[object, int] = {}
    for candidate in candidates.sort_values(["entry_time", "strategy"]).to_dict("records"):
        session = candidate["entry_time"].date()
        if occupied_until is not None and candidate["entry_time"] <= occupied_until:
            continue
        if daily_trades.get(session, 0) >= config.maximum_trades_per_day:
            continue
        if daily_losses.get(session, 0) >= config.maximum_losses_per_day:
            continue
        hold_minutes = (
            config.scalp_time_stop_minutes
            if candidate["strategy"] == "GOLDEN_RULE"
            else config.maximum_hold_minutes
        )
        path = data[
            (data["timestamp"] >= candidate["entry_time"])
            & (data["timestamp"].dt.date == session)
        ].head(hold_minutes + 1)
        outcome, exit_price, exit_time = "TIME", None, None
        partial_hit = False
        for _, bar in path.iterrows():
            if bar["timestamp"].strftime("%H:%M") >= config.square_off:
                exit_price, exit_time = float(bar["open"]), bar["timestamp"]
                break
            active_stop = candidate["entry"] if partial_hit else candidate["stop"]
            stop_hit = bar["low"] <= active_stop if candidate["direction"] == 1 else bar["high"] >= active_stop
            target_hit = bar["high"] >= candidate["target"] if candidate["direction"] == 1 else bar["low"] <= candidate["target"]
            if stop_hit:
                outcome = "BREAKEVEN" if partial_hit else "LOSS"
                exit_price, exit_time = active_stop, bar["timestamp"]
                break
            if target_hit:
                if candidate["strategy"] == "GOLDEN_RULE":
                    partial_hit = True
                outcome, exit_price, exit_time = "WIN", candidate["target"], bar["timestamp"]
                break
            if candidate["strategy"] == "GOLDEN_RULE" and not partial_hit:
                partial_hit_now = (
                    bar["high"] >= candidate["partial_target"]
                    if candidate["direction"] == 1
                    else bar["low"] <= candidate["partial_target"]
                )
                if partial_hit_now:
                    partial_hit = True
        if exit_price is None:
            last = path.iloc[-1]
            exit_price, exit_time = float(last["close"]), last["timestamp"]
        if candidate["strategy"] == "GOLDEN_RULE" and partial_hit:
            first_half = 0.5 * candidate["direction"] * (
                candidate["partial_target"] - candidate["entry"]
            )
            second_half = 0.5 * candidate["direction"] * (
                exit_price - candidate["entry"]
            )
            gross = first_half + second_half
        else:
            gross = candidate["direction"] * (exit_price - candidate["entry"])
        net = gross - config.cost_points_per_trade
        result = dict(candidate)
        result.update(
            exit_time=exit_time,
            exit_price=exit_price,
            outcome=outcome,
            gross_points=gross,
            net_points=net,
            result_r=net / candidate["risk_points"],
            partial_1r_hit=partial_hit,
        )
        trades.append(result)
        occupied_until = exit_time + pd.Timedelta(minutes=config.cooldown_minutes)
        daily_trades[session] = daily_trades.get(session, 0) + 1
        daily_losses[session] = daily_losses.get(session, 0) + int(outcome == "LOSS")
    return pd.DataFrame(trades)


def strategy_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["strategy", "trades", "win_rate", "net_points", "expectancy_r"])
    return (
        trades.groupby("strategy", as_index=False)
        .agg(
            trades=("strategy", "size"),
            wins=("outcome", lambda value: int((value == "WIN").sum())),
            net_points=("net_points", "sum"),
            expectancy_r=("result_r", "mean"),
        )
        .assign(win_rate=lambda frame: 100 * frame["wins"] / frame["trades"])
        .sort_values("expectancy_r", ascending=False)
        [["strategy", "trades", "wins", "win_rate", "net_points", "expectancy_r"]]
    )
