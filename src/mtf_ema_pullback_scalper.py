"""Causal 15m/5m/1m NIFTY EMA-pullback scalper.

The module contains no broker order-placement code.  A signal formed on a
closed one-minute candle is filled at the next candle's open in backtests.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScalperConfig:
    trend_fast: int = 20
    trend_slow: int = 50
    setup_fast: int = 9
    setup_slow: int = 21
    entry_fast: int = 9
    entry_slow: int = 21
    atr_period: int = 14
    atr_stop_multiplier: float = 1.0
    rr: float = 2.0
    swing_lookback: int = 5
    max_holding_bars: int = 30
    max_trades_per_day: int = 5
    entry_start: str = "09:30"
    entry_end: str = "14:45"
    square_off: str = "15:15"
    slippage_points: float = 0.5
    cost_points: float = 0.5

    def __post_init__(self) -> None:
        if self.rr < 2:
            raise ValueError("Risk:reward must be at least 1:2")


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")
    if pd.api.types.is_numeric_dtype(out["timestamp"]):
        out["timestamp"] = pd.to_datetime(out["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    else:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="raise")
        if out["timestamp"].dt.tz is None:
            out["timestamp"] = out["timestamp"].dt.tz_localize("Asia/Kolkata")
        else:
            out["timestamp"] = out["timestamp"].dt.tz_convert("Asia/Kolkata")
    if "volume" not in out:
        out["volume"] = 0.0
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return (out.dropna(subset=["timestamp", "open", "high", "low", "close"])
            .sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True))


def _closed_timeframe(one_minute: pd.DataFrame, rule: str, prefix: str, fast: int, slow: int) -> pd.DataFrame:
    bars = (one_minute.set_index("timestamp")
            .resample(rule, label="right", closed="left")
            .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                 close=("close", "last"), volume=("volume", "sum")).dropna().reset_index())
    bars[f"{prefix}_ema_fast"] = bars["close"].ewm(span=fast, adjust=False).mean()
    bars[f"{prefix}_ema_slow"] = bars["close"].ewm(span=slow, adjust=False).mean()
    bars[f"{prefix}_trend"] = np.select(
        [(bars[f"{prefix}_ema_fast"] > bars[f"{prefix}_ema_slow"]) &
         (bars[f"{prefix}_ema_fast"].diff() > 0),
         (bars[f"{prefix}_ema_fast"] < bars[f"{prefix}_ema_slow"]) &
         (bars[f"{prefix}_ema_fast"].diff() < 0)], [1, -1], default=0)
    if prefix == "m5":
        bars["m5_pullback_long"] = ((bars["low"] <= bars["m5_ema_fast"]) &
                                     (bars["close"] > bars["m5_ema_fast"]) &
                                     (bars["m5_ema_fast"] > bars["m5_ema_slow"]))
        bars["m5_pullback_short"] = ((bars["high"] >= bars["m5_ema_fast"]) &
                                      (bars["close"] < bars["m5_ema_fast"]) &
                                      (bars["m5_ema_fast"] < bars["m5_ema_slow"]))
    keep = ["timestamp", f"{prefix}_ema_fast", f"{prefix}_ema_slow", f"{prefix}_trend"]
    if prefix == "m5":
        keep += ["m5_pullback_long", "m5_pullback_short"]
    return bars[keep]


def build_features(frame: pd.DataFrame, cfg: ScalperConfig = ScalperConfig()) -> pd.DataFrame:
    data = normalize_ohlcv(frame)
    data["session"] = data["timestamp"].dt.date
    data["ema_fast"] = data["close"].ewm(span=cfg.entry_fast, adjust=False).mean()
    data["ema_slow"] = data["close"].ewm(span=cfg.entry_slow, adjust=False).mean()
    previous = data["close"].shift()
    tr = pd.concat([(data["high"] - data["low"]), (data["high"] - previous).abs(),
                    (data["low"] - previous).abs()], axis=1).max(axis=1)
    data["atr"] = tr.ewm(alpha=1 / cfg.atr_period, adjust=False,
                          min_periods=cfg.atr_period).mean()
    typical = (data["high"] + data["low"] + data["close"]) / 3
    volume = data["volume"].where(data["volume"] > 0, 1.0)
    data["vwap"] = ((typical * volume).groupby(data["session"]).cumsum() /
                    volume.groupby(data["session"]).cumsum())
    for rule, prefix, fast, slow in [
        ("5min", "m5", cfg.setup_fast, cfg.setup_slow),
        ("15min", "m15", cfg.trend_fast, cfg.trend_slow),
    ]:
        higher = _closed_timeframe(data, rule, prefix, fast, slow)
        data = pd.merge_asof(data.sort_values("timestamp"), higher, on="timestamp", direction="backward")
    data["entry_long"] = ((data["close"] > data["ema_fast"]) &
                          (data["close"].shift() <= data["ema_fast"].shift()) &
                          (data["ema_fast"] > data["ema_slow"]) & (data["close"] > data["open"]))
    data["entry_short"] = ((data["close"] < data["ema_fast"]) &
                           (data["close"].shift() >= data["ema_fast"].shift()) &
                           (data["ema_fast"] < data["ema_slow"]) & (data["close"] < data["open"]))
    return data


def option_chain_confirmation(chain: pd.DataFrame | dict | None, direction: int) -> tuple[bool, dict]:
    """Confirm using CE/PE OI and OI change; never guesses when fields are absent."""
    if chain is None:
        return False, {"status": "UNAVAILABLE"}
    if isinstance(chain, dict):
        chain = pd.DataFrame(chain.get("rows", chain.get("data", [])))
    lookup = {str(c).lower(): c for c in chain.columns}
    aliases = {
        "ce_oi": ["ce_oi", "call_oi"], "pe_oi": ["pe_oi", "put_oi"],
        "ce_change": ["ce_oi_change", "ce_change_in_oi", "call_oi_change"],
        "pe_change": ["pe_oi_change", "pe_change_in_oi", "put_oi_change"],
    }
    values: dict[str, float] = {}
    for name, choices in aliases.items():
        column = next((lookup[x] for x in choices if x in lookup), None)
        if column is None:
            return False, {"status": "UNAVAILABLE", "missing": name}
        values[name] = float(pd.to_numeric(chain[column], errors="coerce").fillna(0).sum())
    pcr = values["pe_oi"] / max(values["ce_oi"], 1.0)
    aligned = (values["pe_change"] > values["ce_change"] and pcr >= 0.8
               if direction == 1 else
               values["ce_change"] > values["pe_change"] and pcr <= 1.2)
    return bool(aligned), {"status": "CONFIRMED" if aligned else "REJECTED", "pcr": pcr, **values}


def signal_at(features: pd.DataFrame, index: int, chain: pd.DataFrame | dict | None = None,
              require_option_confirmation: bool = True,
              cfg: ScalperConfig = ScalperConfig()) -> dict | None:
    if index < 50 or index >= len(features):
        return None
    row = features.iloc[index]
    direction = 1 if bool(row["entry_long"]) else -1 if bool(row["entry_short"]) else 0
    if not direction or int(row["m15_trend"]) != direction or int(row["m5_trend"]) != direction:
        return None
    pullback = bool(row["m5_pullback_long"] if direction == 1 else row["m5_pullback_short"])
    vwap_ok = row["close"] > row["vwap"] if direction == 1 else row["close"] < row["vwap"]
    if not pullback or not vwap_ok or pd.isna(row["atr"]):
        return None
    confirmed, chain_details = option_chain_confirmation(chain, direction)
    if require_option_confirmation and not confirmed:
        return None
    history = features.iloc[max(0, index - cfg.swing_lookback + 1):index + 1]
    entry = float(row["close"])
    atr_stop = entry - direction * cfg.atr_stop_multiplier * float(row["atr"])
    swing_stop = float(history["low"].min() if direction == 1 else history["high"].max())
    stop = min(atr_stop, swing_stop) if direction == 1 else max(atr_stop, swing_stop)
    risk = direction * (entry - stop)
    if risk <= 0:
        return None
    return {"signal_time": row["timestamp"], "direction": direction,
            "signal": "BUY CE" if direction == 1 else "BUY PE", "spot_signal_entry": entry,
            "spot_stop": stop, "spot_target": entry + direction * cfg.rr * risk,
            "risk_points": risk, "rr": cfg.rr, "option_chain": chain_details,
            "confirmation_mode": "STRICT" if require_option_confirmation else "DIAGNOSTIC"}


def backtest(frame: pd.DataFrame, historical_chain: pd.DataFrame | None = None,
             require_option_confirmation: bool = False,
             cfg: ScalperConfig = ScalperConfig(),
             include_all_entries: bool = False) -> tuple[pd.DataFrame, dict]:
    """Replay the same closed-candle signals used by live mode.

    ``include_all_entries=False`` preserves the executable one-position-at-a-time
    simulation and daily trade cap.  ``True`` independently replays every
    qualified live-strategy entry, including signals that occur while another
    hypothetical trade would still be open.  This is intended for signal audits,
    not for calculating a deployable portfolio equity curve.
    """
    data = build_features(frame, cfg)
    chain_data = None
    if historical_chain is not None:
        chain_data = historical_chain.copy()
        chain_data["timestamp"] = pd.to_datetime(chain_data["timestamp"])
    trades: list[dict] = []
    blocked_until = -1
    per_day: dict[object, int] = {}
    for i in range(50, len(data) - 1):
        if not include_all_entries and i <= blocked_until:
            continue
        row, nxt = data.iloc[i], data.iloc[i + 1]
        if row["session"] != nxt["session"]:
            continue
        clock = row["timestamp"].strftime("%H:%M")
        if not cfg.entry_start <= clock <= cfg.entry_end:
            continue
        if (not include_all_entries
                and per_day.get(row["session"], 0) >= cfg.max_trades_per_day):
            continue
        snapshot = None
        if chain_data is not None:
            eligible = chain_data[chain_data["timestamp"] <= row["timestamp"]]
            if not eligible.empty:
                last_time = eligible["timestamp"].max()
                snapshot = eligible[eligible["timestamp"] == last_time]
        setup = signal_at(data, i, snapshot, require_option_confirmation, cfg)
        if setup is None:
            continue
        direction = setup["direction"]
        fill = float(nxt["open"]) + direction * cfg.slippage_points
        risk = direction * (fill - setup["spot_stop"])
        if risk <= 0:
            continue
        stop, target = setup["spot_stop"], fill + direction * cfg.rr * risk
        exit_index, exit_price, reason = i + 1, float(nxt["close"]), "TIME"
        end = min(i + 1 + cfg.max_holding_bars, len(data) - 1)
        for j in range(i + 1, end + 1):
            bar = data.iloc[j]
            if bar["session"] != row["session"] or bar["timestamp"].strftime("%H:%M") >= cfg.square_off:
                exit_index, exit_price, reason = j, float(bar["close"]), "EOD"
                break
            stop_hit = bar["low"] <= stop if direction == 1 else bar["high"] >= stop
            target_hit = bar["high"] >= target if direction == 1 else bar["low"] <= target
            if stop_hit:  # conservative when both are inside one OHLC bar
                exit_index, exit_price, reason = j, stop, "SL"
                break
            if target_hit:
                exit_index, exit_price, reason = j, target, "TARGET"
                break
            exit_index, exit_price = j, float(bar["close"])
        net_points = direction * (exit_price - fill) - cfg.cost_points
        trades.append({**setup, "entry_time": nxt["timestamp"], "entry": fill, "stop": stop,
                       "target": target, "exit_time": data.iloc[exit_index]["timestamp"],
                       "exit": exit_price, "exit_reason": reason, "net_points": net_points,
                       "result_r": net_points / risk})
        per_day[row["session"]] = per_day.get(row["session"], 0) + 1
        if not include_all_entries:
            blocked_until = exit_index
    result = pd.DataFrame(trades)
    if result.empty:
        metrics = {"trades": 0, "net_points": 0.0, "win_rate_pct": 0.0,
                   "profit_factor": 0.0, "max_drawdown_points": 0.0}
    else:
        equity = result["net_points"].cumsum()
        wins, losses = result.loc[result.net_points > 0, "net_points"], result.loc[result.net_points < 0, "net_points"]
        metrics = {"trades": len(result), "net_points": float(result.net_points.sum()),
                   "win_rate_pct": float((result.net_points > 0).mean() * 100),
                   "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else float("inf"),
                   "max_drawdown_points": float((equity.cummax() - equity).max())}
    metrics["mode"] = ("STRICT_OPTION_CHAIN" if require_option_confirmation
                       else "SPOT_DIAGNOSTIC_OPTION_CHAIN_NOT_REQUIRED")
    metrics["entry_scope"] = ("ALL_QUALIFIED_LIVE_STRATEGY_ENTRIES"
                              if include_all_entries
                              else "EXECUTABLE_ONE_POSITION_AT_A_TIME")
    return result, metrics
