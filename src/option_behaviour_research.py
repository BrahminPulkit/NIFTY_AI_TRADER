"""Research-only premium behaviour for frozen Stage-1 V2 opportunities."""

from __future__ import annotations

import numpy as np
import pandas as pd


RESEARCH_VERSION = "option_behaviour_h5_v1"
HORIZON = 5
OPTION_TYPE_PROVENANCE = "CALL_FROM_ACQUISITION_NOTEBOOK_NOT_ROW_LEVEL"


def evaluate_option_behaviour(aligned: pd.DataFrame, setups: pd.DataFrame,
                              horizon: int = HORIZON) -> pd.DataFrame:
    if horizon != HORIZON:
        raise ValueError("Frozen research horizon is five candles")
    required_aligned = {"timestamp", "option_open", "option_high", "option_low",
                        "option_close", "option_volume"}
    required_setups = {"timestamp", "entry_signal", "trade_allowed", "trend_state",
                       "minutes_from_open", "session_id"}
    if required_aligned.difference(aligned.columns):
        raise ValueError(f"Missing aligned fields: {sorted(required_aligned.difference(aligned.columns))}")
    if required_setups.difference(setups.columns):
        raise ValueError(f"Missing setup fields: {sorted(required_setups.difference(setups.columns))}")
    data = aligned.copy()
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    setup = setups.copy()
    setup["timestamp"] = pd.to_datetime(setup.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    allowed = setup.loc[setup.trade_allowed.eq(1) & setup.entry_signal.isin([-1, 1]),
                        ["timestamp", "entry_signal", "trend_state", "minutes_from_open", "session_id"]]
    position = pd.Series(np.arange(len(data)), index=data.timestamp)
    records = []
    for row in allowed.itertuples(index=False):
        if row.timestamp not in position.index:
            continue
        entry_index = int(position.at[row.timestamp])
        entry_price = float(data.at[entry_index, "option_close"])
        entry_volume = float(data.at[entry_index, "option_volume"])
        highs, lows, closes, volumes = [], [], [], []
        timestamps = []
        for offset in range(1, horizon + 1):
            future = entry_index + offset
            if future >= len(data):
                break
            future_timestamp = data.at[future, "timestamp"]
            # Exact consecutive minutes only; never jump across a missing bar.
            if future_timestamp.date() != row.timestamp.date() or (
                future_timestamp - row.timestamp).total_seconds() != offset * 60:
                break
            highs.append(float(data.at[future, "option_high"]))
            lows.append(float(data.at[future, "option_low"]))
            closes.append(float(data.at[future, "option_close"]))
            volumes.append(float(data.at[future, "option_volume"]))
            timestamps.append(future_timestamp)
        if not closes or entry_price <= 0:
            continue
        maximum_high, minimum_low = max(highs), min(lows)
        final_return = closes[-1] / entry_price - 1.0
        expansion = maximum_high / entry_price - 1.0
        decay = 1.0 - minimum_low / entry_price
        minute = int(row.minutes_from_open)
        records.append({
            "timestamp": row.timestamp,
            "direction": "LONG" if row.entry_signal == 1 else "SHORT",
            "option_type": "CALL",
            "option_type_provenance": OPTION_TYPE_PROVENANCE,
            "entry_premium": entry_price,
            "entry_volume": entry_volume,
            "premium_expansion": expansion,
            "premium_decay": decay,
            "premium_mfe": expansion,
            "premium_mae": decay,
            "final_premium_return": final_return,
            "future_average_volume": float(np.mean(volumes)),
            "future_maximum_volume": float(np.max(volumes)),
            "volume_change": float(volumes[-1] - entry_volume),
            "volume_ratio": float(volumes[-1] / entry_volume) if entry_volume > 0 else np.nan,
            "holding_candles": len(closes),
            "holding_minutes": int((timestamps[-1] - row.timestamp).total_seconds() / 60),
            "complete_horizon": len(closes) == horizon,
            "year": int(row.timestamp.year),
            "regime": row.trend_state,
            "session_segment": "OPENING" if minute < 60 else "CLOSING" if minute >= 315 else "MID_SESSION",
        })
    return pd.DataFrame(records)


def summarize_behaviour(data: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in data.groupby(groups, sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append({
            **dict(zip(groups, keys)), "observations": len(group),
            "complete_horizon_rate": float(group.complete_horizon.mean()),
            "average_premium_expansion": float(group.premium_expansion.mean()),
            "median_premium_expansion": float(group.premium_expansion.median()),
            "average_premium_decay": float(group.premium_decay.mean()),
            "median_premium_decay": float(group.premium_decay.median()),
            "average_final_return": float(group.final_premium_return.mean()),
            "median_final_return": float(group.final_premium_return.median()),
            "positive_return_rate": float(group.final_premium_return.gt(0).mean()),
            "average_holding_candles": float(group.holding_candles.mean()),
            "average_volume_ratio": float(group.volume_ratio.mean()),
        })
    return pd.DataFrame(rows)
