"""Research production horizons only for Stage-1 V2 allowed opportunities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.label_horizon_research import ResearchContract, summarize


def evaluate_allowed_setups(frame: pd.DataFrame, horizon: int,
                            contract: ResearchContract = ResearchContract()) -> pd.DataFrame:
    """Evaluate fixed-contract outcomes for allowed Stage-1 setups only."""
    required = {"timestamp", "open", "high", "low", "close", "entry_signal", "trade_allowed",
                "trend_state", "minutes_from_open", "session_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing setup research columns: {sorted(missing)}")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    data = frame.reset_index(drop=True).copy()
    timestamp = pd.to_datetime(data.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    session = timestamp.dt.strftime("%Y%m%d").to_numpy()
    records: list[dict] = []
    eligible = np.flatnonzero(data.trade_allowed.eq(1).to_numpy() & data.entry_signal.isin([-1, 1]).to_numpy())
    for index in eligible:
        direction = int(data.at[index, "entry_signal"])
        entry = float(data.at[index, "close"])
        risk = entry * contract.stop_pct
        stop = entry - risk if direction == 1 else entry + risk
        target = entry + risk * contract.target_r if direction == 1 else entry - risk * contract.target_r
        outcome = "TIMEOUT"
        exit_price = entry
        holding = 0
        mfe_points = 0.0
        mae_points = 0.0
        for offset in range(1, horizon + 1):
            future = index + offset
            if future >= len(data) or session[future] != session[index]:
                break
            holding = offset
            candle_open = float(data.at[future, "open"])
            high = float(data.at[future, "high"])
            low = float(data.at[future, "low"])
            exit_price = float(data.at[future, "close"])
            if direction == 1:
                mfe_points = max(mfe_points, high - entry)
                mae_points = max(mae_points, entry - low)
                hit_stop, hit_target = low <= stop, high >= target
            else:
                mfe_points = max(mfe_points, entry - low)
                mae_points = max(mae_points, high - entry)
                hit_stop, hit_target = high >= stop, low <= target
            if hit_stop:  # stop-first ambiguity handling
                exit_price = min(candle_open, stop) if direction == 1 else max(candle_open, stop)
                outcome = "STOP"
                break
            if hit_target:
                exit_price = max(candle_open, target) if direction == 1 else min(candle_open, target)
                outcome = "WIN"
                break
        if holding == 0:
            continue
        minute = int(data.at[index, "minutes_from_open"])
        segment = "OPENING" if minute < 60 else "CLOSING" if minute >= 315 else "MID_SESSION"
        records.append({
            "timestamp": timestamp.iat[index], "year": timestamp.iat[index].year,
            "direction": "BUY_CALL" if direction == 1 else "BUY_PUT", "horizon": horizon,
            "outcome": outcome, "holding_candles": holding, "mfe_r": mfe_points/risk,
            "mae_r": mae_points/risk, "realized_r": direction*(exit_price-entry)/risk,
            "regime": data.at[index, "trend_state"], "session_segment": segment,
            "session_id": session[index],
        })
    return pd.DataFrame(records)


def build_research_tables(frame: pd.DataFrame, horizons=(5, 10, 15, 20)) -> dict[str, pd.DataFrame]:
    observations = pd.concat([evaluate_allowed_setups(frame, horizon) for horizon in horizons],
                             ignore_index=True)
    overall = summarize(observations, ["horizon"])
    directional = summarize(observations, ["horizon", "direction"])
    yearly = summarize(observations, ["horizon", "direction", "year"])
    regime = summarize(observations, ["horizon", "direction", "regime"])
    session = summarize(observations, ["horizon", "direction", "session_segment"])
    excursions = directional[["horizon", "direction", "observations", "average_mfe_r",
                              "median_mfe_r", "average_mae_r", "median_mae_r"]]
    stability = yearly.groupby("horizon").expected_r.agg(
        yearly_expected_r_mean="mean", yearly_expected_r_std="std",
        minimum_year_direction_expected_r="min", maximum_year_direction_expected_r="max").reset_index()
    stability = stability.merge(regime.groupby("horizon").expected_r.std().rename(
        "regime_expected_r_std").reset_index(), on="horizon")
    comparison = overall.merge(stability, on="horizon").sort_values("horizon")
    return {"observations": observations, "comparison": comparison, "directional": directional,
            "yearly": yearly, "regime": regime, "session": session, "excursions": excursions}

