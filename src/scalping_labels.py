"""Configurable first-touch option-premium labels for scalping research."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from src.option_contract_pipeline import normalize_option_contracts


@dataclass(frozen=True)
class ScalpingLabelConfig:
    target_pct: float = 0.05
    stop_pct: float = 0.03
    maximum_holding_minutes: int = 30
    slippage_bps_each_side: float = 2.0
    brokerage_per_order: float = 20.0
    quantity: int = 1

    def __post_init__(self):
        if self.target_pct <= 0 or self.stop_pct <= 0 or self.maximum_holding_minutes < 1:
            raise ValueError("Target, stop and holding period must be positive")


def build_event_labels(options: pd.DataFrame, entries: pd.DataFrame,
                       config: ScalpingLabelConfig = ScalpingLabelConfig()) -> pd.DataFrame:
    data = normalize_option_contracts(options).set_index("timestamp", drop=False)
    required = {"timestamp", "option_type"}
    if missing := sorted(required.difference(entries.columns)):
        raise ValueError(f"Entries missing required fields: {missing}")
    rows = []
    for candidate in entries.itertuples(index=False):
        timestamp = pd.Timestamp(candidate.timestamp)
        timestamp = timestamp.tz_localize("Asia/Kolkata") if timestamp.tzinfo is None else timestamp.tz_convert("Asia/Kolkata")
        if timestamp not in data.index:
            continue
        matches = data.loc[[timestamp]]
        matches = matches.loc[matches.option_type.eq(str(candidate.option_type).upper())]
        if matches.empty:
            continue
        entry = matches.iloc[0]
        future = data.loc[
            data.contract_segment_id.eq(entry.contract_segment_id)
            & data.timestamp.gt(timestamp)
            & data.timestamp.le(timestamp + pd.Timedelta(minutes=config.maximum_holding_minutes))]
        if future.empty:
            continue
        target, stop = entry.close * (1 + config.target_pct), entry.close * (1 - config.stop_pct)
        outcome, exit_price, exit_time = "TIMEOUT", float(future.close.iloc[-1]), future.timestamp.iloc[-1]
        for candle in future.itertuples():
            target_hit, stop_hit = candle.high >= target, candle.low <= stop
            if target_hit and stop_hit:
                outcome, exit_price, exit_time = "AMBIGUOUS", None, candle.timestamp
                break
            if target_hit:
                outcome, exit_price, exit_time = "WIN", target, candle.timestamp
                break
            if stop_hit:
                outcome, exit_price, exit_time = "LOSS", stop, candle.timestamp
                break
        gross = None if exit_price is None else (float(exit_price) - float(entry.close)) * config.quantity
        turnover = 0 if exit_price is None else (float(entry.close) + float(exit_price)) * config.quantity
        costs = 2 * config.brokerage_per_order + turnover * config.slippage_bps_each_side / 10_000
        net = None if gross is None else gross - costs
        risk = float(entry.close) * config.stop_pct * config.quantity
        rows.append({
            "timestamp": timestamp, "option_type": entry.option_type, "strike": float(entry.strike),
            "expiry": entry.expiry, "contract_segment_id": entry.contract_segment_id,
            "entry_premium": float(entry.close), "outcome": outcome,
            "training_eligible": outcome != "AMBIGUOUS", "exit_timestamp": exit_time,
            "exit_premium": exit_price, "holding_minutes": int((exit_time - timestamp).total_seconds() / 60),
            "gross_pnl": gross, "estimated_costs": costs, "net_pnl": net,
            "realized_r": None if net is None or risk <= 0 else net / risk,
            "label_configuration": asdict(config),
        })
    return pd.DataFrame(rows)
