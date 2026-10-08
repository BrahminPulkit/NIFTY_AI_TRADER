"""Exploratory rule-only scalping validation on verified exact contracts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_pipeline import build_features
from src.paper_trading_engine import PaperTradingConfig, trading_costs
from src.scalping_candidate_engine import STRATEGIES, generate_scalping_candidates


SESSION = "2026-08-07"
TARGET_PCT = 0.05
STOP_PCT = 0.03
MAX_HOLD_MINUTES = 30
QUANTITY = 65
VALIDATED = ROOT / "data/normalized/historical_options/validated_one_day"
REPORTS = ROOT / "reports/phase5/standalone_strategy_validation"


def load_index() -> pd.DataFrame:
    path = VALIDATED / "nifty_index.parquet"
    if path.exists():
        return pd.read_parquet(path)
    payload = requests.get("http://127.0.0.1:8000/api/v1/market", timeout=30).json()
    frame = pd.DataFrame(payload["candles"])
    frame["timestamp"] = pd.to_datetime(
        frame.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    frame = frame.loc[frame.timestamp.dt.strftime("%Y-%m-%d").eq(SESSION)].copy()
    if len(frame) != 375:
        raise RuntimeError(f"Verified NIFTY session requires 375 candles; received {len(frame)}")
    frame.to_parquet(path, index=False)
    return frame


def load_options() -> pd.DataFrame:
    frame = pd.read_parquet(VALIDATED / "ce_pe.parquet")
    if "contract_segment_id" not in frame and "segment_id" in frame:
        frame = frame.rename(columns={"segment_id": "contract_segment_id"})
    frame["timestamp"] = pd.to_datetime(
        frame.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    return frame.sort_values(["option_type", "timestamp"], kind="stable")


def simulate(candidates: pd.DataFrame, options: pd.DataFrame) -> pd.DataFrame:
    config = PaperTradingConfig(quantity=QUANTITY)
    by_side = {
        side: group.set_index("timestamp", drop=False).sort_index()
        for side, group in options.groupby("option_type", sort=False)
    }
    trades = []
    for candidate in candidates.itertuples(index=False):
        side = candidate.option_type
        data = by_side.get(side)
        timestamp = pd.Timestamp(candidate.timestamp)
        if data is None or timestamp not in data.index:
            continue
        entry = data.loc[timestamp]
        if isinstance(entry, pd.DataFrame):
            continue
        future = data.loc[
            data.contract_segment_id.eq(entry.contract_segment_id)
            & data.timestamp.gt(timestamp)
            & data.timestamp.le(timestamp + pd.Timedelta(minutes=MAX_HOLD_MINUTES))]
        if future.empty:
            continue
        entry_price = float(entry.close)
        target = entry_price * (1 + TARGET_PCT)
        stop = entry_price * (1 - STOP_PCT)
        outcome = "TIMEOUT"
        exit_price = float(future.close.iloc[-1])
        exit_time = pd.Timestamp(future.timestamp.iloc[-1])
        for candle in future.itertuples(index=False):
            target_hit = float(candle.high) >= target
            stop_hit = float(candle.low) <= stop
            if target_hit and stop_hit:
                outcome, exit_price, exit_time = "AMBIGUOUS", np.nan, candle.timestamp
                break
            if target_hit:
                outcome, exit_price, exit_time = "WIN", target, candle.timestamp
                break
            if stop_hit:
                outcome, exit_price, exit_time = "LOSS", stop, candle.timestamp
                break
        gross = np.nan if outcome == "AMBIGUOUS" else (exit_price - entry_price) * QUANTITY
        charges = ({"total_costs": np.nan} if outcome == "AMBIGUOUS"
                   else trading_costs(entry_price, exit_price, QUANTITY, config))
        net = np.nan if outcome == "AMBIGUOUS" else gross - charges["total_costs"]
        risk = entry_price * STOP_PCT * QUANTITY
        trades.append({
            "timestamp": timestamp, "strategy": candidate.strategy,
            "option_type": side, "security_id": str(entry.security_id),
            "strike": float(entry.strike), "expiry": str(entry.expiry),
            "contract_segment_id": entry.contract_segment_id,
            "entry_premium": entry_price, "target": target, "stop": stop,
            "exit_timestamp": exit_time, "exit_premium": exit_price,
            "outcome": outcome,
            "holding_minutes": int((pd.Timestamp(exit_time) - timestamp).total_seconds() / 60),
            "gross_pnl": gross, "costs": charges["total_costs"], "net_pnl": net,
            "gross_r": np.nan if risk <= 0 else gross / risk,
            "net_r": np.nan if risk <= 0 else net / risk,
        })
    return pd.DataFrame(trades)


def profit_factor(values: pd.Series) -> float | None:
    values = values.dropna()
    wins, losses = values[values > 0].sum(), -values[values < 0].sum()
    return None if losses == 0 else float(wins / losses)


def max_drawdown(values: pd.Series) -> float:
    equity = values.fillna(0).cumsum()
    return float((equity.cummax() - equity).max()) if len(equity) else 0.0


def summarize(strategy: str, signals: pd.DataFrame, trades: pd.DataFrame) -> dict:
    group = trades.loc[trades.strategy.eq(strategy)].sort_values("timestamp")
    total = len(group)
    executable = group.loc[group.outcome.ne("AMBIGUOUS")]
    return {
        "strategy": strategy, "total_signals": int(signals.strategy.eq(strategy).sum()),
        "valid_trades": int(total), "wins": int(group.outcome.eq("WIN").sum()),
        "losses": int(group.outcome.eq("LOSS").sum()),
        "timeouts": int(group.outcome.eq("TIMEOUT").sum()),
        "ambiguous": int(group.outcome.eq("AMBIGUOUS").sum()),
        "win_rate": float(group.outcome.eq("WIN").mean()) if total else None,
        "loss_rate": float(group.outcome.eq("LOSS").mean()) if total else None,
        "timeout_rate": float(group.outcome.eq("TIMEOUT").mean()) if total else None,
        "ambiguous_rate": float(group.outcome.eq("AMBIGUOUS").mean()) if total else None,
        "before_cost_expectancy": float(executable.gross_pnl.mean()) if len(executable) else None,
        "after_cost_expectancy": float(executable.net_pnl.mean()) if len(executable) else None,
        "average_gross_r": float(executable.gross_r.mean()) if len(executable) else None,
        "average_net_r": float(executable.net_r.mean()) if len(executable) else None,
        "before_cost_profit_factor": profit_factor(executable.gross_pnl),
        "after_cost_profit_factor": profit_factor(executable.net_pnl),
        "average_net_profit": float(executable.loc[executable.net_pnl.gt(0), "net_pnl"].mean())
            if executable.net_pnl.gt(0).any() else None,
        "average_net_loss": float(executable.loc[executable.net_pnl.lt(0), "net_pnl"].mean())
            if executable.net_pnl.lt(0).any() else None,
        "maximum_drawdown": max_drawdown(executable.net_pnl),
        "average_holding_minutes": float(group.holding_minutes.mean()) if total else None,
        "best_trade_after_cost": float(executable.net_pnl.max()) if len(executable) else None,
        "worst_trade_after_cost": float(executable.net_pnl.min()) if len(executable) else None,
        "ce_trades": int(group.option_type.eq("CE").sum()),
        "pe_trades": int(group.option_type.eq("PE").sum()),
    }


def main() -> None:
    index = load_index()
    options = load_options()
    features = build_features(index.reset_index(drop=True))
    candidates = generate_scalping_candidates(features)
    trades = simulate(candidates, options)
    summaries = [summarize(strategy, candidates, trades) for strategy in STRATEGIES]
    usable_dates = sorted(options.timestamp.dt.strftime("%Y-%m-%d").unique().tolist())
    result = {
        "status": "EXPLORATORY_DATA_INSUFFICIENT",
        "conclusion": "C",
        "conclusion_text": (
            "Only one complete exact-contract CE/PE session is usable; expectancy signs are "
            "exploratory and cannot establish strategy profitability."),
        "configuration": {
            "target_pct": TARGET_PCT, "stop_pct": STOP_PCT,
            "maximum_holding_minutes": MAX_HOLD_MINUTES, "quantity": QUANTITY,
            "entry": "candidate candle close; exits start on next candle",
            "cost_model": "paper_trading_engine full brokerage/STT/exchange/SEBI/stamp/GST/slippage",
        },
        "data": {
            "metadata_resolved_dates": 27, "complete_exact_ce_pe_dates": len(usable_dates),
            "usable_dates": usable_dates, "raw_index_candles": len(index),
            "ce_candles": int(options.option_type.eq("CE").sum()),
            "pe_candles": int(options.option_type.eq("PE").sum()),
        },
        "funnel": {
            "raw_candles": len(index), "candidates": len(candidates),
            "valid_trades": len(trades), "wins": int(trades.outcome.eq("WIN").sum()),
            "losses": int(trades.outcome.eq("LOSS").sum()),
            "timeouts": int(trades.outcome.eq("TIMEOUT").sum()),
            "ambiguous": int(trades.outcome.eq("AMBIGUOUS").sum()),
            "after_cost_expectancy": float(trades.net_pnl.dropna().mean()),
        },
        "strategies": summaries,
        "limitations": [
            "One trading session cannot represent regime, expiry, or volatility variation.",
            "Signals are evaluated independently and may overlap; this is setup expectancy, not portfolio P&L.",
            "Historical metadata-resolved dates without exact matching CE and PE candles were excluded.",
        ],
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(REPORTS / "trades.parquet", index=False)
    pd.DataFrame(summaries).to_csv(REPORTS / "strategy_summary.csv", index=False)
    (REPORTS / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
