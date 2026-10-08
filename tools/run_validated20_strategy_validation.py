"""Research-only standalone comparison on the validated 20-session dataset."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_pipeline import build_features  # noqa: E402
from src.expired_option_dataset import load_selected_contract_panel  # noqa: E402
from src.paper_trading_engine import PaperTradingConfig, trading_costs  # noqa: E402
from src.scalping_candidate_engine import STRATEGIES, generate_scalping_candidates  # noqa: E402
from tools.run_standalone_strategy_validation import (  # noqa: E402
    MAX_HOLD_MINUTES, QUANTITY, STOP_PCT, TARGET_PCT, summarize,
)


SOURCE = ROOT / "data/normalized/historical_options/validated20"
OUTPUT = ROOT / "reports/phase8_strategy_validation"
RAW = ROOT / "data/external/shoonyatrader/validated20_raw"


def simulate_contract_locked(candidates: pd.DataFrame, selected: pd.DataFrame,
                             panel: pd.DataFrame) -> pd.DataFrame:
    """Enter from aligned ATM rows, then evaluate only the immutable entry contract."""
    entry_lookup = selected.set_index(["timestamp", "option_type"], drop=False)
    contract_lookup = {
        key: group.set_index("timestamp", drop=False).sort_index()
        for key, group in panel.groupby(["expiry", "strike", "option_type"], sort=False)
    }
    config = PaperTradingConfig(quantity=QUANTITY)
    trades = []
    for candidate in candidates.itertuples(index=False):
        timestamp = pd.Timestamp(candidate.timestamp)
        key = (timestamp, candidate.option_type)
        if key not in entry_lookup.index:
            continue
        entry = entry_lookup.loc[key]
        if isinstance(entry, pd.DataFrame):
            raise ValueError(f"Multiple selected entry contracts for {key}")
        contract_key = (str(entry.expiry), float(entry.strike), str(entry.option_type))
        contract = contract_lookup.get(contract_key)
        future = (contract.iloc[0:0] if contract is not None else pd.DataFrame())
        if contract is not None:
            future = contract.loc[
                contract.timestamp.gt(timestamp)
                & contract.timestamp.le(timestamp + pd.Timedelta(minutes=MAX_HOLD_MINUTES))]
        entry_price = float(entry.close)
        target, stop = entry_price * (1 + TARGET_PCT), entry_price * (1 - STOP_PCT)
        if future.empty:
            outcome, exit_price, exit_time = "DATA_UNAVAILABLE", np.nan, timestamp
        else:
            outcome, exit_price = "TIMEOUT", float(future.close.iloc[-1])
            exit_time = pd.Timestamp(future.timestamp.iloc[-1])
            for candle in future.itertuples(index=False):
                target_hit, stop_hit = float(candle.high) >= target, float(candle.low) <= stop
                if target_hit and stop_hit:
                    outcome, exit_price, exit_time = "AMBIGUOUS", np.nan, candle.timestamp
                    break
                if target_hit:
                    outcome, exit_price, exit_time = "WIN", target, candle.timestamp
                    break
                if stop_hit:
                    outcome, exit_price, exit_time = "LOSS", stop, candle.timestamp
                    break
        executable = outcome not in {"AMBIGUOUS", "DATA_UNAVAILABLE"}
        gross = (exit_price - entry_price) * QUANTITY if executable else np.nan
        costs = trading_costs(entry_price, exit_price, QUANTITY, config)["total_costs"] \
            if executable else np.nan
        risk = entry_price * STOP_PCT * QUANTITY
        trades.append({
            "timestamp": timestamp, "strategy": candidate.strategy,
            "option_type": candidate.option_type, "security_id": pd.NA,
            "strike": float(entry.strike), "expiry": str(entry.expiry),
            "contract_segment_id": contract.contract_segment_id.iloc[0] if contract is not None else pd.NA,
            "entry_premium": entry_price, "target": target, "stop": stop,
            "exit_timestamp": exit_time, "exit_premium": exit_price, "outcome": outcome,
            "holding_minutes": int((pd.Timestamp(exit_time) - timestamp).total_seconds() / 60),
            "gross_pnl": gross, "costs": costs, "net_pnl": gross - costs if executable else np.nan,
            "gross_r": gross / risk if executable and risk > 0 else np.nan,
            "net_r": (gross - costs) / risk if executable and risk > 0 else np.nan,
            "entry_contract_locked": True,
            "exit_contract_matches_entry": bool(
                contract is not None and contract.expiry.eq(str(entry.expiry)).all()
                and contract.strike.eq(float(entry.strike)).all()
                and contract.option_type.eq(str(entry.option_type)).all()),
        })
    return pd.DataFrame(trades)


def context(index: pd.DataFrame) -> pd.DataFrame:
    features = build_features(index)
    up = ((features.ema_5 > features.ema_9) & (features.ema_9 > features.ema_20)
          & (features.ema_20 > features.ema_50))
    down = ((features.ema_5 < features.ema_9) & (features.ema_9 < features.ema_20)
            & (features.ema_20 < features.ema_50))
    features["trend_regime"] = np.select([up, down], ["TREND_UP", "TREND_DOWN"],
                                          default="SIDEWAYS")
    return features[["timestamp", "trend_regime"]]


def grouped_summary(candidates: pd.DataFrame, trades: pd.DataFrame,
                    group_name: str) -> pd.DataFrame:
    rows = []
    for (value, strategy), group in trades.groupby([group_name, "strategy"], sort=False):
        signals = candidates.loc[
            candidates.strategy.eq(strategy) & candidates[group_name].eq(value)]
        row = summarize(strategy, signals, group)
        row[group_name] = value
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    candidates, trades = [], []
    for session in sorted(path for path in SOURCE.iterdir() if path.is_dir()):
        index = pd.read_parquet(session / "nifty.parquet")
        ce, pe = pd.read_parquet(session / "ce.parquet"), pd.read_parquet(session / "pe.parquet")
        options = pd.concat([ce, pe], ignore_index=True)
        raw_archive = RAW / f"{session.name.replace('-', '')}.zip"
        panel = load_selected_contract_panel(raw_archive, ce, pe)
        session_candidates = generate_scalping_candidates(build_features(index))
        session_candidates = session_candidates.merge(context(index), on="timestamp", how="left")
        session_candidates["session_date"] = session.name
        session_candidates["expiry_group"] = session.name
        session_trades = simulate_contract_locked(session_candidates, options, panel)
        session_trades = session_trades.merge(
            session_candidates[["timestamp", "strategy", "trend_regime", "session_date", "expiry_group"]],
            on=["timestamp", "strategy"], how="left", validate="many_to_one")
        candidates.append(session_candidates)
        trades.append(session_trades)
    candidate_frame = pd.concat(candidates, ignore_index=True)
    trade_frame = pd.concat(trades, ignore_index=True)
    summaries = [summarize(strategy, candidate_frame, trade_frame) for strategy in STRATEGIES]
    side_rows = []
    for strategy in STRATEGIES:
        for side in ("CE", "PE"):
            selected_candidates = candidate_frame.loc[
                candidate_frame.option_type.eq(side)]
            selected_trades = trade_frame.loc[trade_frame.option_type.eq(side)]
            row = summarize(strategy, selected_candidates, selected_trades)
            row["option_type"] = side
            side_rows.append(row)
    session_rows = grouped_summary(candidate_frame, trade_frame, "session_date")
    expiry_rows = grouped_summary(candidate_frame, trade_frame, "expiry_group")
    regime_rows = grouped_summary(candidate_frame, trade_frame, "trend_regime")
    contract_lock_pass = bool(
        trade_frame.entry_contract_locked.all()
        and trade_frame.exit_contract_matches_entry.all()
        and trade_frame.outcome.ne("DATA_UNAVAILABLE").all())
    result = {
        "status": "EXPLORATORY_20_SESSION_COMPARISON",
        "strategy_selected": False,
        "catboost_trained": False,
        "production_changed": False,
        "contract_lock_validation": "PASS" if contract_lock_pass else "FAIL",
        "data": {"sessions": 20, "nifty_rows": 7500, "ce_rows": 7500, "pe_rows": 7500},
        "funnel": {
            "raw_candles": 7500, "candidates": len(candidate_frame),
            "valid_trades": len(trade_frame),
            "wins": int(trade_frame.outcome.eq("WIN").sum()),
            "losses": int(trade_frame.outcome.eq("LOSS").sum()),
            "timeouts": int(trade_frame.outcome.eq("TIMEOUT").sum()),
            "ambiguous": int(trade_frame.outcome.eq("AMBIGUOUS").sum()),
            "data_unavailable": int(trade_frame.outcome.eq("DATA_UNAVAILABLE").sum()),
            "average_holding_minutes": float(trade_frame.holding_minutes.mean()),
            "after_cost_expectancy": float(trade_frame.net_pnl.dropna().mean()),
        },
        "strategies": summaries,
        "limitations": [
            "NIFTY cash-index volume is unavailable, so NIFTY VWAP is null and VWAP_RECLAIM cannot be evaluated.",
            "Sessions are expiry days; conclusions do not yet generalize to all days-to-expiry.",
            "Security IDs are unavailable; this dataset is research-only, not live-execution-ready.",
            "Signals are evaluated independently and can overlap; results are setup expectancy, not portfolio P&L.",
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidate_frame.to_parquet(OUTPUT / "candidates.parquet", index=False)
    trade_frame.to_parquet(OUTPUT / "trades.parquet", index=False)
    pd.DataFrame(summaries).to_csv(OUTPUT / "strategy_summary.csv", index=False)
    pd.DataFrame(side_rows).to_csv(OUTPUT / "strategy_side_summary.csv", index=False)
    session_rows.to_csv(OUTPUT / "strategy_session_summary.csv", index=False)
    expiry_rows.to_csv(OUTPUT / "strategy_expiry_summary.csv", index=False)
    regime_rows.to_csv(OUTPUT / "strategy_regime_summary.csv", index=False)
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
