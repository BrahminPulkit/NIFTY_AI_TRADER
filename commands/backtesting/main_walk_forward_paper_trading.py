"""Run Step 20E without altering any frozen artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.institutional_backtest import CONFIG
from src.walk_forward_paper_trading import (
    ReplayPolicy,
    replay_frozen_stream,
    replay_metrics,
)


STAGE1 = Path("data/setups/stage1_setup_dataset.parquet")
PREDICTIONS = Path("data/prediction/prediction_dataset.parquet")
JOINED = Path("data/research/probability_threshold/joined_probability_dataset.parquet")
STEP20D_METADATA = Path("reports/institutional_backtest/metadata.json")
REPORTS = Path("reports/walk_forward_paper_trading")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _period_statistics(
    trades: pd.DataFrame, daily_curve: pd.DataFrame, frequency: str
) -> pd.DataFrame:
    calendar = daily_curve.copy()
    calendar["period"] = (
        calendar.date.dt.tz_localize(None).dt.to_period(frequency).astype(str)
    )
    equity = calendar.groupby("period", sort=True).agg(
        period_return=("daily_return", lambda values: (1 + values).prod() - 1),
        ending_equity=("equity", "last"),
        maximum_drawdown=("drawdown", "min"),
    ).reset_index()
    if trades.empty:
        equity[["trades", "wins", "gross_pnl", "costs", "net_pnl"]] = 0
        equity[["win_rate", "average_trade"]] = float("nan")
        return equity
    frame = trades.copy()
    frame["period"] = (
        pd.to_datetime(frame.exit_timestamp).dt.tz_localize(None)
        .dt.to_period(frequency).astype(str)
    )
    activity = frame.groupby("period", sort=True).agg(
        trades=("net_pnl", "size"),
        wins=("win", "sum"),
        win_rate=("win", "mean"),
        gross_pnl=("gross_pnl", "sum"),
        costs=("total_costs", "sum"),
        net_pnl=("net_pnl", "sum"),
        average_trade=("net_pnl", "mean"),
    ).reset_index()
    result = equity.merge(activity, on="period", how="left", validate="one_to_one")
    for column in ("trades", "wins", "gross_pnl", "costs", "net_pnl"):
        result[column] = result[column].fillna(0)
    return result


def main() -> None:
    if REPORTS.exists():
        raise FileExistsError(
            f"{REPORTS} already exists; frozen Step 20E output will not be overwritten"
        )
    frozen_paths = (STAGE1, PREDICTIONS, JOINED, STEP20D_METADATA)
    for path in frozen_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    hashes_before = {str(path): sha256(path) for path in frozen_paths}
    backtest_metadata = json.loads(STEP20D_METADATA.read_text(encoding="utf-8"))
    recommendation = backtest_metadata["robust_thresholds"][0]
    policy = ReplayPolicy(
        model=recommendation["model"],
        threshold=float(recommendation["threshold"]),
    )
    if policy.model != "CATBOOST" or policy.threshold != 0.90:
        raise AssertionError("Frozen Step 20D leading recommendation changed")

    stage1 = pd.read_parquet(STAGE1)
    predictions = pd.read_parquet(PREDICTIONS)
    joined = pd.read_parquet(JOINED)
    trades, skipped, events = replay_frozen_stream(
        stage1, predictions, joined, policy, CONFIG
    )
    REPORTS.mkdir(parents=True)
    trades.to_csv(REPORTS / "trade_log.csv", index=False)
    skipped.to_csv(REPORTS / "skipped_trade_log.csv", index=False)

    session_dates = (
        pd.to_datetime(stage1.timestamp, utc=True)
        .dt.tz_convert("Asia/Kolkata").dt.normalize().drop_duplicates()
    )
    event_equity = events.set_index("timestamp").equity
    daily_index = pd.DatetimeIndex(session_dates)
    daily_equity = (
        event_equity.reindex(event_equity.index.union(daily_index))
        .sort_index().ffill().reindex(daily_index)
    )
    daily_curve = pd.DataFrame({
        "date": daily_index,
        "equity": daily_equity.to_numpy(),
    })
    daily_curve["daily_return"] = daily_curve.equity.pct_change().fillna(0.0)
    daily_curve["running_peak"] = daily_curve.equity.cummax()
    daily_curve["drawdown"] = (
        daily_curve.equity / daily_curve.running_peak - 1
    )
    daily_curve.to_csv(REPORTS / "paper_equity_curve.csv", index=False)
    daily_curve[["date", "equity", "running_peak", "drawdown"]].to_csv(
        REPORTS / "drawdown_curve.csv", index=False
    )

    daily = _period_statistics(trades, daily_curve, "D")
    monthly = _period_statistics(trades, daily_curve, "M")
    yearly = _period_statistics(trades, daily_curve, "Y")
    daily.to_csv(REPORTS / "daily_statistics.csv", index=False)
    monthly.to_csv(REPORTS / "monthly_statistics.csv", index=False)
    yearly.to_csv(REPORTS / "yearly_statistics.csv", index=False)

    start = pd.to_datetime(stage1.timestamp.iloc[0])
    end = pd.to_datetime(stage1.timestamp.iloc[-1])
    metrics = replay_metrics(
        trades, CONFIG.initial_capital, start, end,
        daily_curve.daily_return.iloc[1:],
    )
    risk = pd.DataFrame([
        {"metric": key, "value": value} for key, value in metrics.items()
    ])
    risk.to_csv(REPORTS / "risk_statistics.csv", index=False)
    skip_counts = (
        skipped.groupby(["skip_category", "skip_reason"], dropna=False)
        .size().rename("count").reset_index()
    )
    execution = pd.concat([
        pd.DataFrame([{
            "skip_category": "EXECUTED",
            "skip_reason": "COMPLETED",
            "count": len(trades),
        }]),
        skip_counts,
    ], ignore_index=True)
    execution.to_csv(REPORTS / "execution_summary.csv", index=False)

    hashes_after = {str(path): sha256(path) for path in frozen_paths}
    if hashes_before != hashes_after:
        raise AssertionError("A frozen input changed during Step 20E")
    probability_rows = int(
        joined[policy.probability_column].ge(policy.threshold).sum()
    )
    metadata = {
        "step": "20E",
        "research_only": True,
        "retraining": False,
        "probability_regeneration": False,
        "policy": {
            "model": policy.model,
            "threshold": policy.threshold,
            "threshold_source": str(STEP20D_METADATA),
            "exit_rule": "frozen 15-minute Strategy Decision research response",
            "one_active_trade": True,
        },
        "cost_config": CONFIG.as_dict(),
        "frozen_input_sha256_before": hashes_before,
        "frozen_input_sha256_after": hashes_after,
        "validation": {
            "frozen_hashes_unchanged": hashes_before == hashes_after,
            "stage1_rows_replayed": len(stage1),
            "prediction_rows": len(predictions),
            "oof_rows": len(joined),
            "probability_rows_at_or_above_threshold": probability_rows,
            "completed_trades": len(trades),
            "skipped_opportunities": len(skipped),
            "trade_timestamps_unique": bool(
                trades.entry_timestamp.is_unique if len(trades) else True
            ),
            "trades_chronological": bool(
                trades.entry_timestamp.is_monotonic_increasing
                if len(trades) else True
            ),
            "capital_accounting_reconciles": bool(
                abs(
                    CONFIG.initial_capital
                    + (trades.net_pnl.sum() if len(trades) else 0)
                    - (
                        trades.capital_after.iloc[-1]
                        if len(trades) else CONFIG.initial_capital
                    )
                ) < 1e-8
            ),
        },
    }
    (REPORTS / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    report = f"""# Step 20E Walk-Forward Paper-Trading Report

## Frozen replay contract

- Model: {policy.model}
- Frozen threshold: {policy.threshold:.2f}
- Replay candles: {len(stage1):,}
- Frozen OOF rows: {len(joined):,}
- Completed virtual trades: {len(trades):,}
- Skipped setup opportunities: {len(skipped):,}
- Exit response horizon: {policy.exit_minutes} minutes

## Performance after frozen Step 20D costs

- Net return: {metrics['net_return']:.2%}
- CAGR: {metrics['cagr']:.2%}
- Win rate: {metrics['win_rate']:.2%}
- Profit factor: {metrics['profit_factor']:.3f}
- Maximum drawdown: {metrics['maximum_drawdown']:.2%}
- Expectancy: INR {metrics['expectancy']:,.2f} per trade
- Sharpe ratio: {metrics['sharpe_ratio']:.3f}
- Sortino ratio: {metrics['sortino_ratio']:.3f}
- Calmar ratio: {metrics['calmar_ratio']:.3f}

## Leakage controls

The engine traverses the frozen Stage-1 dataset in timestamp order.  It reads
only stored causal Decision Engine fields and stored OOF probabilities at
entry.  The frozen terminal response is not posted to equity until the
15-minute research horizon has elapsed.  No model or probability is called.

## Interpretation limit

This replay inherits the Step 20D limitation: the frozen response contains a
premium return but no executable option contract, quantity, bid/ask quote or
intratrade mark-to-market path.  Equity is therefore realized-equity research,
not a live broker fill reconstruction.
"""
    (REPORTS / "walk_forward_report.md").write_text(report, encoding="utf-8")

    adequate = len(trades) >= 100
    recommendation_text = f"""# Step 20E Recommendation

This remains research-only and is not promoted to production.

- Completed trades: {len(trades)}
- Net return after configured costs: {metrics['net_return']:.2%}
- Maximum drawdown: {metrics['maximum_drawdown']:.2%}
- Minimum 100-trade evidence requirement met: {adequate}

{"The replay passes the preliminary sample-size screen, but still requires executable option-price data."
 if adequate else
 "The replay does not have enough executed trades for a reliable production conclusion."}

Do not begin live or broker-connected paper trading from this result.
"""
    (REPORTS / "recommendation.md").write_text(
        recommendation_text, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
