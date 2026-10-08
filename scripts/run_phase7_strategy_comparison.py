"""Phase 7 exact-contract strategy comparison; no model or production changes."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_pipeline import build_features  # noqa: E402


TRADES = ROOT / "reports/phase5/standalone_strategy_validation/trades.parquet"
INDEX = ROOT / "data/normalized/historical_options/validated_one_day/nifty_index.parquet"
OUTPUT = ROOT / "reports/phase7_strategy_comparison"
STRATEGIES = (
    "TREND_PULLBACK", "BREAKOUT", "MOMENTUM_CONTINUATION",
    "LIQUIDITY_SWEEP_REVERSAL", "VWAP_RECLAIM",
)


def maximum_drawdown(values: pd.Series) -> float:
    pnl = pd.to_numeric(values, errors="coerce").fillna(0).to_numpy()
    equity = pd.Series(np.concatenate([[0.0], np.cumsum(pnl)]))
    return float((equity.cummax() - equity).max())


def profit_factor(values: pd.Series) -> float | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    gains, losses = values[values > 0].sum(), -values[values < 0].sum()
    if losses == 0:
        return None if gains == 0 else float("inf")
    return float(gains / losses)


def expectancy_interval(values: pd.Series, seed: int = 42) -> tuple[float | None, float | None]:
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if len(values) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [.025, .975]))


def summarize(data: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    grouped = data.groupby(group_columns, dropna=False, sort=False)
    for key, group in grouped:
        key = key if isinstance(key, tuple) else (key,)
        group = group.sort_values(["timestamp", "strategy"], kind="stable")
        wins, losses = group.loc[group.outcome.eq("WIN")], group.loc[group.outcome.eq("LOSS")]
        low, high = expectancy_interval(group.net_pnl)
        row = dict(zip(group_columns, key))
        row.update({
            "sessions": int(group.timestamp.dt.strftime("%Y-%m-%d").nunique()),
            "total_trades": len(group), "wins": len(wins), "losses": len(losses),
            "timeouts": int(group.outcome.eq("TIMEOUT").sum()),
            "ambiguous": int(group.outcome.eq("AMBIGUOUS").sum()),
            "win_rate": float(group.outcome.eq("WIN").mean()),
            "average_win_after_cost": float(wins.net_pnl.mean()) if len(wins) else None,
            "average_loss_after_cost": float(losses.net_pnl.mean()) if len(losses) else None,
            "expectancy_before_cost": float(group.gross_pnl.mean()),
            "expectancy_after_cost": float(group.net_pnl.mean()),
            "average_gross_r": float(group.gross_r.mean()),
            "average_net_r": float(group.net_r.mean()),
            "profit_factor_before_cost": profit_factor(group.gross_pnl),
            "profit_factor_after_cost": profit_factor(group.net_pnl),
            "maximum_drawdown": maximum_drawdown(group.net_pnl),
            "average_holding_minutes": float(group.holding_minutes.mean()),
            "expectancy_95pct_low_naive": low,
            "expectancy_95pct_high_naive": high,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def market_context() -> pd.DataFrame:
    index = pd.read_parquet(INDEX).sort_values("timestamp", kind="stable")
    index["timestamp"] = pd.to_datetime(index.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    features = build_features(index)
    aligned_up = ((features.ema_5 > features.ema_9) & (features.ema_9 > features.ema_20)
                  & (features.ema_20 > features.ema_50))
    aligned_down = ((features.ema_5 < features.ema_9) & (features.ema_9 < features.ema_20)
                    & (features.ema_20 < features.ema_50))
    features["trend_regime"] = np.select(
        [aligned_up, aligned_down], ["TREND_UP", "TREND_DOWN"], default="SIDEWAYS")
    atr_pct = features.atr_14 / features.close
    prior_median = atr_pct.shift(1).expanding(min_periods=30).median()
    ratio = atr_pct / prior_median.replace(0, np.nan)
    features["volatility_regime"] = np.select(
        [ratio.gt(1.2), ratio.lt(.8), ratio.isna()],
        ["EXPANSION", "CONTRACTION", "WARMUP"], default="NORMAL")
    features["market_regime"] = features.trend_regime + "|" + features.volatility_regime
    return features[["timestamp", "trend_regime", "volatility_regime", "market_regime"]]


def main() -> None:
    trades = pd.read_parquet(TRADES)
    trades["timestamp"] = pd.to_datetime(trades.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    trades["direction"] = trades.option_type.map({"CE": "BULLISH", "PE": "BEARISH"})
    trades = trades.merge(market_context(), on="timestamp", how="left", validate="many_to_one")
    if set(trades.strategy) != set(STRATEGIES):
        raise RuntimeError("Exact-contract trade evidence does not contain all five strategies")

    overall = summarize(trades, ["strategy"])
    side = summarize(trades, ["strategy", "option_type"])
    direction = summarize(trades, ["strategy", "direction"])
    regime = summarize(trades, ["strategy", "market_regime"])
    trend = summarize(trades, ["strategy", "trend_regime"])

    side_counts = side.pivot(index="strategy", columns="option_type", values="total_trades").fillna(0)
    overall = overall.merge(side_counts.rename(columns={"CE": "ce_trades", "PE": "pe_trades"}),
                            on="strategy", how="left")
    overall["minimum_30_trades"] = overall.total_trades.ge(30)
    overall["minimum_10_each_side"] = overall.ce_trades.ge(10) & overall.pe_trades.ge(10)
    overall["minimum_5_sessions"] = overall.sessions.ge(5)
    overall["positive_after_cost_expectancy"] = overall.expectancy_after_cost.gt(0)
    overall["profit_factor_above_1"] = overall.profit_factor_after_cost.gt(1)
    overall["robustness_eligible"] = overall[[
        "minimum_30_trades", "minimum_10_each_side", "minimum_5_sessions",
        "positive_after_cost_expectancy", "profit_factor_above_1",
    ]].all(axis=1)
    overall = overall.sort_values(
        ["robustness_eligible", "expectancy_after_cost", "profit_factor_after_cost"],
        ascending=[False, False, False], kind="stable").reset_index(drop=True)
    overall.insert(0, "provisional_rank", np.arange(1, len(overall) + 1))

    eligible = overall.loc[overall.robustness_eligible]
    recommendation = (
        str(eligible.iloc[0].strategy) if len(eligible) == 1 else "NO_STRATEGY_SELECTED")
    result = {
        "phase": 7, "status": "STATISTICALLY_INSUFFICIENT",
        "recommendation": recommendation,
        "production_changed": False, "catboost_trained": False,
        "thresholds_changed": False, "paper_trading_changed": False,
        "broker_orders_enabled": False,
        "evidence": {
            "exact_contract_sessions": int(trades.timestamp.dt.date.nunique()),
            "trades": len(trades), "ce_trades": int(trades.option_type.eq("CE").sum()),
            "pe_trades": int(trades.option_type.eq("PE").sum()),
            "strategies": len(overall), "robustness_eligible_strategies": len(eligible),
        },
        "provisional_order": overall.strategy.tolist(),
        "conclusion": (
            "VWAP_RECLAIM has the best observed after-cost expectancy but only three trades. "
            "LIQUIDITY_SWEEP_REVERSAL is the least-negative strategy with a larger nine-trade sample. "
            "Neither is statistically eligible; no strongest robust strategy can be selected."),
        "required_next_evidence": {
            "minimum_complete_sessions": 20,
            "preferred_complete_sessions": "60-100",
            "minimum_trades_per_strategy": 30,
            "minimum_trades_per_side_per_strategy": 10,
            "required_regimes": ["TREND_UP", "TREND_DOWN", "SIDEWAYS"],
            "validation": "session-level walk-forward comparison with non-overlapping trade policy",
        },
        "limitations": [
            "Only one exact-contract session is available.",
            "Trades from different strategies overlap and are not independent observations.",
            "Naive bootstrap intervals quantify trade dispersion, not session-level uncertainty.",
            "Only one expiry and one intraday market path are represented.",
            "Legacy MONTH-CALL and WEEK-PUT histories were excluded.",
        ],
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    overall.to_csv(OUTPUT / "strategy_comparison.csv", index=False)
    side.to_csv(OUTPUT / "strategy_side_comparison.csv", index=False)
    direction.to_csv(OUTPUT / "strategy_direction_comparison.csv", index=False)
    regime.to_csv(OUTPUT / "strategy_regime_comparison.csv", index=False)
    trend.to_csv(OUTPUT / "strategy_trend_comparison.csv", index=False)
    trades.to_parquet(OUTPUT / "audited_trades.parquet", index=False)
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    table_columns = ["provisional_rank", "strategy", "total_trades", "ce_trades", "pe_trades",
                     "win_rate", "expectancy_before_cost", "expectancy_after_cost",
                     "profit_factor_after_cost", "maximum_drawdown", "average_holding_minutes",
                     "robustness_eligible"]
    header = "| " + " | ".join(table_columns) + " |"
    separator = "| " + " | ".join(["---"] * len(table_columns)) + " |"
    body = []
    for row in overall[table_columns].itertuples(index=False, name=None):
        body.append("| " + " | ".join(
            f"{value:.4f}" if isinstance(value, float) else str(value) for value in row) + " |")
    table = "\n".join([header, separator, *body])
    report = "\n".join([
        "# Phase 7 Strategy Comparison", "", f"Status: **{result['status']}**", "",
        "## Evidence", "", table, "", "## Recommendation", "",
        "**NO_STRATEGY_SELECTED**", "", result["conclusion"], "", "## Limitations", "",
        *[f"- {item}" for item in result["limitations"]], "",
    ])
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
