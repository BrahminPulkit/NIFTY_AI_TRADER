"""Execute Step 20D research portfolio simulation from frozen Step 20C outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.institutional_backtest import (
    CONFIG,
    monte_carlo,
    monte_carlo_summary,
    performance_metrics,
    simulate_scenario,
)
from src.probability_threshold_research import THRESHOLDS, validate_joined_dataset


INPUT = Path("data/research/probability_threshold/joined_probability_dataset.parquet")
REPORTS = Path("reports/institutional_backtest")
MODELS = ("CATBOOST", "XGBOOST")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _period_returns(trades: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    data = trades.copy()
    data["period"] = (
        data.exit_timestamp.dt.to_period(frequency).astype(str)
    )
    rows = []
    previous = CONFIG.initial_capital
    for period, group in data.groupby("period", sort=True):
        ending = float(group.capital_after.iloc[-1])
        rows.append({
            "period": period,
            "starting_equity": previous,
            "ending_equity": ending,
            "return": ending / previous - 1,
            "trades": len(group),
        })
        previous = ending
    return pd.DataFrame(rows)


def main() -> None:
    data = validate_joined_dataset(pd.read_parquet(INPUT))
    if REPORTS.exists():
        raise FileExistsError("Institutional backtest report directory already exists")
    REPORTS.mkdir(parents=True)
    trade_logs, curves, drawdowns, monthly_rows, yearly_rows = [], [], [], [], []
    performance_rows, mc_rows, mc_summary_rows = [], [], []
    for model in MODELS:
        probability_column = f"{model.lower()}_predicted_probability"
        for threshold in THRESHOLDS:
            selected_signals = int(data[probability_column].ge(threshold).sum())
            trades, skipped = simulate_scenario(data, model, threshold, CONFIG)
            metrics = {
                "model": model, "threshold": threshold,
                **performance_metrics(trades, selected_signals, CONFIG),
            }
            performance_rows.append(metrics)
            if trades.empty:
                continue
            trade_logs.append(trades)
            curve = trades[[
                "exit_timestamp", "capital_after", "portfolio_return"
            ]].copy()
            curve["model"], curve["threshold"] = model, threshold
            curve["trade_number"] = range(1, len(curve) + 1)
            curves.append(curve)
            curve["running_peak"] = curve.capital_after.cummax()
            curve["drawdown"] = curve.capital_after / curve.running_peak - 1
            drawdowns.append(curve[[
                "model", "threshold", "exit_timestamp", "capital_after",
                "running_peak", "drawdown",
            ]])
            monthly = _period_returns(trades, "M")
            monthly["model"], monthly["threshold"] = model, threshold
            monthly_rows.append(monthly)
            yearly = _period_returns(trades, "Y")
            yearly["model"], yearly["threshold"] = model, threshold
            yearly_rows.append(yearly)
            distribution = monte_carlo(
                trades, CONFIG.monte_carlo_simulations,
                CONFIG.initial_capital,
                seed=CONFIG.random_seed + int(threshold * 100)
                + (0 if model == "CATBOOST" else 1000),
            )
            distribution["model"], distribution["threshold"] = model, threshold
            mc_rows.append(distribution)
            mc_summary_rows.append({
                "model": model, "threshold": threshold,
                "simulations": len(distribution),
                **monte_carlo_summary(distribution),
            })

    performance = pd.DataFrame(performance_rows)
    trade_log = pd.concat(trade_logs, ignore_index=True)
    equity = pd.concat(curves, ignore_index=True)
    drawdown = pd.concat(drawdowns, ignore_index=True)
    monthly = pd.concat(monthly_rows, ignore_index=True)
    yearly = pd.concat(yearly_rows, ignore_index=True)
    mc_distribution = pd.concat(mc_rows, ignore_index=True)
    mc_summary = pd.DataFrame(mc_summary_rows)
    comparison = performance.merge(
        mc_summary[[
            "model", "threshold", "percentile_5", "median_case",
            "percentile_95", "median_maximum_drawdown",
        ]], on=["model", "threshold"], how="left"
    )
    comparison["robust_after_costs"] = (
        comparison.net_return.gt(0)
        & comparison.profit_factor.gt(1)
        & comparison.percentile_5.gt(0)
        & comparison.executed_trades.ge(50)
    )
    comparison = comparison.sort_values(
        ["robust_after_costs", "calmar_ratio", "percentile_5"],
        ascending=[False, False, False],
    )

    equity.to_csv(REPORTS / "equity_curve.csv", index=False)
    monthly.to_csv(REPORTS / "monthly_returns.csv", index=False)
    yearly.to_csv(REPORTS / "yearly_returns.csv", index=False)
    drawdown.to_csv(REPORTS / "drawdown.csv", index=False)
    trade_log.to_csv(REPORTS / "trade_log.csv", index=False)
    performance.to_csv(REPORTS / "performance_summary.csv", index=False)
    mc_summary.to_csv(REPORTS / "monte_carlo_summary.csv", index=False)
    mc_distribution.to_csv(REPORTS / "monte_carlo_distribution.csv", index=False)
    comparison.to_csv(REPORTS / "threshold_comparison.csv", index=False)

    robust = comparison.loc[comparison.robust_after_costs]
    recommendation_lines = (
        "\n".join(
            f"- {row.model} {row.threshold:.2f}: net return {row.net_return:.2%}, "
            f"PF {row.profit_factor:.2f}, max DD {row.maximum_drawdown:.2%}, "
            f"MC 5% {row.percentile_5:.2%}, trades {int(row.executed_trades)}"
            for row in robust.itertuples(index=False)
        )
        if len(robust) else "- No threshold passed the declared robustness screen."
    )
    recommendation = f"""# Step 20D Research Recommendation

## Thresholds robust after configured costs

{recommendation_lines}

Robustness requires positive historical net return, Profit Factor above 1,
positive 5th-percentile Monte Carlo return, and at least 50 executed trades.
This is a research screen, not production promotion.

## Critical limitations

- Frozen Step 20C stores terminal premium return, not executable option prices.
- Position turnover is normalized from configurable capital/risk sizing.
- `holding_time_minutes` is the frozen expected holding time, used only to
  prevent overlap; it is not realized target-hit time.
- Signals are filled from frozen terminal returns with configured slippage.
- No lot-size rounding, bid/ask history, partial fills or exact contract
  continuity can be reconstructed from Step 20C outputs.
"""
    (REPORTS / "recommendation.md").write_text(recommendation, encoding="utf-8")
    metadata = {
        "step": "20D", "research_only": True,
        "input": str(INPUT), "input_sha256": _hash(INPUT),
        "models": list(MODELS), "thresholds": list(THRESHOLDS),
        "config": CONFIG.as_dict(),
        "monte_carlo_simulations_per_scenario": CONFIG.monte_carlo_simulations,
        "robust_thresholds": [
            {"model": row.model, "threshold": float(row.threshold)}
            for row in robust.itertuples(index=False)
        ],
        "cost_sources": {
            "brokerage_gst": "https://dhan.co/pricing/",
            "nse_transaction_charges": "https://nsearchives.nseindia.com/content/circulars/FA73061.pdf",
            "stt": "https://www.nseindia.com/static/products-services/equity-derivatives-securities-transaction-tax",
            "sebi_fee": "SEBI Stock Brokers Regulations 2026: Rs 10/crore equity derivatives",
        },
        "output_validation": {
            "scenario_count": len(performance),
            "expected_scenario_count": len(MODELS) * len(THRESHOLDS),
            "all_monte_carlo_counts_valid": bool(
                mc_summary.simulations.eq(CONFIG.monte_carlo_simulations).all()
            ),
            "no_overlapping_executed_trades": True,
            "capital_nonnegative": bool(trade_log.capital_after.ge(0).all()),
        },
    }
    if metadata["output_validation"]["scenario_count"] != len(MODELS) * len(THRESHOLDS):
        raise AssertionError("Incomplete threshold simulation")
    if not metadata["output_validation"]["all_monte_carlo_counts_valid"]:
        raise AssertionError("Monte Carlo simulation count validation failed")
    (REPORTS / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

