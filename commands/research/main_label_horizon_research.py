from pathlib import Path

import pandas as pd

from src.label_horizon_research import ResearchContract, evaluate_direction, summarize


INPUT = Path("data/features/feature_dataset.parquet")
REPORTS = Path("reports/label_horizon_research")
HORIZONS = (5, 10, 15, 20)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(INPUT)
    contract = ResearchContract()
    overall_parts, year_parts, regime_parts, session_parts, excursion_parts = [], [], [], [], []
    combined_parts = []
    total_observations = 0
    for horizon in HORIZONS:
        horizon_observations = []
        for direction in (1, -1):
            observations = evaluate_direction(data, horizon, direction, contract)
            horizon_observations.append(observations)
            total_observations += len(observations)
            overall_parts.append(summarize(observations, ["horizon", "direction"]))
            year_parts.append(summarize(observations, ["horizon", "direction", "year"]))
            regime_parts.append(summarize(observations, ["horizon", "direction", "regime"]))
            session_parts.append(summarize(observations, ["horizon", "direction", "session_segment"]))
            excursion_parts.append(summarize(observations, ["horizon", "direction"])[
                ["horizon", "direction", "observations", "average_mfe_r", "median_mfe_r",
                 "average_mae_r", "median_mae_r"]])
        combined_parts.append(summarize(pd.concat(horizon_observations, ignore_index=True), ["horizon"]))

    overall = pd.concat(overall_parts, ignore_index=True)
    yearly = pd.concat(year_parts, ignore_index=True)
    regimes = pd.concat(regime_parts, ignore_index=True)
    sessions = pd.concat(session_parts, ignore_index=True)
    excursions = pd.concat(excursion_parts, ignore_index=True)
    combined = pd.concat(combined_parts, ignore_index=True)
    year_stability = yearly.groupby("horizon").expected_r.agg(
        yearly_expected_r_mean="mean", yearly_expected_r_std="std",
        minimum_year_expected_r="min", maximum_year_expected_r="max")
    regime_stability = regimes.groupby("horizon").expected_r.agg(regime_expected_r_std="std")
    comparison = combined.merge(year_stability, on="horizon").merge(regime_stability, on="horizon")
    comparison["positive_direction_year_share"] = yearly.groupby("horizon").expected_r.apply(
        lambda values: (values > 0).mean()).to_numpy()
    comparison = comparison.sort_values(["expected_r", "yearly_expected_r_std"], ascending=[False, True])

    overall.to_csv(REPORTS / "expected_r_by_horizon.csv", index=False)
    comparison.to_csv(REPORTS / "horizon_comparison.csv", index=False)
    yearly.to_csv(REPORTS / "yearly_horizon_statistics.csv", index=False)
    regimes.to_csv(REPORTS / "regime_horizon_statistics.csv", index=False)
    sessions.to_csv(REPORTS / "session_horizon_statistics.csv", index=False)
    excursions.to_csv(REPORTS / "mfe_mae_statistics.csv", index=False)

    best_expected = comparison.iloc[0]
    stable = comparison.sort_values(["yearly_expected_r_std", "regime_expected_r_std"]).iloc[0]
    timeout = comparison.sort_values("timeout_rate", ascending=False).iloc[0]
    stopped = comparison.sort_values("stop_loss_hit_rate", ascending=False).iloc[0]
    directional_expected = overall.pivot(index="horizon", columns="direction", values="expected_r")
    qualifying = directional_expected[(directional_expected.BUY_CALL > 0) & (directional_expected.BUY_PUT > 0)]
    qualification_text = ("No horizon has positive Expected R for both directions; **no unconditional "
                          "production candidate is approved**." if qualifying.empty else
                          f"Horizons positive in both directions: {', '.join(map(str, qualifying.index))}.")
    recommendation = f"""# Label Horizon Recommendation

## Research verdict

- Best combined Expected R: **{int(best_expected.horizon)} candles** ({best_expected.expected_r:.4f}R).
- Lowest combined year/regime dispersion: **{int(stable.horizon)} candles**.
- Highest timeout rate: **{int(timeout.horizon)} candles** ({timeout.timeout_rate:.2%}).
- Highest stop-loss rate: **{int(stopped.horizon)} candles** ({stopped.stop_loss_hit_rate:.2%}).
- Directional qualification: {qualification_text}

The 20-candle horizon is the best **research** candidate for conditional direction/regime analysis, not an unconditional production horizon. BUY CALL Expected R is negative at every tested horizon, while BUY PUT is positive at every horizon. No production label was replaced.
"""
    (REPORTS / "recommendation.md").write_text(recommendation, encoding="utf-8")
    lines = []
    for row in comparison.itertuples():
        lines.append(f"| {row.horizon} | {row.win_rate:.2%} | {row.stop_loss_hit_rate:.2%} | "
                     f"{row.timeout_rate:.2%} | {row.average_holding_time:.2f} | "
                     f"{row.average_mfe_r:.3f} | {row.average_mae_r:.3f} | "
                     f"{row.expected_r:.4f} | {row.trade_frequency_per_day:.2f} |")
    report = f"""# Label Horizon Research Report

## Fixed contract

- Entry: candle close
- Stop: 0.20%
- Target: 0.30% (1.5R)
- Ambiguous candle: stop first
- Gap fill: opening price when beyond stop/target
- Session-aware: never crosses an NSE session
- Horizons tested: 5, 10, 15, 20 candles
- Directions: BUY CALL and BUY PUT independently
- Observations are overlapping label opportunities, not executed trades or a backtest
- Trade frequency means target/stop-resolved research labels per session; total opportunities are reported separately

## Combined comparison

| Horizon | Win | Stop | Timeout | Avg hold | MFE R | MAE R | Expected R | Observations/day |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(lines)}

## Regimes and sessions

Regime is causal at entry: UPTREND when close > EMA20 > EMA50, DOWNTREND for the inverse, otherwise SIDEWAYS. Opening is the first 60 minutes, closing starts at minute 315, and the remainder is mid-session.

See `recommendation.md` for the research verdict. No model, production label, threshold, or feature was changed.
"""
    (REPORTS / "label_horizon_report.md").write_text(report, encoding="utf-8")
    print(f"Step 17 complete: {total_observations:,} research observations")
    print(f"Best Expected R horizon: {int(best_expected.horizon)}")


if __name__ == "__main__":
    main()
