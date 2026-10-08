"""Run Step 18C research on frozen allowed Stage-1 setups."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.option_behaviour_research import (
    HORIZON, OPTION_TYPE_PROVENANCE, RESEARCH_VERSION,
    evaluate_option_behaviour, summarize_behaviour,
)


def main() -> None:
    reports = Path("reports/option_behaviour"); reports.mkdir(parents=True, exist_ok=True)
    aligned = pd.read_parquet("data/aligned/canonical_index_option_aligned.parquet")
    setups = pd.read_parquet("data/setups/stage1_setup_dataset.parquet")
    observations = evaluate_option_behaviour(aligned, setups)
    observations.to_parquet(reports / "option_setup_observations.parquet", index=False)
    overall = summarize_behaviour(observations, ["option_type", "direction"])
    yearly = summarize_behaviour(observations, ["year", "direction"])
    session = summarize_behaviour(observations, ["session_segment", "direction"])
    regime = summarize_behaviour(observations, ["regime", "direction"])
    overall.to_csv(reports / "premium_expansion_statistics.csv", index=False)
    overall.to_csv(reports / "premium_decay_statistics.csv", index=False)
    observations.groupby(["direction"], sort=True).agg(
        observations=("timestamp", "size"), average_mfe=("premium_mfe", "mean"),
        median_mfe=("premium_mfe", "median"), average_mae=("premium_mae", "mean"),
        median_mae=("premium_mae", "median")).reset_index().to_csv(
            reports / "mfe_mae_option_statistics.csv", index=False)
    observations.groupby(["direction"], sort=True).agg(
        observations=("timestamp", "size"), average_holding_candles=("holding_candles", "mean"),
        median_holding_candles=("holding_candles", "median"),
        complete_horizon_rate=("complete_horizon", "mean")).reset_index().to_csv(
            reports / "holding_time_statistics.csv", index=False)
    yearly.to_csv(reports / "yearly_option_statistics.csv", index=False)
    session.to_csv(reports / "session_option_statistics.csv", index=False)
    regime.to_csv(reports / "regime_option_statistics.csv", index=False)
    call = {
        "observations": len(observations),
        "average_return": float(observations.final_premium_return.mean()),
        "positive_return_rate": float(observations.final_premium_return.gt(0).mean()),
        "average_expansion": float(observations.premium_expansion.mean()),
        "average_decay": float(observations.premium_decay.mean()),
    }
    comparison = pd.DataFrame([
        {"option_type":"CALL", "status":"AVAILABLE", **call},
        {"option_type":"PUT", "status":"UNAVAILABLE_NO_PUT_SERIES", "observations":0,
         "average_return":np.nan, "positive_return_rate":np.nan,
         "average_expansion":np.nan, "average_decay":np.nan},
    ])
    comparison.to_csv(reports / "call_put_comparison.csv", index=False)
    observations.groupby("direction", sort=True).agg(
        observations=("timestamp", "size"), entry_average_volume=("entry_volume", "mean"),
        future_average_volume=("future_average_volume", "mean"),
        future_maximum_volume=("future_maximum_volume", "mean"),
        average_volume_change=("volume_change", "mean"),
        average_volume_ratio=("volume_ratio", "mean")).reset_index().to_csv(
            reports / "volume_behaviour_statistics.csv", index=False)
    distribution = observations.groupby("direction").final_premium_return.quantile(
        [0.01, .05, .25, .50, .75, .95, .99]).rename("premium_return").reset_index().rename(
            columns={"level_1": "quantile"})
    distribution.to_csv(reports / "premium_return_distribution.csv", index=False)
    allowed = int(setups.trade_allowed.eq(1).sum())
    metadata = {"research_version":RESEARCH_VERSION, "horizon":HORIZON,
                "allowed_setups":allowed, "aligned_entries":len(observations),
                "unaligned_entries":allowed-len(observations),
                "option_type_provenance":OPTION_TYPE_PROVENANCE,
                "put_available":False, "models_trained":False, "thresholds_tuned":False}
    (reports / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    avg = observations.final_premium_return.mean()
    report = f"""# Option Behaviour Research

- Frozen allowed setups: **{allowed:,}**; aligned research observations: **{len(observations):,}**
- Five-candle complete-path rate: **{observations.complete_horizon.mean():.2%}**
- Average/median CALL premium return: **{avg:.4%} / {observations.final_premium_return.median():.4%}**
- Average favourable/adverse premium excursion: **{observations.premium_mfe.mean():.4%} / {observations.premium_mae.mean():.4%}**
- Positive final premium return: **{observations.final_premium_return.gt(0).mean():.2%}**

The acquisition notebook proves this is a rolling CALL series. No row-level option type exists and no PUT series is available. Therefore PUT behaviour and a genuine CALL-versus-PUT comparison cannot be measured.

This descriptive research does not establish incremental predictive information beyond Index features; that requires a later leakage-safe comparative out-of-sample model study.
"""
    (reports / "option_behaviour_report.md").write_text(report, encoding="utf-8")
    (reports / "recommendation.md").write_text(
        "# Recommendation\n\nDo not use this dataset as symmetric CALL/PUT confirmation. Preserve it as CALL-only research. Step 19 should remain paused until the VWAP/target-policy blocker is resolved and PUT history is available or the intended model scope is explicitly CALL-only.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
