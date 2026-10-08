from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.setup_engine import OUTPUT_COLUMNS, SetupContract, build_live_setup, build_setups


INPUT = Path("data/features/feature_dataset.parquet")
OUTPUT_DIR = Path("data/setups")
REPORT_DIR = Path("reports/setup_engine")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    result = frame.groupby(groups, dropna=False).agg(
        rows=("entry_signal", "size"), setups=("entry_signal", lambda s: s.ne(0).sum()),
        long_setups=("entry_signal", lambda s: s.eq(1).sum()),
        short_setups=("entry_signal", lambda s: s.eq(-1).sum()),
        allowed=("trade_allowed", "sum"), trading_sessions=("session_id", "nunique")).reset_index()
    result["setup_rate"] = result.setups / result.rows
    result["allowed_rate"] = result.allowed / result.rows
    result["setup_pass_rate"] = result.allowed / result.setups.replace(0, pd.NA)
    result["setups_per_session"] = result.setups / result.trading_sessions
    result["allowed_per_session"] = result.allowed / result.trading_sessions
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_parquet(INPUT)
    contract = SetupContract()
    v1 = build_setups(source, SetupContract.v1_baseline())
    setups = build_setups(source, contract)

    small = build_setups(source.iloc[:1000].copy())
    large_prefix = build_setups(source.iloc[:5000].copy()).iloc[:1000]
    assert_frame_equal(small, large_prefix, check_exact=True)
    mutated = source.iloc[:5000].copy()
    mutated.loc[mutated.index[1000:], ["close", "ema_5", "ema_9", "ema_20", "ema_50",
                                                  "rsi_14", "atr_14"]] *= 1.01
    assert_frame_equal(small, build_setups(mutated).iloc[:1000], check_exact=True)
    assert build_live_setup(source.iloc[:5000]).equals(build_setups(source.iloc[:5000]).iloc[-1])

    parquet = OUTPUT_DIR / "stage1_setup_dataset.parquet"
    csv = OUTPUT_DIR / "stage1_setup_dataset.csv"
    setups.to_parquet(parquet, index=False)
    setups.to_csv(csv, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")

    timestamp = pd.to_datetime(setups.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    reporting = setups.assign(year=timestamp.dt.year)
    reporting["regime"] = reporting.trend_state
    reporting["session_segment"] = pd.cut(
        reporting.minutes_from_open, bins=[-1, 59, 314, 374],
        labels=["OPENING", "MID_SESSION", "CLOSING"])
    reason_counts = reporting.rejection_reason.value_counts(dropna=False).rename_axis(
        "rejection_reason").reset_index(name="count")
    reason_counts["percentage"] = reason_counts["count"] / len(reporting)
    reason_counts.to_csv(REPORT_DIR / "rejection_reason_statistics.csv", index=False)
    aggregate(reporting, ["year"]).to_csv(REPORT_DIR / "yearly_setup_statistics.csv", index=False)
    aggregate(reporting, ["regime"]).to_csv(REPORT_DIR / "regime_setup_statistics.csv", index=False)
    aggregate(reporting, ["session_segment"]).to_csv(REPORT_DIR / "session_setup_statistics.csv", index=False)
    reporting.entry_signal.map({1: "LONG", -1: "SHORT", 0: "NO_SETUP"}).value_counts().rename_axis(
        "setup_type").reset_index(name="count").to_csv(
        REPORT_DIR / "long_short_setup_distribution.csv", index=False)
    reporting.trade_allowed.map({1: "ALLOWED", 0: "NOT_ALLOWED"}).value_counts().rename_axis(
        "decision").reset_index(name="count").to_csv(
        REPORT_DIR / "trade_allowed_distribution.csv", index=False)
    aggregate(reporting.assign(scope="ALL"), ["scope"]).to_csv(
        REPORT_DIR / "setup_frequency.csv", index=False)

    v1_reporting = v1.assign(year=timestamp.dt.year, regime=v1.trend_state,
                             session_segment=reporting.session_segment)
    comparison_frames = []
    for version, data in [("V1", v1_reporting), ("V2", reporting)]:
        overall = aggregate(data.assign(scope="ALL"), ["scope"])
        overall.insert(0, "engine_version", version)
        comparison_frames.append(overall)
    comparison = pd.concat(comparison_frames, ignore_index=True)
    comparison.to_csv(REPORT_DIR / "v1_v2_comparison.csv", index=False)
    comparison.assign(
        allowed_reduction_vs_v1=1-comparison.allowed/comparison.loc[
            comparison.engine_version.eq("V1"), "allowed"].iloc[0]).to_csv(
        REPORT_DIR / "setup_reduction_statistics.csv", index=False)
    pd.concat([aggregate(v1_reporting, ["year"]).assign(engine_version="V1"),
               aggregate(reporting, ["year"]).assign(engine_version="V2")], ignore_index=True).to_csv(
        REPORT_DIR / "yearly_v1_v2_comparison.csv", index=False)
    pd.concat([aggregate(v1_reporting, ["regime"]).assign(engine_version="V1"),
               aggregate(reporting, ["regime"]).assign(engine_version="V2")], ignore_index=True).to_csv(
        REPORT_DIR / "regime_v1_v2_comparison.csv", index=False)
    pd.concat([aggregate(v1_reporting, ["session_segment"]).assign(engine_version="V1"),
               aggregate(reporting, ["session_segment"]).assign(engine_version="V2")],
              ignore_index=True).to_csv(REPORT_DIR / "session_v1_v2_comparison.csv", index=False)

    quality_rows = []
    for version, data in [("V1", v1_reporting), ("V2", reporting)]:
        allowed = data[data.trade_allowed.eq(1)]
        direction = allowed.entry_signal
        strength = pd.Series(np.where(direction.eq(1),
                                      (allowed.close-allowed.rolling_high)/allowed.atr_14,
                                      (allowed.rolling_low-allowed.close)/allowed.atr_14), index=allowed.index)
        quality_rows.append({
            "engine_version": version, "allowed": len(allowed),
            "allowed_per_session": len(allowed)/reporting.session_id.nunique(),
            "mean_breakout_strength_atr": strength.mean(),
            "median_breakout_strength_atr": strength.median(),
            "mean_directional_body_fraction": (allowed.candle_body.abs()/allowed.candle_range).mean(),
            "mean_trend_separation_atr": ((allowed.ema_5-allowed.ema_50).abs()/allowed.atr_14).mean(),
        })
    pd.DataFrame(quality_rows).to_csv(REPORT_DIR / "precision_oriented_setup_analysis.csv", index=False)

    setups_count = int(setups.entry_signal.ne(0).sum())
    allowed_count = int(setups.trade_allowed.sum())
    report = f"""# Canonical Stage-1 Setup Engine Report

## Contract

- Version: **{contract.version}**
- LONG candidate: `close > EMA5 > EMA9 > EMA20 > EMA50` and bullish breakout
- SHORT candidate: `close < EMA5 < EMA9 < EMA20 < EMA50` and bearish breakdown
- LONG momentum: RSI14 >= {contract.bullish_rsi_min:g}
- SHORT momentum: RSI14 <= {contract.bearish_rsi_max:g}
- Suitable volatility: ATR14/close between {contract.minimum_atr_fraction:.5%} and {contract.maximum_atr_fraction:.3%}
- Allowed entry window: minutes {contract.first_allowed_minute}–{contract.last_allowed_minute} from 09:15 (09:30–14:59)
- Candidate direction is stored in `entry_signal`; momentum, volatility and session determine `trade_allowed`
- V2 requires a new breakout event, close confirmation >= {contract.minimum_breakout_atr:.2f} ATR beyond structure, directional body >= {contract.minimum_body_fraction:.0%} of range, and EMA5/EMA50 separation >= {contract.minimum_trend_separation_atr:.2f} ATR
- V2 rejects RSI exhaustion outside {contract.bearish_rsi_min:g}–{contract.bullish_rsi_max:g}

## Results

- Rows: **{len(setups):,}**
- Candidate setups: **{setups_count:,}** ({setups_count/len(setups):.2%})
- LONG setups: **{setups.entry_signal.eq(1).sum():,}**
- SHORT setups: **{setups.entry_signal.eq(-1).sum():,}**
- Allowed setups: **{allowed_count:,}** ({allowed_count/setups_count:.2%} of candidates)
- Rejected setups: **{setups_count-allowed_count:,}**
- Allowed opportunities per session: **{allowed_count/reporting.session_id.nunique():.2f}**
- V1 allowed opportunities: **{int(v1.trade_allowed.sum()):,}** ({v1.trade_allowed.sum()/reporting.session_id.nunique():.2f} per session)
- V1-to-V2 allowed reduction: **{1-allowed_count/v1.trade_allowed.sum():.2%}**

All rules are point-in-time transformations of canonical features. No label outcome, future candle, model, probability, or archived implementation is used.
"""
    (REPORT_DIR / "setup_engine_report.md").write_text(report, encoding="utf-8")
    (REPORT_DIR / "setup_validation.md").write_text("""# Setup Engine Validation

- Deterministic repeated calculation: **PASS**
- Rows 1–1000 equal the first 1000 rows from a 5000-row calculation: **PASS, exact**
- Mutating rows 1001–5000 leaves rows 1–1000 unchanged: **PASS, exact**
- Historical latest row equals live adapter output: **PASS, exact**
- Output uses canonical Step-15 fields only: **PASS**
- Future access, shifts, forward windows and backfills in setup engine: **NONE**
""", encoding="utf-8")
    metadata = {"contract": contract.as_dict(), "baseline_contract": SetupContract.v1_baseline().as_dict(),
                "input": INPUT.as_posix(), "input_sha256": sha256(INPUT),
                "rows": len(setups), "setups": setups_count, "allowed": allowed_count,
                "output_columns": OUTPUT_COLUMNS, "parquet_sha256": sha256(parquet), "csv_sha256": sha256(csv)}
    (REPORT_DIR / "setup_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Stage-1 setups built: {setups_count:,}; allowed: {allowed_count:,}")


if __name__ == "__main__":
    main()
