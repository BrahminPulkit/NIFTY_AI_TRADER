"""Research-only comparison of current and risk-adjusted scalping labels."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_label_objective_audit import group_breakdown, summarize_label  # noqa: E402


INPUT = ROOT / "reports/label_objective_audit/label_comparison_observations.parquet"
OUTPUT = ROOT / "reports/corrected_scalping_label_audit"


def finite_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.where(denominator.gt(0))
    return result.where(np.isfinite(result))


def build_risk_adjusted_label(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Create an untuned, fixed-horizon opportunity label in entry-time risk units."""
    option_atr = np.where(data.option_type.eq("CE"), data.ce_atr_pct, data.pe_atr_pct)
    option_atr = pd.Series(option_atr, index=data.index)
    final_return = data[f"selected_return_{horizon}m_pct"]
    opposite_return = data[f"opposite_return_{horizon}m_pct"]
    directional_nifty = data[f"nifty_directional_return_{horizon}m_pct"]

    terminal_r = finite_ratio(final_return, option_atr)
    excursion_balance_r = finite_ratio(
        data[f"mfe_{horizon}m_pct"] - data[f"mae_{horizon}m_pct"].abs(), option_atr)
    relative_r = finite_ratio(final_return - opposite_return, option_atr)
    nifty_r = finite_ratio(directional_nifty, data.nifty_atr_pct)

    # Clipping prevents expiry-day premium explosions from dominating the score.
    components = pd.DataFrame({
        "terminal_r": terminal_r.clip(-3, 3),
        "excursion_balance_r": excursion_balance_r.clip(-3, 3),
        "relative_r": relative_r.clip(-3, 3),
        "nifty_r": nifty_r.clip(-3, 3),
    })
    score = (0.40 * components.terminal_r
             + 0.25 * components.excursion_balance_r
             + 0.20 * components.relative_r
             + 0.15 * components.nifty_r)
    complete = components.notna().all(axis=1)
    label = pd.Series("NEUTRAL", index=data.index, dtype="object")
    label.loc[~complete] = "AMBIGUOUS"
    label.loc[complete & score.ge(0.5) & terminal_r.gt(0) & relative_r.gt(0)] = "WIN"
    label.loc[complete & score.le(-0.5) & terminal_r.lt(0)] = "LOSS"
    return pd.DataFrame({
        f"RISK_ADJUSTED_{horizon}M": label,
        f"risk_score_{horizon}m": score,
        f"terminal_r_{horizon}m": terminal_r,
        f"excursion_balance_r_{horizon}m": excursion_balance_r,
        f"relative_r_{horizon}m": relative_r,
        f"nifty_r_{horizon}m": nifty_r,
    })


def correlations(data: pd.DataFrame, label: str, horizon: int) -> dict:
    score = data[f"risk_score_{horizon}m"]
    return {
        "score_vs_forward_return": float(score.corr(data[f"selected_return_{horizon}m_pct"])),
        "score_vs_mfe": float(score.corr(data[f"mfe_{horizon}m_pct"])),
        "score_vs_mae": float(score.corr(data[f"mae_{horizon}m_pct"])),
        "tradable_opportunity_pct": float(data[label].eq("WIN").mean() * 100),
    }


def label_relationships(data: pd.DataFrame, label: str, horizon: int) -> dict:
    encoded = data[label].map({"WIN": 1.0, "NEUTRAL": 0.0, "LOSS": -1.0})
    return {
        "label_vs_forward_return": float(encoded.corr(data[f"selected_return_{horizon}m_pct"])),
        "label_vs_mfe": float(encoded.corr(data[f"mfe_{horizon}m_pct"])),
        "label_vs_mae": float(encoded.corr(data[f"mae_{horizon}m_pct"])),
        "positive_label_pct_all_observations": float(data[label].eq("WIN").mean() * 100),
    }


def session_breakdown(data: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    rows = []
    for label in labels:
        for session, group in data.groupby("session_date", sort=True):
            counts = group[label].value_counts()
            rows.append({
                "label": label, "session_date": session, "rows": len(group),
                "wins": int(counts.get("WIN", 0)), "losses": int(counts.get("LOSS", 0)),
                "neutral": int(counts.get("NEUTRAL", 0)),
                "ambiguous": int(counts.get("AMBIGUOUS", 0)),
                "tradable_opportunity_pct": float(counts.get("WIN", 0) / len(group) * 100),
            })
    return pd.DataFrame(rows)


def main() -> None:
    data = pd.read_parquet(INPUT)
    features = pd.read_parquet(
        ROOT / "data/normalized/historical_options/phase13_contract_features/feature_dataset_49.parquet")
    data = data.merge(features[["timestamp", "nifty_atr_pct"]], on="timestamp", how="left",
                      validate="many_to_one")

    # The Phase-15 artifact already contains 5m opposite/NIFTY paths; add their 10m peers.
    from tools.run_label_objective_audit import forward_paths, at_horizon
    option_paths, nifty_paths = forward_paths()
    opposite_10, nifty_10 = [], []
    for row in data.itertuples(index=False):
        date, timestamp = str(row.session_date), pd.Timestamp(row.timestamp)
        opposite_side = "PE" if row.option_type == "CE" else "CE"
        opposite = option_paths[(date, str(row.expiry), float(row.strike), opposite_side)]
        opposite_entry = float(opposite.loc[timestamp, "close"]) if timestamp in opposite.index else np.nan
        opposite_10.append((at_horizon(opposite, timestamp, 10) / opposite_entry - 1) * 100)
        nifty = nifty_paths[date]
        nifty_entry = float(nifty.loc[timestamp, "close"])
        raw_nifty = (at_horizon(nifty, timestamp, 10) / nifty_entry - 1) * 100
        nifty_10.append(raw_nifty if row.option_type == "CE" else -raw_nifty)
    data["opposite_return_10m_pct"] = opposite_10
    data["nifty_directional_return_10m_pct"] = nifty_10

    data = pd.concat([data, build_risk_adjusted_label(data, 5),
                      build_risk_adjusted_label(data, 10)], axis=1)
    labels = ["CURRENT_FIXED_5_3", "RISK_ADJUSTED_5M", "RISK_ADJUSTED_10M"]
    summaries, fold_frames = [], []
    for label in labels:
        summary, fold_frame = summarize_label(data, label)
        summaries.append(summary)
        fold_frames.append(fold_frame)

    sessions = session_breakdown(data, labels)
    candidate_metrics = {
        "RISK_ADJUSTED_5M": correlations(data, "RISK_ADJUSTED_5M", 5),
        "RISK_ADJUSTED_10M": correlations(data, "RISK_ADJUSTED_10M", 10),
    }
    label_metrics = {
        "CURRENT_FIXED_5_3": label_relationships(data, "CURRENT_FIXED_5_3", 5),
        "RISK_ADJUSTED_5M": label_relationships(data, "RISK_ADJUSTED_5M", 5),
        "RISK_ADJUSTED_10M": label_relationships(data, "RISK_ADJUSTED_10M", 10),
    }
    for label in labels:
        label_metrics[label]["session_positive_rate_std"] = float(
            sessions.loc[sessions.label.eq(label), "tradable_opportunity_pct"].std(ddof=0))
    candidate_metrics["RISK_ADJUSTED_5M"]["session_opportunity_rate_std"] = label_metrics[
        "RISK_ADJUSTED_5M"]["session_positive_rate_std"]
    candidate_metrics["RISK_ADJUSTED_10M"]["session_opportunity_rate_std"] = label_metrics[
        "RISK_ADJUSTED_10M"]["session_positive_rate_std"]

    # Prefer the shorter horizon unless 10m materially improves correlation and stability.
    five, ten = candidate_metrics["RISK_ADJUSTED_5M"], candidate_metrics["RISK_ADJUSTED_10M"]
    ten_materially_better = (ten["score_vs_forward_return"] >= five["score_vs_forward_return"] + 0.05
                             and ten["session_opportunity_rate_std"] <=
                             five["session_opportunity_rate_std"])
    recommendation = ("REPLACE_WITH_10M_RISK_ADJUSTED_LABEL" if ten_materially_better
                      else "REPLACE_WITH_5M_RISK_ADJUSTED_LABEL")
    current_ambiguous = data.CURRENT_FIXED_5_3.eq("AMBIGUOUS")
    result = {
        "status": "CORRECTED_SCALPING_LABEL_DESIGN_COMPLETE",
        "recommendation": recommendation,
        "observations": len(data), "sessions": int(data.session_date.nunique()),
        "definition": {
            "formula": "0.40*terminal_R + 0.25*(MFE-|MAE|)_R + 0.20*relative_R + 0.15*NIFTY_direction_R",
            "normalization": "Entry-time option ATR% for option components; entry-time NIFTY ATR% for direction; each component clipped to [-3,3].",
            "WIN": "score >= 0.5, horizon option return > 0, and selected option outperforms opposite option",
            "LOSS": "score <= -0.5 and horizon option return < 0",
            "NEUTRAL": "complete observation that meets neither WIN nor LOSS",
            "AMBIGUOUS": "required fixed-horizon path or entry-time normalization unavailable",
            "current_ambiguous_policy": "No target/stop order is inferred. Corrected labels measure a different fixed-horizon objective.",
        },
        "label_summaries": summaries,
        "label_relationships": label_metrics,
        "candidate_metrics": candidate_metrics,
        "current_ambiguous_corrected_distribution": {
            "5m": data.loc[current_ambiguous, "RISK_ADJUSTED_5M"].value_counts().to_dict(),
            "10m": data.loc[current_ambiguous, "RISK_ADJUSTED_10M"].value_counts().to_dict(),
        },
        "model_trained": False, "features_changed": False, "current_label_changed": False,
        "strategy_selected": False, "production_changed": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUTPUT / "label_comparison.parquet", index=False)
    pd.DataFrame(summaries).to_json(OUTPUT / "class_balance.json", orient="records", indent=2)
    pd.concat(fold_frames, ignore_index=True).to_csv(OUTPUT / "fold_stability.csv", index=False)
    group_breakdown(data, labels).to_csv(OUTPUT / "group_breakdown.csv", index=False)
    sessions.to_csv(OUTPUT / "session_stability.csv", index=False)
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
