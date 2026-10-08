"""Research-only audit of whether the fixed 5%/3% label matches scalping intent."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.expired_option_dataset import load_selected_contract_panel  # noqa: E402
from tools.run_phase12_feature_signal_audit import folds  # noqa: E402


OUTPUT = ROOT / "reports/label_objective_audit"
FEATURES = ROOT / "data/normalized/historical_options/phase13_contract_features/feature_dataset_49.parquet"


def sources():
    root = ROOT / "data/normalized/historical_options/validated20"
    for session in sorted(path for path in root.iterdir() if path.is_dir()):
        yield session, session.name, ROOT / "data/external/shoonyatrader/validated20_raw" / f"{session.name.replace('-', '')}.zip"
    root = ROOT / "data/normalized/historical_options/non_expiry15"
    for session in sorted(path for path in root.iterdir() if path.is_dir()):
        manifest = json.loads((session / "manifest.json").read_text())
        yield session, manifest["date"], ROOT / "data/external/shoonyatrader/validated20_raw" / manifest["source_archive"]


def forward_paths() -> tuple[dict, dict]:
    option_paths, nifty_paths = {}, {}
    for session, date, archive in sources():
        index, ce, pe = (pd.read_parquet(session / name) for name in
                         ("nifty.parquet", "ce.parquet", "pe.parquet"))
        index["timestamp"] = pd.to_datetime(index.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
        nifty_paths[date] = index.set_index("timestamp").sort_index()
        panel = load_selected_contract_panel(archive, ce, pe, session_date=date)
        for key, group in panel.groupby(["expiry", "strike", "option_type"], sort=False):
            option_paths[(date, str(key[0]), float(key[1]), str(key[2]))] = group.set_index(
                "timestamp", drop=False).sort_index()
    return option_paths, nifty_paths


def at_horizon(path: pd.DataFrame, timestamp: pd.Timestamp, minutes: int) -> float:
    target = timestamp + pd.Timedelta(minutes=minutes)
    return float(path.loc[target, "close"]) if target in path.index else np.nan


def classify_sign(value: float) -> str:
    if pd.isna(value):
        return "AMBIGUOUS"
    if value > 0:
        return "WIN"
    if value < 0:
        return "LOSS"
    return "AMBIGUOUS"


def alternative_labels(observations: pd.DataFrame, option_paths: dict,
                       nifty_paths: dict) -> pd.DataFrame:
    rows = []
    for row in observations.itertuples(index=False):
        timestamp, date = pd.Timestamp(row.timestamp), str(row.session_date)
        selected = option_paths[(date, str(row.expiry), float(row.strike), str(row.option_type))]
        opposite_side = "PE" if row.option_type == "CE" else "CE"
        opposite = option_paths[(date, str(row.expiry), float(row.strike), opposite_side)]
        nifty = nifty_paths[date]
        selected_5 = (at_horizon(selected, timestamp, 5) / row.entry_premium - 1) * 100
        selected_10 = (at_horizon(selected, timestamp, 10) / row.entry_premium - 1) * 100
        opposite_entry = float(opposite.loc[timestamp, "close"]) if timestamp in opposite.index else np.nan
        opposite_5 = (at_horizon(opposite, timestamp, 5) / opposite_entry - 1) * 100
        nifty_entry = float(nifty.loc[timestamp, "close"])
        nifty_5 = (at_horizon(nifty, timestamp, 5) / nifty_entry - 1) * 100
        directional_nifty = nifty_5 if row.option_type == "CE" else -nifty_5
        atr_pct = row.ce_atr_pct if row.option_type == "CE" else row.pe_atr_pct
        risk_adjusted = selected_10 / atr_pct if pd.notna(atr_pct) and atr_pct > 0 else np.nan
        risk_label = ("WIN" if risk_adjusted >= 1 else "LOSS" if risk_adjusted <= -1
                      else "NEUTRAL" if pd.notna(risk_adjusted) else "AMBIGUOUS")
        rows.append({
            "A_DIRECTIONAL_NIFTY_5M": classify_sign(directional_nifty),
            "B_CE_PE_RELATIVE_WINNER_5M": classify_sign(selected_5 - opposite_5),
            "C_RISK_ADJUSTED_OPTION_10M": risk_label,
            "D_FORWARD_OPTION_5M": classify_sign(selected_5),
            "D_FORWARD_OPTION_10M": classify_sign(selected_10),
            "E_MFE_OPPORTUNITY_10M": "WIN" if row.mfe_10m_pct >= 5 else "LOSS",
            "nifty_directional_return_5m_pct": directional_nifty,
            "selected_return_5m_pct": selected_5, "selected_return_10m_pct": selected_10,
            "opposite_return_5m_pct": opposite_5, "risk_adjusted_return_10m": risk_adjusted,
        })
    return pd.DataFrame(rows)


def summarize_label(data: pd.DataFrame, column: str) -> tuple[dict, pd.DataFrame]:
    counts = data[column].value_counts().to_dict()
    fold_rows = []
    for fold_id, (_, test_dates) in enumerate(folds(data), start=1):
        test = data.loc[data.session_date.isin(test_dates)]
        fold_counts = test[column].value_counts()
        eligible = int(fold_counts.get("WIN", 0) + fold_counts.get("LOSS", 0))
        fold_rows.append({"label": column, "fold": fold_id, "rows": len(test),
                          "wins": int(fold_counts.get("WIN", 0)),
                          "losses": int(fold_counts.get("LOSS", 0)),
                          "neutral": int(fold_counts.get("NEUTRAL", 0)),
                          "ambiguous": int(fold_counts.get("AMBIGUOUS", 0)),
                          "win_rate_eligible": float(fold_counts.get("WIN", 0) / eligible)
                              if eligible else None})
    fold_frame = pd.DataFrame(fold_rows)
    return {
        "label": column, "class_balance": counts,
        "ambiguity_pct": float(data[column].eq("AMBIGUOUS").mean() * 100),
        "neutral_pct": float(data[column].eq("NEUTRAL").mean() * 100),
        "fold_win_rate_std": float(fold_frame.win_rate_eligible.std(ddof=0)),
        "fold_win_rate_min": float(fold_frame.win_rate_eligible.min()),
        "fold_win_rate_max": float(fold_frame.win_rate_eligible.max()),
    }, fold_frame


def group_breakdown(data: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    rows = []
    for label in labels:
        for dimension in ("expiry_day_status", "option_type", "trend_regime"):
            for value, group in data.groupby(dimension, sort=False):
                counts = group[label].value_counts()
                eligible = int(counts.get("WIN", 0) + counts.get("LOSS", 0))
                rows.append({"label": label, "dimension": dimension, "group": value,
                             "rows": len(group), "wins": int(counts.get("WIN", 0)),
                             "losses": int(counts.get("LOSS", 0)),
                             "neutral": int(counts.get("NEUTRAL", 0)),
                             "ambiguous": int(counts.get("AMBIGUOUS", 0)),
                             "win_rate_eligible": float(counts.get("WIN", 0) / eligible)
                                 if eligible else None})
    return pd.DataFrame(rows)


def main() -> None:
    raw = pd.concat([
        pd.read_parquet(ROOT / "reports/phase9_label_behavior_audit/trade_path_audit.parquet"),
        pd.read_parquet(ROOT / "reports/phase10_label_suitability/non_expiry_trade_path_audit.parquet"),
    ], ignore_index=True)
    observations = raw.sort_values("timestamp", kind="stable").drop_duplicates(
        ["timestamp", "expiry", "strike", "option_type"], keep="first").reset_index(drop=True)
    features = pd.read_parquet(FEATURES)
    observations = observations.merge(
        features[["timestamp", "ce_atr_pct", "pe_atr_pct"]], on="timestamp", how="left",
        validate="many_to_one")
    option_paths, nifty_paths = forward_paths()
    alternatives = alternative_labels(observations, option_paths, nifty_paths)
    data = pd.concat([observations, alternatives], axis=1)
    data["CURRENT_FIXED_5_3"] = data.actual_exit_reason
    label_columns = ["CURRENT_FIXED_5_3", "A_DIRECTIONAL_NIFTY_5M",
                     "B_CE_PE_RELATIVE_WINNER_5M", "C_RISK_ADJUSTED_OPTION_10M",
                     "D_FORWARD_OPTION_5M", "D_FORWARD_OPTION_10M", "E_MFE_OPPORTUNITY_10M"]
    summaries, fold_frames = [], []
    for column in label_columns:
        summary, fold_frame = summarize_label(data, column)
        summaries.append(summary)
        fold_frames.append(fold_frame)
    first_candle = data.holding_minutes.eq(1)
    first_resolved = first_candle & data.actual_exit_reason.isin(["WIN", "LOSS"])
    current_binary = data.actual_exit_reason.isin(["WIN", "LOSS"])
    directional_agreement = float(
        data.loc[current_binary, "CURRENT_FIXED_5_3"].eq(
            data.loc[current_binary, "A_DIRECTIONAL_NIFTY_5M"]).mean())
    result = {
        "status": "LABEL_TARGET_OBJECTIVE_AUDIT_COMPLETE",
        "verdict": "CURRENT_LABEL_DOES_NOT_MATCH_SCALPING_OBJECTIVE",
        "raw_strategy_trades": len(raw), "unique_economic_observations": len(data),
        "current_label": {
            "meaning": "Entry-premium first-touch of +5% target or -3% stop within the locked contract path; costs do not determine the class.",
            "counts": data.actual_exit_reason.value_counts().to_dict(),
            "first_candle_resolved_win_loss": int(first_resolved.sum()),
            "first_candle_resolved_pct": float(first_resolved.sum() / current_binary.sum() * 100),
            "first_candle_all_outcomes": int(first_candle.sum()),
            "directional_nifty_label_agreement": directional_agreement,
            "interpretation": "Primarily measures short-horizon option-premium barrier volatility, especially on expiry days; it is not a direct NIFTY-direction label.",
        },
        "alternatives": summaries,
        "alternative_meanings": {
            "A_DIRECTIONAL_NIFTY_5M": "Whether NIFTY's 5m close moved in the selected CE/PE direction; directional but ignores premium economics.",
            "B_CE_PE_RELATIVE_WINNER_5M": "Whether selected side outperformed the opposite option at the same strike over 5m; relative, not absolute profit.",
            "C_RISK_ADJUSTED_OPTION_10M": "10m selected-premium return divided by entry ATR%; WIN >= +1R, LOSS <= -1R, otherwise NEUTRAL.",
            "D_FORWARD_OPTION_5M": "Sign of selected option's fixed 5m close return.",
            "D_FORWARD_OPTION_10M": "Sign of selected option's fixed 10m close return.",
            "E_MFE_OPPORTUNITY_10M": "Whether selected option reached +5% MFE within 10m, ignoring adverse-path order.",
        },
        "labels_changed": False, "model_trained": False, "features_changed": False,
        "strategy_selected": False, "production_changed": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUTPUT / "label_comparison_observations.parquet", index=False)
    pd.DataFrame(summaries).to_json(OUTPUT / "label_class_balance.json", orient="records", indent=2)
    pd.concat(fold_frames, ignore_index=True).to_csv(OUTPUT / "label_fold_stability.csv", index=False)
    group_breakdown(data, label_columns).to_csv(OUTPUT / "label_group_breakdown.csv", index=False)
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
