"""Research-only audit of option time decay and entry-time economic context."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_label_objective_audit import at_horizon, forward_paths  # noqa: E402
from tools.run_phase15_signal_discovery import evaluate  # noqa: E402
from tools.run_risk_adjusted_catboost_research import FEATURES, load_data  # noqa: E402


OUTPUT = ROOT / "reports/phase16_time_decay_audit"
LABEL_DATA = ROOT / "reports/corrected_scalping_label_audit/label_comparison.parquet"


def intrinsic_value(spot: float, strike: float, option_type: str) -> float:
    return max(spot - strike, 0.0) if option_type == "CE" else max(strike - spot, 0.0)


def contract_snapshot(path: pd.DataFrame, nifty: pd.DataFrame, timestamp: pd.Timestamp,
                      strike: float, option_type: str) -> dict | None:
    if timestamp not in path.index or timestamp not in nifty.index:
        return None
    row = path.loc[timestamp]
    spot, premium = float(nifty.loc[timestamp, "close"]), float(row.close)
    intrinsic = intrinsic_value(spot, strike, option_type)
    return {"spot": spot, "premium": premium, "intrinsic": intrinsic,
            "extrinsic": premium - intrinsic, "volume": row.get("volume"), "oi": row.get("oi")}


def enrich(data: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_parquet(LABEL_DATA)
    metadata = ["timestamp", "expiry", "strike", "entry_premium",
                "nifty_directional_return_5m_pct"]
    data = data.merge(raw[metadata], on="timestamp", how="left", validate="one_to_one")
    option_paths, nifty_paths = forward_paths()
    records = []
    for row in data.itertuples(index=False):
        timestamp, date = pd.Timestamp(row.timestamp), str(row.session_date)
        expiry = pd.Timestamp(row.expiry).date()
        expiry_close = pd.Timestamp(f"{expiry} 15:30", tz="Asia/Kolkata")
        session_close = pd.Timestamp(f"{date} 15:30", tz="Asia/Kolkata")
        selected_key = (date, str(row.expiry), float(row.strike), row.option_type)
        opposite_type = "PE" if row.option_type == "CE" else "CE"
        opposite_key = (date, str(row.expiry), float(row.strike), opposite_type)
        selected, opposite = option_paths[selected_key], option_paths[opposite_key]
        nifty = nifty_paths[date]
        current = contract_snapshot(selected, nifty, timestamp, row.strike, row.option_type)
        other = contract_snapshot(opposite, nifty, timestamp, row.strike, opposite_type)
        prior_time, future_time = timestamp - pd.Timedelta(minutes=5), timestamp + pd.Timedelta(minutes=5)
        prior = contract_snapshot(selected, nifty, prior_time, row.strike, row.option_type)
        prior_other = contract_snapshot(opposite, nifty, prior_time, row.strike, opposite_type)
        future = contract_snapshot(selected, nifty, future_time, row.strike, row.option_type)
        signed_moneyness = ((current["spot"] - row.strike) / current["spot"] * 100
                            if row.option_type == "CE" else
                            (row.strike - current["spot"]) / current["spot"] * 100)
        option_atr = row.ce_atr_pct if row.option_type == "CE" else row.pe_atr_pct
        selected_decay = ((current["extrinsic"] - prior["extrinsic"]) / 5 if prior else np.nan)
        opposite_decay = ((other["extrinsic"] - prior_other["extrinsic"]) / 5
                          if prior_other else np.nan)
        records.append({
            "calendar_days_to_expiry": (expiry - timestamp.date()).days,
            "minutes_to_expiry_close": (expiry_close - timestamp).total_seconds() / 60,
            "minutes_to_session_close": (session_close - timestamp).total_seconds() / 60,
            "spot": current["spot"], "selected_premium": current["premium"],
            "opposite_premium": other["premium"], "selected_intrinsic": current["intrinsic"],
            "selected_extrinsic": current["extrinsic"], "opposite_extrinsic": other["extrinsic"],
            "signed_moneyness_pct": signed_moneyness,
            "absolute_moneyness_pct": abs(signed_moneyness),
            "intrinsic_to_premium": current["intrinsic"] / current["premium"],
            "extrinsic_to_premium": current["extrinsic"] / current["premium"],
            "opposite_extrinsic_to_premium": other["extrinsic"] / other["premium"],
            "relative_premium_log_ratio": np.log(current["premium"] / other["premium"]),
            "selected_volume": current["volume"], "selected_oi": current["oi"],
            "opposite_volume": other["volume"], "opposite_oi": other["oi"],
            "extrinsic_change_per_minute_5m": selected_decay,
            "normalized_extrinsic_change_5m":
                ((selected_decay / current["premium"] * 100) / option_atr
                 if option_atr > 0 else np.nan),
            "ce_pe_relative_extrinsic_change_5m": selected_decay - opposite_decay,
            "extrinsic_per_remaining_minute":
                current["extrinsic"] / max((expiry_close - timestamp).total_seconds() / 60, 1),
            "forward_extrinsic_change_5m":
                future["extrinsic"] - current["extrinsic"] if future else np.nan,
            "forward_extrinsic_change_pct_of_premium_5m":
                ((future["extrinsic"] - current["extrinsic"]) / current["premium"] * 100
                 if future else np.nan),
        })
    result = pd.concat([data.reset_index(drop=True), pd.DataFrame(records)], axis=1)
    result["moneyness_bucket"] = pd.cut(
        result.signed_moneyness_pct, [-np.inf, -0.1, 0.1, np.inf], labels=["OTM", "ATM", "ITM"])
    return result


def source_audit(data: pd.DataFrame) -> list[dict]:
    fields = {
        "expiry_date": ("expiry", "SOURCE"), "entry_timestamp": ("timestamp", "SOURCE"),
        "strike": ("strike", "SOURCE"), "spot": ("spot", "SOURCE"),
        "ce_pe_premium": ("selected_premium", "SOURCE"), "volume": ("selected_volume", "SOURCE"),
        "oi": ("selected_oi", "SOURCE"), "iv": (None, "UNAVAILABLE"),
        "greeks_theta": (None, "UNAVAILABLE"), "bid_ask_spread": (None, "UNAVAILABLE"),
        "time_to_expiry": ("minutes_to_expiry_close", "DERIVED_FROM_EXPIRY_DATE_AND_NSE_CLOSE"),
        "intrinsic_value": ("selected_intrinsic", "DERIVED_EXACT"),
        "extrinsic_value": ("selected_extrinsic", "DERIVED_EXACT"),
        "moneyness": ("signed_moneyness_pct", "DERIVED_EXACT"),
    }
    rows = []
    for field, (column, provenance) in fields.items():
        coverage = float(data[column].notna().mean() * 100) if column else 0.0
        rows.append({"field": field, "provenance": provenance, "coverage_pct": coverage,
                     "available": bool(column and coverage == 100)})
    return rows


def behavior(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dimensions = ["expiry_day_status", "option_type", "moneyness_bucket", "calendar_days_to_expiry"]
    for dimension in dimensions:
        for value, group in data.groupby(dimension, observed=True):
            rows.append({"dimension": dimension, "group": str(value), "rows": len(group),
                         "mean_forward_premium_return_5m": float(group.selected_return_5m_pct.mean()),
                         "mean_forward_extrinsic_change_5m": float(group.forward_extrinsic_change_5m.mean()),
                         "mean_forward_extrinsic_pct_5m": float(
                             group.forward_extrinsic_change_pct_of_premium_5m.mean()),
                         "mean_entry_extrinsic": float(group.selected_extrinsic.mean()),
                         "win_rate": float(group.label.mean())})
    flat = data.loc[data.nifty_directional_return_5m_pct.abs().le(0.05)]
    for value, group in flat.groupby("expiry_day_status"):
        rows.append({"dimension": "NIFTY_NEAR_FLAT_ABS_5M_LE_0.05PCT", "group": value,
                     "rows": len(group),
                     "mean_forward_premium_return_5m": float(group.selected_return_5m_pct.mean()),
                     "mean_forward_extrinsic_change_5m": float(group.forward_extrinsic_change_5m.mean()),
                     "mean_forward_extrinsic_pct_5m": float(
                         group.forward_extrinsic_change_pct_of_premium_5m.mean()),
                     "mean_entry_extrinsic": float(group.selected_extrinsic.mean()),
                     "win_rate": float(group.label.mean())})
    return pd.DataFrame(rows)


def controlled_underlying_response(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bins = [-np.inf, -0.10, -0.05, 0.05, 0.10, np.inf]
    labels = ["LE_-0.10", "-0.10_TO_-0.05", "NEAR_FLAT", "0.05_TO_0.10", "GE_0.10"]
    framed = data.assign(nifty_move_bucket=pd.cut(
        data.nifty_directional_return_5m_pct, bins, labels=labels))
    for (expiry_status, side), group in framed.groupby(
            ["expiry_day_status", "option_type"], observed=True):
        valid = group[["nifty_directional_return_5m_pct", "selected_return_5m_pct"]].dropna()
        slope, intercept = np.polyfit(valid.nifty_directional_return_5m_pct,
                                      valid.selected_return_5m_pct, 1)
        rows.append({"record_type": "LINEAR_CONTROL", "expiry_day_status": expiry_status,
                     "option_type": side, "nifty_move_bucket": "ALL", "rows": len(valid),
                     "premium_return_5m": float(valid.selected_return_5m_pct.mean()),
                     "response_slope": float(slope), "response_intercept": float(intercept),
                     "correlation": float(valid.corr().iloc[0, 1])})
    for (expiry_status, side, bucket), group in framed.groupby(
            ["expiry_day_status", "option_type", "nifty_move_bucket"], observed=True):
        rows.append({"record_type": "MOVE_BUCKET", "expiry_day_status": expiry_status,
                     "option_type": side, "nifty_move_bucket": str(bucket), "rows": len(group),
                     "premium_return_5m": float(group.selected_return_5m_pct.mean()),
                     "response_slope": np.nan, "response_intercept": np.nan,
                     "correlation": np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    base, leakage = load_data()
    data = enrich(base)
    tte = ["calendar_days_to_expiry", "minutes_to_expiry_close", "minutes_to_session_close"]
    moneyness = ["signed_moneyness_pct", "absolute_moneyness_pct", "intrinsic_to_premium"]
    extrinsic = ["extrinsic_to_premium", "opposite_extrinsic_to_premium",
                 "relative_premium_log_ratio", "selected_extrinsic"]
    decay = ["extrinsic_change_per_minute_5m", "normalized_extrinsic_change_5m",
             "ce_pe_relative_extrinsic_change_5m", "extrinsic_per_remaining_minute"]
    matrices = {
        "A_EXISTING_48": FEATURES,
        "B_48_PLUS_TIME_TO_EXPIRY": FEATURES + tte,
        "C_48_PLUS_MONEYNESS": FEATURES + moneyness,
        "D_48_PLUS_EXTRINSIC": FEATURES + extrinsic,
        "E_48_PLUS_TIME_DECAY_PROXY": FEATURES + decay,
        "F_COMBINED_VALIDATED": FEATURES + tte + moneyness + extrinsic + decay,
    }
    ranking, fold_rows, group_rows = [], [], []
    for name, columns in matrices.items():
        row, folds_result, groups_result = evaluate(data, name, "TIME_DECAY_MATRIX", columns)
        ranking.append(row); fold_rows.extend(folds_result); group_rows.extend(groups_result)
    ranking = pd.DataFrame(ranking)
    baseline = float(ranking.loc[ranking.candidate.eq("A_EXISTING_48"), "mean_oos_auc"].iloc[0])
    ranking["auc_improvement_vs_phase16_baseline"] = ranking.mean_oos_auc - baseline
    ranking["passes_phase16_gate"] = (
        ranking.mean_oos_auc.ge(0.55) & ranking.minimum_fold_auc.ge(0.52)
        & ranking.positive_folds.ge(4) & ranking.ce_pe_min_auc.ge(0.52)
        & ranking.expiry_min_auc.ge(0.52) & ranking.regime_min_auc.ge(0.48)
        & ranking.auc_improvement_vs_phase16_baseline.ge(0.02))
    additions = ranking.loc[~ranking.candidate.eq("A_EXISTING_48")]
    passing = additions.loc[additions.passes_phase16_gate]
    source = source_audit(data)
    behavior_frame = behavior(data)
    controlled = controlled_underlying_response(data)
    expiry = data.groupby("expiry_day_status").agg(
        rows=("label", "size"), premium_return_5m=("selected_return_5m_pct", "mean"),
        extrinsic_change_5m=("forward_extrinsic_change_5m", "mean"),
        opportunity_rate=("label", "mean"), score=("risk_score_5m", "mean")).reset_index()
    moneyness_effect = data.groupby("moneyness_bucket", observed=True).agg(
        rows=("label", "size"), premium_return_5m=("selected_return_5m_pct", "mean"),
        extrinsic_change_5m=("forward_extrinsic_change_5m", "mean"),
        opportunity_rate=("label", "mean")).reset_index()
    verdict = "VALID_TIME_DECAY_PROXY_AVAILABLE" if len(passing) else "NO_VALID_TIME_DECAY_SIGNAL"
    result = {
        "status": "PHASE16_OPTION_TIME_DECAY_AUDIT_COMPLETE", "verdict": verdict,
        "observations": len(data), "sessions": int(data.session_date.nunique()),
        "true_theta_available": False, "historical_iv_available": False,
        "valid_volume_available": True, "valid_oi_available": True,
        "bid_ask_available": False,
        "theta_calculation_decision": "Not calculated: historical IV, rate/dividend inputs and quotes are not validated.",
        "research_proxy": {
            "valid": True, "name": "same-contract extrinsic change per minute over prior 5m",
            "causal": True, "production_feature": False,
            "predictive_gate_passed": bool(len(passing)),
        },
        "data_quality": {
            "negative_extrinsic_observations": int(data.selected_extrinsic.lt(0).sum()),
            "negative_extrinsic_pct": float(data.selected_extrinsic.lt(0).mean() * 100),
            "proxy_missing_observations": int(data.extrinsic_change_per_minute_5m.isna().sum()),
            "handling": "Raw arithmetic retained; negative extrinsic was not clamped or repaired.",
        },
        "source_fields": source,
        "expiry_non_expiry": expiry.to_dict("records"),
        "moneyness_effect": moneyness_effect.to_dict("records"),
        "near_flat_nifty": behavior_frame.loc[
            behavior_frame.dimension.eq("NIFTY_NEAR_FLAT_ABS_5M_LE_0.05PCT")].to_dict("records"),
        "underlying_control_regressions": controlled.loc[
            controlled.record_type.eq("LINEAR_CONTROL")].replace({np.nan: None}).to_dict("records"),
        "matrix_results": ranking.replace({np.nan: None}).to_dict("records"),
        "passing_feature_families": passing.candidate.tolist(),
        "leakage_audit": leakage,
        "catboost_trained": False, "production_features_changed": False,
        "label_changed": False, "strategy_selected": False, "threshold_tuned": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(source).to_csv(OUTPUT / "raw_data_coverage.csv", index=False)
    behavior_frame.to_csv(OUTPUT / "time_decay_behavior.csv", index=False)
    controlled.to_csv(OUTPUT / "underlying_controlled_response.csv", index=False)
    ranking.to_csv(OUTPUT / "predictive_matrix_ranking.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUTPUT / "fold_results.csv", index=False)
    pd.DataFrame(group_rows).to_csv(OUTPUT / "group_results.csv", index=False)
    data[["timestamp", "session_date", "option_type", "expiry", "strike", "spot",
          "selected_premium", "selected_intrinsic", "selected_extrinsic", "signed_moneyness_pct",
          "calendar_days_to_expiry", "minutes_to_expiry_close", "selected_volume", "selected_oi",
          *decay]].to_parquet(OUTPUT / "economic_observations.parquet", index=False)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
