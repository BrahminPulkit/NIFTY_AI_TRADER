"""Research-only relationship feature audit and fixed-model ablation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ce_pe_feature_preview import feature_columns  # noqa: E402
from src.expired_option_dataset import load_selected_contract_panel  # noqa: E402
from tools.run_phase12_feature_signal_audit import folds, rank_features  # noqa: E402


SOURCE = ROOT / "data/normalized/historical_options/phase13_contract_features/feature_dataset_49.parquet"
OUTPUT = ROOT / "reports/relationship_feature_audit"
EXISTING = [name for name in feature_columns() if name != "nifty_vwap_distance"]
REDUNDANT_ALIASES = [
    "ce_strength_vs_nifty", "pe_strength_vs_nifty",
    "ce_strength_vs_pe", "pe_strength_vs_ce",
]


def proposed_formulas() -> dict[str, str]:
    formulas = {}
    for horizon in (1, 3):
        formulas[f"ce_response_vs_nifty_{horizon}m"] = f"ce_return_{horizon}m - nifty_return_{horizon}m"
        formulas[f"pe_response_vs_nifty_{horizon}m"] = f"pe_return_{horizon}m + nifty_return_{horizon}m"
        formulas[f"ce_pe_return_spread_{horizon}m"] = f"ce_return_{horizon}m - pe_return_{horizon}m"
        formulas[f"ce_pe_momentum_spread_{horizon}m"] = f"ce_momentum_{horizon}m - pe_momentum_{horizon}m"
    for prefix in ("nifty", "ce", "pe"):
        formulas[f"{prefix}_return_acceleration_1v3"] = f"{prefix}_return_1m - {prefix}_return_3m / 3"
        formulas[f"{prefix}_return_acceleration_3v5"] = f"{prefix}_return_3m / 3 - {prefix}_return_5m / 5"
        formulas[f"{prefix}_momentum_persistence"] = "sign(return_1m) when sign(return_1m)=sign(return_3m)=sign(return_5m), else 0"
        formulas[f"{prefix}_short_term_reversal"] = "1 when sign(return_1m) differs from sign(return_3m), else 0"
    formulas.update({
        "ce_pe_acceleration_spread": "ce_return_acceleration_1v3 - pe_return_acceleration_1v3",
        "ce_pe_relative_volatility": "(ce_volatility - pe_volatility) / (abs(ce_volatility) + abs(pe_volatility))",
        "ce_pe_atr_expansion_balance": "(ce_atr_pct - pe_atr_pct) / (abs(ce_atr_pct) + abs(pe_atr_pct))",
        "ce_pe_volume_imbalance": "(ce_volume_ratio - pe_volume_ratio) / (abs(ce_volume_ratio) + abs(pe_volume_ratio))",
        "nifty_option_directional_response_1m": "sign(nifty_return_1m) * (ce_return_1m - pe_return_1m)",
        "nifty_option_directional_response_3m": "sign(nifty_return_3m) * (ce_return_3m - pe_return_3m)",
        "bullish_response_strength": "ce_return_3m - pe_return_3m when nifty_return_3m > 0, else 0",
        "bearish_response_strength": "pe_return_3m - ce_return_3m when nifty_return_3m < 0, else 0",
    })
    for side in ("ce", "pe"):
        formulas[f"{side}_recent_range_position"] = "(close - prior_20m_low) / (prior_20m_high - prior_20m_low)"
        formulas[f"{side}_distance_recent_high_pct"] = "100 * (close / prior_20m_high - 1)"
        formulas[f"{side}_distance_recent_low_pct"] = "100 * (close / prior_20m_low - 1)"
    return formulas


def existing_category(name: str) -> str:
    if name in {"ce_premium_acceleration", "pe_premium_acceleration",
                "ce_volume_ratio", "pe_volume_ratio", "ce_volume_ratio_minus_pe_volume_ratio"}:
        return "volume_acceleration"
    if name.startswith("nifty_"):
        return "nifty_price_structure" if name == "nifty_ema_structure" else "nifty_momentum_volatility"
    if "minus_nifty" in name or "plus_nifty" in name or "vs_nifty" in name or "option_confirmation" in name:
        return "option_vs_nifty_relationship"
    if "minus_pe" in name or "minus_pe" in name or "vs_pe" in name or "vs_ce" in name:
        return "ce_vs_pe_relative_strength"
    if name.startswith("ce_"):
        return "ce_state"
    if name.startswith("pe_"):
        return "pe_state"
    return "ce_vs_pe_relative_strength"


def _safe_balance(left: pd.Series, right: pd.Series) -> pd.Series:
    denominator = left.abs() + right.abs()
    return (left - right) / denominator.replace(0, np.nan)


def relationship_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Causal relationships derived only from current entry-time feature values."""
    x = frame
    out = pd.DataFrame(index=x.index)
    for horizon in (1, 3):
        out[f"ce_response_vs_nifty_{horizon}m"] = (
            x[f"ce_return_{horizon}m"] - x[f"nifty_return_{horizon}m"])
        out[f"pe_response_vs_nifty_{horizon}m"] = (
            x[f"pe_return_{horizon}m"] + x[f"nifty_return_{horizon}m"])
        out[f"ce_pe_return_spread_{horizon}m"] = (
            x[f"ce_return_{horizon}m"] - x[f"pe_return_{horizon}m"])
        out[f"ce_pe_momentum_spread_{horizon}m"] = (
            x[f"ce_momentum_{horizon}m"] - x[f"pe_momentum_{horizon}m"])
    for prefix in ("nifty", "ce", "pe"):
        out[f"{prefix}_return_acceleration_1v3"] = (
            x[f"{prefix}_return_1m"] - x[f"{prefix}_return_3m"] / 3.0)
        out[f"{prefix}_return_acceleration_3v5"] = (
            x[f"{prefix}_return_3m"] / 3.0 - x[f"{prefix}_return_5m"] / 5.0)
        sign1, sign3, sign5 = (np.sign(x[f"{prefix}_return_{h}m"]) for h in (1, 3, 5))
        out[f"{prefix}_momentum_persistence"] = sign1.where(
            sign1.eq(sign3) & sign3.eq(sign5), 0)
        out[f"{prefix}_short_term_reversal"] = sign1.mul(sign3).lt(0).astype("int8")
    out["ce_pe_acceleration_spread"] = (
        out.ce_return_acceleration_1v3 - out.pe_return_acceleration_1v3)
    out["ce_pe_relative_volatility"] = _safe_balance(x.ce_volatility, x.pe_volatility)
    out["ce_pe_atr_expansion_balance"] = _safe_balance(x.ce_atr_pct, x.pe_atr_pct)
    out["ce_pe_volume_imbalance"] = _safe_balance(x.ce_volume_ratio, x.pe_volume_ratio)
    out["nifty_option_directional_response_1m"] = np.sign(x.nifty_return_1m) * (
        x.ce_return_1m - x.pe_return_1m)
    out["nifty_option_directional_response_3m"] = np.sign(x.nifty_return_3m) * (
        x.ce_return_3m - x.pe_return_3m)
    out["bullish_response_strength"] = (
        x.ce_return_3m - x.pe_return_3m).where(x.nifty_return_3m.gt(0), 0)
    out["bearish_response_strength"] = (
        x.pe_return_3m - x.ce_return_3m).where(x.nifty_return_3m.lt(0), 0)
    return out


def range_features() -> pd.DataFrame:
    rows = []
    roots = [
        (ROOT / "data/normalized/historical_options/validated20", "EXPIRY_DAY"),
        (ROOT / "data/normalized/historical_options/non_expiry15", "NON_EXPIRY_DAY"),
    ]
    for root, day_type in roots:
        for session in sorted(path for path in root.iterdir() if path.is_dir()):
            ce, pe = pd.read_parquet(session / "ce.parquet"), pd.read_parquet(session / "pe.parquet")
            if day_type == "EXPIRY_DAY":
                date, archive = session.name, ROOT / "data/external/shoonyatrader/validated20_raw" / f"{session.name.replace('-', '')}.zip"
            else:
                manifest = json.loads((session / "manifest.json").read_text())
                date = manifest["date"]
                archive = ROOT / "data/external/shoonyatrader/validated20_raw" / manifest["source_archive"]
            panel = load_selected_contract_panel(archive, ce, pe, session_date=date)
            result = pd.DataFrame({"timestamp": pd.to_datetime(ce.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")})
            for side, selected in (("ce", ce), ("pe", pe)):
                selected = selected.copy()
                selected["timestamp"] = pd.to_datetime(selected.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
                panel_side = panel.loc[panel.option_type.eq(side.upper())].copy()
                feature_rows = []
                for _, contract in panel_side.groupby("contract_segment_id", sort=False):
                    contract = contract.sort_values("timestamp", kind="stable").copy()
                    prior_high = contract.high.shift(1).rolling(20, min_periods=20).max()
                    prior_low = contract.low.shift(1).rolling(20, min_periods=20).min()
                    width = (prior_high - prior_low).replace(0, np.nan)
                    feature_rows.append(pd.DataFrame({
                        "timestamp": contract.timestamp, "expiry": contract.expiry,
                        "strike": contract.strike,
                        f"{side}_recent_range_position": (contract.close - prior_low) / width,
                        f"{side}_distance_recent_high_pct": (contract.close / prior_high - 1) * 100,
                        f"{side}_distance_recent_low_pct": (contract.close / prior_low - 1) * 100,
                    }))
                available = pd.concat(feature_rows, ignore_index=True)
                keys = selected[["timestamp", "expiry", "strike"]]
                chosen = keys.merge(available, on=["timestamp", "expiry", "strike"], how="left",
                                    validate="one_to_one")
                result = result.merge(chosen.drop(columns=["expiry", "strike"]), on="timestamp",
                                      how="left", validate="one_to_one")
            rows.append(result)
    return pd.concat(rows, ignore_index=True)


def evaluate_matrix(data: pd.DataFrame, names: list[str], label: str) -> tuple[dict, pd.DataFrame]:
    fold_rows, predictions = [], []
    for fold_id, (train_dates, test_dates) in enumerate(folds(data), start=1):
        train, test = data.loc[data.session_date.isin(train_dates)], data.loc[data.session_date.isin(test_dates)]
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=1.0, max_iter=2000, random_state=42))
        model.fit(train[names], train.label)
        probability = model.predict_proba(test[names])[:, 1]
        auc = float(roc_auc_score(test.label, probability))
        fold_rows.append({"matrix": label, "fold": fold_id, "train_sessions": len(train_dates),
                          "test_sessions": len(test_dates), "test_rows": len(test), "auc": auc})
        part = test[["timestamp", "session_date", "option_type", "expiry_day_status",
                     "trend_regime", "label"]].copy()
        part["matrix"], part["fold"], part["probability"] = label, fold_id, probability
        predictions.append(part)
    fold_frame = pd.DataFrame(fold_rows)
    pred = pd.concat(predictions, ignore_index=True)
    group_rows = []
    for column in ("option_type", "expiry_day_status", "trend_regime"):
        for value, group in pred.groupby(column, sort=False):
            if group.label.nunique() < 2:
                continue
            group_rows.append({"matrix": label, "dimension": column, "group": value,
                               "rows": len(group), "auc": float(roc_auc_score(group.label, group.probability))})
    summary = {"matrix": label, "features": len(names), "mean_auc": float(fold_frame.auc.mean()),
               "minimum_auc": float(fold_frame.auc.min()),
               "positive_folds": int(fold_frame.auc.gt(.5).sum())}
    return summary, pd.concat([fold_frame, pd.DataFrame(group_rows)], ignore_index=True, sort=False)


def main() -> None:
    base = pd.read_parquet(SOURCE)
    base = base.merge(range_features(), on="timestamp", how="left", validate="one_to_one")
    proposed = relationship_features(base)
    range_names = [name for name in base.columns if "recent_" in name]
    for name in range_names:
        proposed[name] = base[name]
    proposed_names = proposed.columns.tolist()
    labels = pd.concat([
        pd.read_parquet(ROOT / "reports/phase9_label_behavior_audit/trade_path_audit.parquet"),
        pd.read_parquet(ROOT / "reports/phase10_label_suitability/non_expiry_trade_path_audit.parquet"),
    ], ignore_index=True)
    labels = labels.loc[labels.actual_exit_reason.isin(["WIN", "LOSS"])].sort_values(
        "timestamp").drop_duplicates(["timestamp", "expiry", "strike", "option_type"])
    labels["label"] = labels.actual_exit_reason.eq("WIN").astype("int8")
    meta = ["timestamp", "session_date", "expiry_day_status", "option_type", "trend_regime", "label"]
    feature_frame = pd.concat([base[["timestamp", *EXISTING]], proposed], axis=1)
    data = labels[meta].merge(feature_frame, on="timestamp", how="left", validate="many_to_one")
    matrices = {
        "A_EXISTING_48": EXISTING,
        "B_RELATIONSHIPS_ONLY": proposed_names,
        "C_EXISTING_PLUS_RELATIONSHIPS": [*EXISTING, *proposed_names],
        "D_DEDUPED_EXISTING_PLUS_RELATIONSHIPS": [
            *[name for name in EXISTING if name not in REDUNDANT_ALIASES], *proposed_names],
    }
    summaries, details = [], []
    for label, names in matrices.items():
        summary, detail = evaluate_matrix(data, names, label)
        summaries.append(summary)
        details.append(detail)
    summary_frame = pd.DataFrame(summaries)
    detail_frame = pd.concat(details, ignore_index=True)
    proposed_ranking = rank_features(data, proposed_names, "PROPOSED")
    formulas = proposed_formulas()
    coverage = []
    for name in proposed_names:
        finite = pd.to_numeric(data[name], errors="coerce").replace([np.inf, -np.inf], np.nan).notna()
        coverage.append({"feature": name, "formula": formulas[name],
                         "source": "entry-time NIFTY/CE/PE features or fixed-contract OHLCV",
                         "coverage_pct": float(finite.mean() * 100), "causal": True,
                         "contract_segment_safe": True})
    coverage = pd.DataFrame(coverage).merge(
        proposed_ranking[["feature", "mean_oos_auc", "minimum_oos_auc", "positive_oos_folds"]],
        on="feature", how="left")
    base_result = summary_frame.loc[summary_frame.matrix.eq("A_EXISTING_48")].iloc[0]
    combined = summary_frame.loc[summary_frame.matrix.eq(
        "D_DEDUPED_EXISTING_PLUS_RELATIONSHIPS")].iloc[0]
    groups = detail_frame.loc[
        detail_frame.matrix.eq("D_DEDUPED_EXISTING_PLUS_RELATIONSHIPS")
        & detail_frame.dimension.notna()]
    stable_groups = bool(len(groups) and groups.auc.ge(.52).all())
    meaningful = bool(combined.mean_auc >= base_result.mean_auc + .02
                      and combined.minimum_auc >= base_result.minimum_auc - .01
                      and combined.positive_folds >= 4 and stable_groups)
    result = {
        "status": "RELATIONSHIP_FEATURE_AUDIT_COMPLETE",
        "verdict": ("MISSING_RELATIONSHIPS_ADD_STABLE_SIGNAL" if meaningful else
                    "PREDICTIVE_SIGNAL_REMAINS_WEAK_NO_EXPANSION_JUSTIFIED"),
        "relationship_features_added_to_phase6": False,
        "proposed_feature_count": len(proposed_names),
        "ablation": summaries, "base_to_best_gain": float(combined.mean_auc - base_result.mean_auc),
        "all_group_auc_at_least_052": stable_groups,
        "catboost_trained": False, "production_changed": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    category_table = pd.DataFrame({"feature": EXISTING,
                                   "category": [existing_category(name) for name in EXISTING]})
    correlations = base[EXISTING].corr(min_periods=100)
    redundant = []
    for index, left in enumerate(EXISTING):
        for right in EXISTING[index + 1:]:
            value = correlations.loc[left, right]
            if pd.notna(value) and abs(value) >= .95:
                redundant.append({"feature_a": left, "feature_b": right,
                                  "correlation": float(value)})
    category_table.to_csv(OUTPUT / "existing_feature_categories.csv", index=False)
    pd.DataFrame(redundant).to_csv(OUTPUT / "existing_redundancy.csv", index=False)
    coverage.to_csv(OUTPUT / "proposed_feature_audit.csv", index=False)
    summary_frame.to_csv(OUTPUT / "ablation_summary.csv", index=False)
    detail_frame.to_csv(OUTPUT / "ablation_folds_and_groups.csv", index=False)
    proposed_ranking.to_csv(OUTPUT / "proposed_univariate_ranking.csv", index=False)
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
