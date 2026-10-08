"""Research-only signal discovery for the fixed 5-minute opportunity label."""

from __future__ import annotations

from bisect import bisect_right, insort
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_phase12_feature_signal_audit import folds  # noqa: E402
from tools.run_risk_adjusted_catboost_research import FEATURES, load_data  # noqa: E402


OUTPUT = ROOT / "reports/phase15_signal_discovery"
MIN_GROUP_ROWS, MIN_GROUP_CLASS = 100, 20
BASELINE_AUC = 0.5374099284263469

FAMILIES = {
    "NIFTY_TREND": ["nifty_ema_structure", "bullish_option_confirmation", "bearish_option_confirmation"],
    "NIFTY_MOMENTUM": [f"nifty_{kind}_{h}m" for kind in ("return", "momentum") for h in (1, 3, 5)],
    "NIFTY_VOLATILITY": ["nifty_atr_pct", "nifty_volatility"],
    "CE_MOMENTUM": [f"ce_{kind}_{h}m" for kind in ("return", "momentum") for h in (1, 3, 5)],
    "PE_MOMENTUM": [f"pe_{kind}_{h}m" for kind in ("return", "momentum") for h in (1, 3, 5)],
    "CE_VOLATILITY": ["ce_atr_pct", "ce_volatility"],
    "PE_VOLATILITY": ["pe_atr_pct", "pe_volatility"],
    "CE_PE_RELATIVE_STRENGTH": [
        "ce_return_5m_minus_nifty_return_5m", "pe_return_5m_minus_nifty_return_5m",
        "pe_return_5m_plus_nifty_return_5m", "ce_return_5m_minus_pe_return_5m",
        "ce_momentum_5m_minus_pe_momentum_5m", "ce_strength_vs_nifty",
        "pe_strength_vs_nifty", "ce_strength_vs_pe", "pe_strength_vs_ce"],
    "PREMIUM_ACCELERATION": ["ce_premium_acceleration", "pe_premium_acceleration"],
    "EMA_STRUCTURE": ["nifty_ema_structure", "ce_ema_structure", "pe_ema_structure"],
    "RSI": ["nifty_rsi", "ce_rsi", "pe_rsi"],
    "ATR": ["nifty_atr_pct", "ce_atr_pct", "pe_atr_pct"],
    "VOLUME": ["ce_volume_ratio", "pe_volume_ratio", "ce_volume_ratio_minus_pe_volume_ratio"],
    "REGIME_CONTEXT": ["bullish_option_confirmation", "bearish_option_confirmation"],
}


def add_interactions(data: pd.DataFrame) -> dict[str, list[str]]:
    specs = {
        "NIFTY_X_CE_MOMENTUM": data.nifty_momentum_5m * data.ce_momentum_5m,
        "NIFTY_X_PE_MOMENTUM": data.nifty_momentum_5m * data.pe_momentum_5m,
        "CE_MINUS_PE_MOMENTUM": data.ce_momentum_5m - data.pe_momentum_5m,
        "CE_MINUS_PE_ACCELERATION": data.ce_premium_acceleration - data.pe_premium_acceleration,
        "CE_TO_PE_VOLATILITY": data.ce_volatility / data.pe_volatility.replace(0, np.nan),
        "CE_MOMENTUM_X_VOLATILITY": data.ce_momentum_5m * data.ce_volatility,
        "PE_MOMENTUM_X_VOLATILITY": data.pe_momentum_5m * data.pe_volatility,
        "TREND_X_NIFTY_MOMENTUM": data.nifty_ema_structure * data.nifty_momentum_5m,
        "CE_RSI_X_MOMENTUM": (data.ce_rsi - 50) * data.ce_momentum_5m,
        "PE_RSI_X_MOMENTUM": (data.pe_rsi - 50) * data.pe_momentum_5m,
        "BULLISH_RESPONSE_X_NIFTY": data.ce_strength_vs_pe * data.nifty_momentum_5m,
        "BEARISH_RESPONSE_X_NIFTY": data.pe_strength_vs_ce * -data.nifty_momentum_5m,
    }
    columns = {}
    for name, values in specs.items():
        column = f"interaction__{name.lower()}"
        data[column] = values.replace([np.inf, -np.inf], np.nan)
        columns[name] = [column]
    return columns


def causal_percentile(values: pd.Series, sessions: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    for _, indexes in sessions.groupby(sessions, sort=False).groups.items():
        history: list[float] = []
        for index in indexes:
            value = values.loc[index]
            if pd.isna(value):
                continue
            insort(history, float(value))
            result.loc[index] = bisect_right(history, float(value)) / len(history)
    return result


def add_rank_features(data: pd.DataFrame) -> list[str]:
    raw = ["nifty_momentum_5m", "ce_momentum_5m", "pe_momentum_5m", "ce_volatility",
           "pe_volatility", "ce_premium_acceleration", "pe_premium_acceleration"]
    columns = []
    for source in raw:
        column = f"rank__{source}"
        data[column] = causal_percentile(data[source], data.session_date)
        columns.append(column)
    data["rank__ce_vs_pe_momentum"] = np.where(
        data.ce_momentum_5m > data.pe_momentum_5m, 1.0,
        np.where(data.ce_momentum_5m < data.pe_momentum_5m, 0.0, 0.5))
    data["rank__ce_vs_pe_volatility"] = np.where(
        data.ce_volatility > data.pe_volatility, 1.0,
        np.where(data.ce_volatility < data.pe_volatility, 0.0, 0.5))
    return columns + ["rank__ce_vs_pe_momentum", "rank__ce_vs_pe_volatility"]


def evaluate(data: pd.DataFrame, name: str, kind: str, columns: list[str]) -> tuple[dict, list[dict], list[dict]]:
    fold_rows, predictions = [], []
    for fold_id, (train_dates, test_dates) in enumerate(folds(data), start=1):
        train = data.loc[data.session_date.isin(train_dates)]
        test = data.loc[data.session_date.isin(test_dates)]
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                                                 random_state=42))
        model.fit(train[columns], train.label)
        probability = model.predict_proba(test[columns])[:, 1]
        auc = float(roc_auc_score(test.label, probability))
        fold_rows.append({"candidate_type": kind, "candidate": name, "fold": fold_id,
                          "train_rows": len(train), "test_rows": len(test), "roc_auc": auc,
                          "pr_auc": float(average_precision_score(test.label, probability))})
        part = test[["option_type", "expiry_day_status", "trend_regime", "label"]].copy()
        part["probability"] = probability
        predictions.append(part)
    fold_frame = pd.DataFrame(fold_rows)
    oos = pd.concat(predictions, ignore_index=True)
    group_rows = []
    for dimension in ("option_type", "expiry_day_status", "trend_regime"):
        for value, group in oos.groupby(dimension, sort=False):
            positive, negative = int(group.label.sum()), int(group.label.eq(0).sum())
            eligible = len(group) >= MIN_GROUP_ROWS and min(positive, negative) >= MIN_GROUP_CLASS
            group_rows.append({"candidate_type": kind, "candidate": name, "dimension": dimension,
                               "group": value, "rows": len(group), "positive": positive,
                               "negative": negative, "sample_gate": eligible,
                               "roc_auc": float(roc_auc_score(group.label, group.probability))
                               if eligible else np.nan})
    group_frame = pd.DataFrame(group_rows)
    valid_groups = group_frame.loc[group_frame.sample_gate]
    side_min = valid_groups.loc[valid_groups.dimension.eq("option_type"), "roc_auc"].min()
    expiry_min = valid_groups.loc[valid_groups.dimension.eq("expiry_day_status"), "roc_auc"].min()
    regime_min = valid_groups.loc[valid_groups.dimension.eq("trend_regime"), "roc_auc"].min()
    mean_auc, median_auc, minimum_auc = (float(fold_frame.roc_auc.mean()),
                                         float(fold_frame.roc_auc.median()),
                                         float(fold_frame.roc_auc.min()))
    row = {"candidate_type": kind, "candidate": name, "feature_count": len(columns),
           "features": "|".join(columns), "mean_oos_auc": mean_auc,
           "median_oos_auc": median_auc, "minimum_fold_auc": minimum_auc,
           "positive_folds": int(fold_frame.roc_auc.gt(0.5).sum()),
           "ce_pe_min_auc": float(side_min), "expiry_min_auc": float(expiry_min),
           "regime_min_auc": float(regime_min), "auc_improvement_vs_logistic": mean_auc - BASELINE_AUC}
    row["passes_gate"] = bool(mean_auc >= 0.55 and median_auc >= 0.55
                              and minimum_auc >= 0.52 and row["positive_folds"] >= 4
                              and side_min >= 0.52 and expiry_min >= 0.52 and regime_min >= 0.48
                              and row["auc_improvement_vs_logistic"] >= 0.02)
    return row, fold_rows, group_rows


def bucket_analysis(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold_id, (train_dates, test_dates) in enumerate(folds(data), start=1):
        train, test = data[data.session_date.isin(train_dates)], data[data.session_date.isin(test_dates)]
        for feature in FEATURES:
            edges = np.unique(train[feature].quantile(np.linspace(0, 1, 6)).to_numpy())
            if len(edges) < 3:
                continue
            edges[0], edges[-1] = -np.inf, np.inf
            buckets = pd.cut(test[feature], bins=edges, labels=False, include_lowest=True)
            framed = test.assign(bucket=buckets)
            for regime, group in [("ALL", framed), *list(framed.groupby("trend_regime"))]:
                for bucket, part in group.groupby("bucket", observed=True):
                    if len(part) < 5:
                        continue
                    rows.append({"feature": feature, "fold": fold_id, "regime": regime,
                                 "bucket": int(bucket), "rows": len(part),
                                 "win_probability": float(part.label.mean()),
                                 "mean_forward_return_5m": float(part.selected_return_5m_pct.mean())})
    return pd.DataFrame(rows)


def main() -> None:
    data, leakage = load_data()
    interactions = add_interactions(data)
    rank_columns = add_rank_features(data)
    candidates = [("BASELINE", "ALL_48_LOGISTIC", FEATURES)]
    candidates.extend(("FAMILY", name, columns) for name, columns in FAMILIES.items())
    candidates.extend(("INTERACTION", name, columns) for name, columns in interactions.items())
    candidates.append(("RANK", "CAUSAL_PERCENTILES", rank_columns))

    rankings, fold_rows, regime_rows = [], [], []
    for kind, name, columns in candidates:
        row, candidate_folds, candidate_groups = evaluate(data, name, kind, columns)
        rankings.append(row); fold_rows.extend(candidate_folds); regime_rows.extend(candidate_groups)
    ranking = pd.DataFrame(rankings).sort_values("mean_oos_auc", ascending=False, kind="stable")
    family = ranking[ranking.candidate_type.eq("FAMILY")]
    interaction = ranking[ranking.candidate_type.eq("INTERACTION")]
    buckets = bucket_analysis(data)
    monotonic = []
    overall_buckets = buckets[buckets.regime.eq("ALL")]
    for feature, group in overall_buckets.groupby("feature"):
        fold_correlations = []
        for _, fold_group in group.groupby("fold"):
            aggregated = fold_group.groupby("bucket").agg(
                win_probability=("win_probability", "mean"), rows=("rows", "sum")).reset_index()
            if len(aggregated) >= 3:
                fold_correlations.append(float(spearmanr(
                    aggregated.bucket, aggregated.win_probability).statistic))
        monotonic.append({"feature": feature, "folds": len(fold_correlations),
                          "mean_bucket_spearman": float(np.mean(fold_correlations)),
                          "consistent_direction_folds": int(max(sum(x > 0 for x in fold_correlations),
                                                                sum(x < 0 for x in fold_correlations)))})
    monotonic_frame = pd.DataFrame(monotonic)
    passed = ranking[ranking.passes_gate]
    verdict = "STABLE_SIGNAL_FOUND" if len(passed) else "NO_STABLE_SIGNAL_FOUND"
    result = {
        "status": "PHASE15_SIGNAL_DISCOVERY_COMPLETE", "verdict": verdict,
        "observations": len(data), "sessions": int(data.session_date.nunique()),
        "features": len(FEATURES), "families_tested": len(FAMILIES),
        "interactions_tested": len(interactions), "rank_features_tested": len(rank_columns),
        "protocol": "five expanding session folds; one-session embargo; OOS only",
        "minimum_subgroup": {"rows": MIN_GROUP_ROWS, "each_class": MIN_GROUP_CLASS},
        "success_gate": {"mean_auc": 0.55, "median_auc": 0.55, "minimum_fold_auc": 0.52,
                         "positive_folds": 4, "ce_pe_min_auc": 0.52,
                         "expiry_min_auc": 0.52, "regime_min_auc": 0.48,
                         "improvement_vs_logistic": 0.02},
        "best_candidate": ranking.iloc[0].replace({np.nan: None}).to_dict(),
        "passing_candidates": passed.candidate.tolist(),
        "stable_monotonic_features": monotonic_frame.loc[
            monotonic_frame.consistent_direction_folds.ge(4)
            & monotonic_frame.mean_bucket_spearman.abs().ge(0.30)].feature.tolist(),
        "leakage_audit": leakage,
        "catboost_tuned": False, "label_changed": False, "production_changed": False,
        "strategy_selected": False, "threshold_tuned": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    family.to_csv(OUTPUT / "feature_family_ranking.csv", index=False)
    interaction.to_csv(OUTPUT / "feature_interaction_ranking.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(OUTPUT / "regime_signal_ranking.csv", index=False)
    buckets.to_csv(OUTPUT / "feature_bucket_analysis.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUTPUT / "fold_results.csv", index=False)
    ranking.to_csv(OUTPUT / "final_signal_candidates.csv", index=False)
    monotonic_frame.to_csv(OUTPUT / "monotonic_feature_summary.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
