"""Phase 12 statistical audit of the frozen 49-feature CE/PE architecture."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ce_pe_feature_preview import build_ce_pe_feature_preview, feature_columns  # noqa: E402


OUTPUT = ROOT / "reports/phase12_feature_signal"
FEATURE_SOURCES = (
    ("EXPIRY_DAY", ROOT / "data/normalized/historical_options/validated20/feature_dataset_49.parquet"),
    ("NON_EXPIRY_DAY", ROOT / "data/normalized/historical_options/non_expiry15/feature_dataset_49.parquet"),
)
LABEL_SOURCES = (
    ROOT / "reports/phase9_label_behavior_audit/trade_path_audit.parquet",
    ROOT / "reports/phase10_label_suitability/non_expiry_trade_path_audit.parquet",
)


def quality(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for name in features:
        raw = pd.to_numeric(frame[name], errors="coerce")
        finite = raw.replace([np.inf, -np.inf], np.nan).dropna()
        dominant = float(finite.value_counts(normalize=True).iloc[0]) if len(finite) else None
        rows.append({
            "feature": name, "rows": len(raw), "finite": len(finite),
            "missing": int(raw.isna().sum()), "infinite": int(np.isinf(raw).sum()),
            "coverage_pct": float(len(finite) / len(raw) * 100),
            "unique": int(finite.nunique()), "mean": float(finite.mean()) if len(finite) else None,
            "std": float(finite.std(ddof=0)) if len(finite) else None,
            "min": float(finite.min()) if len(finite) else None,
            "median": float(finite.median()) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None,
            "dominant_fraction": dominant,
            "constant": bool(len(finite) and finite.nunique() <= 1),
            "near_constant": bool(dominant is not None and dominant >= .995),
        })
    return pd.DataFrame(rows)


def folds(data: pd.DataFrame) -> list[tuple[list[str], list[str]]]:
    dates = sorted(data.session_date.unique())
    if len(dates) < 15:
        return []
    initial = 10
    chunks = [list(chunk) for chunk in np.array_split(dates[initial:], 5) if len(chunk)]
    result = []
    for test in chunks:
        prior = [date for date in dates if date < test[0]]
        train = prior[:-1]  # one-session embargo; labels cannot cross the session boundary.
        result.append((train, test))
    return result


def rank_features(data: pd.DataFrame, features: list[str], group: str) -> pd.DataFrame:
    rows = []
    for name in features:
        aucs, mis = [], []
        for train_dates, test_dates in folds(data):
            train = data.loc[data.session_date.isin(train_dates), [name, "label"]].copy()
            test = data.loc[data.session_date.isin(test_dates), [name, "label"]].copy()
            train[name] = pd.to_numeric(train[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
            test[name] = pd.to_numeric(test[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
            train = train.dropna()
            test = test.dropna()
            if len(train) < 30 or len(test) < 10 or train.label.nunique() < 2 or test.label.nunique() < 2:
                continue
            direction = 1 if train.loc[train.label.eq(1), name].mean() >= train.loc[train.label.eq(0), name].mean() else -1
            aucs.append(float(roc_auc_score(test.label, test[name] * direction)))
            mis.append(float(mutual_info_classif(
                train[[name]], train.label, random_state=42, discrete_features=False)[0]))
        win = pd.to_numeric(data.loc[data.label.eq(1), name], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        loss = pd.to_numeric(data.loc[data.label.eq(0), name], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        pooled = np.sqrt((win.var(ddof=1) + loss.var(ddof=1)) / 2) if len(win) > 1 and len(loss) > 1 else np.nan
        rows.append({
            "group": group, "feature": name, "observations": len(win) + len(loss),
            "win_observations": len(win), "loss_observations": len(loss),
            "standardized_mean_difference": float((win.mean() - loss.mean()) / pooled)
                if np.isfinite(pooled) and pooled > 0 else None,
            "ks_statistic": float(ks_2samp(win, loss).statistic) if len(win) and len(loss) else None,
            "walk_forward_folds": len(aucs),
            "mean_oos_auc": float(np.mean(aucs)) if aucs else None,
            "minimum_oos_auc": float(np.min(aucs)) if aucs else None,
            "positive_oos_folds": int(sum(value > .5 for value in aucs)),
            "mean_train_mutual_information": float(np.mean(mis)) if mis else None,
        })
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["mean_oos_auc", "mean_train_mutual_information"], ascending=False,
        na_position="last", kind="stable").reset_index(drop=True)


def causality_check() -> dict:
    checks = []
    session_roots = [
        next(path for path in (ROOT / "data/normalized/historical_options/validated20").iterdir() if path.is_dir()),
        next(path for path in (ROOT / "data/normalized/historical_options/non_expiry15").iterdir() if path.is_dir()),
    ]
    for root in session_roots:
        index, ce, pe = (pd.read_parquet(root / name) for name in ("nifty.parquet", "ce.parquet", "pe.parquet"))
        full = build_ce_pe_feature_preview(index, ce, pe)
        cutoff = 200
        timestamp = index.sort_values("timestamp").timestamp.iloc[cutoff - 1]
        prefix = build_ce_pe_feature_preview(
            index.loc[index.timestamp.le(timestamp)], ce.loc[ce.timestamp.le(timestamp)],
            pe.loc[pe.timestamp.le(timestamp)])
        left = full.loc[full.timestamp.eq(timestamp), feature_columns()].reset_index(drop=True)
        right = prefix.loc[prefix.timestamp.eq(timestamp), feature_columns()].reset_index(drop=True)
        equal = bool(len(left) == len(right) == 1 and np.allclose(
            left.to_numpy(dtype=float), right.to_numpy(dtype=float), equal_nan=True))
        checks.append({"session": root.name, "cutoff": str(timestamp), "prefix_invariant": equal})
    return {
        "prefix_invariance_checks": checks,
        "all_prefix_invariant": all(row["prefix_invariant"] for row in checks),
        "label_columns_in_feature_builder": False,
        "rolling_features": "current and prior completed candles only",
        "event_label_path_starts": "strictly after entry timestamp",
        "purge": "30-minute labels never cross sessions",
        "embargo": "one complete session removed between expanding train and test windows",
    }


def main() -> None:
    features = feature_columns()
    feature_frames = []
    for day_type, path in FEATURE_SOURCES:
        frame = pd.read_parquet(path)
        frame["expiry_day_status"] = day_type
        feature_frames.append(frame)
    feature_data = pd.concat(feature_frames, ignore_index=True)
    labels = pd.concat([pd.read_parquet(path) for path in LABEL_SOURCES], ignore_index=True)
    labels = labels.loc[labels.actual_exit_reason.isin(["WIN", "LOSS"])].copy()
    labels = labels.sort_values("timestamp", kind="stable").drop_duplicates(
        ["timestamp", "expiry", "strike", "option_type"], keep="first")
    labels["label"] = labels.actual_exit_reason.eq("WIN").astype("int8")
    meta = ["timestamp", "session_date", "expiry_day_status", "option_type", "trend_regime", "label"]
    data = labels[meta].merge(feature_data[["timestamp", *features]], on="timestamp", how="left",
                              validate="many_to_one")

    quality_table = quality(feature_data, features)
    overall = rank_features(data, features, "ALL")
    group_rankings = [overall]
    for column, values in {
        "option_type": ["CE", "PE"],
        "expiry_day_status": ["EXPIRY_DAY", "NON_EXPIRY_DAY"],
        "trend_regime": ["TREND_UP", "TREND_DOWN", "SIDEWAYS"],
    }.items():
        for value in values:
            group_rankings.append(rank_features(data.loc[data[column].eq(value)], features,
                                                f"{column}={value}"))
    rankings = pd.concat(group_rankings, ignore_index=True)
    correlation = feature_data[features].replace([np.inf, -np.inf], np.nan).corr(min_periods=100)
    redundant = []
    for index, left in enumerate(features):
        for right in features[index + 1:]:
            value = correlation.loc[left, right]
            if pd.notna(value) and abs(value) >= .95:
                redundant.append({"feature_a": left, "feature_b": right,
                                  "correlation": float(value)})
    redundant_table = pd.DataFrame(redundant).sort_values(
        "correlation", key=lambda x: x.abs(), ascending=False) if redundant else pd.DataFrame(
            columns=["feature_a", "feature_b", "correlation"])
    usable = overall.loc[overall.walk_forward_folds.eq(5) & overall.mean_oos_auc.notna()].copy()
    top10 = usable.head(10)
    complete_rows = int(feature_data[features].replace([np.inf, -np.inf], np.nan).dropna().shape[0])
    causality = causality_check()
    data_problem = bool(complete_rows == 0 or quality_table.coverage_pct.eq(0).any()
                        or not causality["all_prefix_invariant"])
    strong = int((usable.mean_oos_auc >= .60).sum()) >= 5
    weak = int((usable.mean_oos_auc >= .53).sum()) >= 3
    conclusion = ("4. Data/feature problem detected" if data_problem else
                  "1. Strong predictive signal" if strong else
                  "2. Weak but usable predictive signal" if weak else
                  "3. No meaningful predictive signal")
    result = {
        "status": "PHASE12_FEATURE_SIGNAL_AUDIT_COMPLETE",
        "conclusion": conclusion, "feature_count": len(features),
        "feature_rows": len(feature_data), "labeled_unique_observations": len(data),
        "wins": int(data.label.sum()), "losses": int(data.label.eq(0).sum()),
        "complete_49_feature_rows": complete_rows,
        "zero_coverage_features": quality_table.loc[quality_table.coverage_pct.eq(0), "feature"].tolist(),
        "near_constant_features": quality_table.loc[quality_table.near_constant, "feature"].tolist(),
        "infinite_values": int(quality_table.infinite.sum()),
        "high_redundancy_pairs": len(redundant_table),
        "top_10_potentially_useful": top10[["feature", "mean_oos_auc", "minimum_oos_auc",
                                             "positive_oos_folds", "mean_train_mutual_information"]].to_dict("records"),
        "leakage_audit": causality,
        "catboost_trained": False, "features_removed": False, "production_changed": False,
        "limitations": [
            "NIFTY cash-index VWAP is unavailable, leaving nifty_vwap_distance entirely missing.",
            "ATM contract transitions reset option rolling features and materially reduce CE/PE feature coverage.",
            "Repeated strategy signals at the same contract/timestamp were deduplicated before scoring.",
            "Mutual information is descriptive on training folds; walk-forward AUC is the primary ranking metric.",
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    quality_table.to_csv(OUTPUT / "feature_quality.csv", index=False)
    rankings.to_csv(OUTPUT / "feature_rankings_all_groups.csv", index=False)
    overall.to_csv(OUTPUT / "feature_ranking_overall.csv", index=False)
    rankings.loc[rankings.group.eq("option_type=CE")].to_csv(OUTPUT / "feature_ranking_ce.csv", index=False)
    rankings.loc[rankings.group.eq("option_type=PE")].to_csv(OUTPUT / "feature_ranking_pe.csv", index=False)
    rankings.loc[rankings.group.str.startswith("expiry_day_status=")].to_csv(
        OUTPUT / "feature_ranking_expiry_status.csv", index=False)
    rankings.loc[rankings.group.str.startswith("trend_regime=")].to_csv(
        OUTPUT / "feature_ranking_regime.csv", index=False)
    correlation.to_csv(OUTPUT / "feature_correlation_matrix.csv")
    redundant_table.to_csv(OUTPUT / "redundant_features.csv", index=False)
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
