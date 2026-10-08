"""Leakage-safe CALL/PUT paper-candidate training across causal setup families."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from src.model_benchmark import BenchmarkProtocol, build_pipeline
from src.option_feature_pipeline import OPTION_FEATURES, OPTION_FEATURE_VERSION, build_option_features
from src.stage1_v3_research import build_candidate_setups


STRATEGIES = ("PULLBACK_EMA20", "TREND_CONTINUATION", "MTF15_CONTINUATION")
TARGET_PCT = 0.05
HORIZON = 15
THRESHOLDS = tuple(np.arange(.50, .91, .05).round(2))


def _targets(option: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    option = option.copy().sort_values("timestamp", kind="stable").reset_index(drop=True)
    option["timestamp"] = pd.to_datetime(option.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    option["session"] = option.timestamp.dt.strftime("%Y-%m-%d")
    locations = pd.Series(option.index, index=option.timestamp)
    rows = []
    for timestamp in pd.DatetimeIndex(candidates.timestamp):
        if timestamp not in locations:
            continue
        location = int(locations.loc[timestamp])
        entry = option.iloc[location]
        future = option.iloc[location + 1:location + HORIZON + 1]
        future = future.loc[future.session.eq(entry.session)]
        if "strike" in option and pd.notna(entry.get("strike")):
            future = future.loc[future.strike.eq(entry.strike)]
        if future.empty:
            continue
        rows.append({
            "timestamp": timestamp,
            "entry_premium": float(entry.close),
            "target": int(float(future.high.max()) >= float(entry.close) * (1 + TARGET_PCT)),
        })
    return pd.DataFrame(rows)


def build_dataset(features: pd.DataFrame, option: pd.DataFrame, strategy: str,
                  direction: int) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    candidates = build_candidate_setups(features)[strategy][1]
    selected = candidates.loc[candidates.entry_signal.eq(direction), ["timestamp"]]
    labels = _targets(option, selected)
    option_features = build_option_features(option).rename(
        columns={name: f"option_{name}" for name in OPTION_FEATURES})
    base = features.merge(selected, on="timestamp", how="inner", validate="one_to_one")
    base = base.merge(option_features[["timestamp", *[f"option_{x}" for x in OPTION_FEATURES]]],
                      on="timestamp", how="inner", validate="one_to_one")
    merged = base.merge(labels, on="timestamp", how="inner", validate="one_to_one").sort_values("timestamp")
    metadata = merged[["timestamp", "entry_premium"]].copy()
    y = merged.pop("target").astype("int8")
    forbidden = {"timestamp", "source_row", "volume_raw", "session_id", "entry_premium"}
    X = merged.drop(columns=[name for name in forbidden if name in merged])
    X["strategy"] = strategy
    return X, y, metadata


def _threshold_table(y: pd.Series, probability: np.ndarray, sessions: int) -> list[dict]:
    rows = []
    for threshold in THRESHOLDS:
        prediction = probability >= threshold
        rows.append({
            "threshold": threshold, "signals": int(prediction.sum()),
            "signals_per_session": float(prediction.sum() / max(sessions, 1)),
            "precision": float(precision_score(y, prediction, zero_division=0)),
            "recall": float(recall_score(y, prediction, zero_division=0)),
        })
    return rows


def train_candidate(features: pd.DataFrame, option: pd.DataFrame, strategy: str,
                    direction: int, output: Path) -> dict:
    required_contract = {"strike", "option_type", "expiry_code"}
    missing_contract = sorted(required_contract.difference(option.columns))
    if missing_contract:
        raise ValueError(
            f"Option training data lacks contract identity: {missing_contract}. "
            "Synthetic or inferred strikes are not accepted for model validation.")
    X, y, metadata = build_dataset(features, option, strategy, direction)
    train_end, calibration_end = int(len(X) * .60), int(len(X) * .80)
    if train_end < 500 or len(X) - calibration_end < 100:
        raise ValueError(f"Insufficient chronological rows for {strategy}")
    protocol = BenchmarkProtocol(n_splits=4, threshold=.5, random_state=42)
    evaluation = build_pipeline("CATBOOST", X.iloc[:train_end], protocol)
    evaluation.fit(X.iloc[:train_end], y.iloc[:train_end])
    calibration_probability = evaluation.predict_proba(X.iloc[train_end:calibration_end])[:, 1]
    calibration_meta = metadata.iloc[train_end:calibration_end]
    calibration_sessions = pd.to_datetime(calibration_meta.timestamp).dt.strftime("%Y-%m-%d").nunique()
    thresholds = _threshold_table(
        y.iloc[train_end:calibration_end], calibration_probability, calibration_sessions)
    eligible = [row for row in thresholds if row["signals_per_session"] >= .25 and row["signals"] >= 30]
    selected = max(eligible, key=lambda row: (row["precision"], row["threshold"]), default=None)
    test_probability = evaluation.predict_proba(X.iloc[calibration_end:])[:, 1]
    test_meta = metadata.iloc[calibration_end:].reset_index(drop=True)
    test_sessions = pd.to_datetime(test_meta.timestamp).dt.strftime("%Y-%m-%d").nunique()
    test = (_threshold_table(y.iloc[calibration_end:], test_probability, test_sessions)
            [THRESHOLDS.index(selected["threshold"])] if selected else None)
    passed = bool(test and test["signals"] >= 20 and
                  test["signals_per_session"] >= .15 and test["precision"] >= .55)
    pipeline = build_pipeline("CATBOOST", X.iloc[:calibration_end], protocol)
    pipeline.fit(X.iloc[:calibration_end], y.iloc[:calibration_end])
    output.mkdir(parents=True, exist_ok=True)
    pipeline.named_steps["model"].save_model(output / "model.cbm", format="cbm")
    joblib.dump(pipeline.named_steps["preprocessor"], output / "preprocessor.joblib", compress=3)
    result = {
        "status": "FORWARD_PAPER_CANDIDATE" if passed else "RESEARCH_ONLY_VALIDATION_FAILED",
        "paper_only": True, "broker_orders_enabled": False, "strategy": strategy,
        "feature_data_contract": OPTION_FEATURE_VERSION,
        "direction": "CALL" if direction == 1 else "PUT", "target": "PREMIUM_PLUS_5PCT_WITHIN_15_MIN",
        "training_rows": train_end, "calibration_rows": calibration_end - train_end,
        "final_test_rows": len(X) - calibration_end,
        "final_test_start": str(test_meta.timestamp.iloc[0]),
        "final_test_end": str(test_meta.timestamp.iloc[-1]),
        "selected_threshold": selected["threshold"] if selected else None,
        "calibration_result": selected, "final_test_result": test,
        "calibration_thresholds": thresholds,
        "feature_order": X.columns.tolist(),
    }
    (output / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def train_all(root: Path) -> list[dict]:
    features = pd.read_parquet(root / "data/features/feature_dataset.parquet")
    features["timestamp"] = pd.to_datetime(features.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    aligned = pd.read_parquet(root / "data/aligned/canonical_index_option_aligned.parquet")
    call = aligned.rename(columns={
        "option_open": "open", "option_high": "high", "option_low": "low",
        "option_close": "close", "option_volume": "volume",
    })[["timestamp", "open", "high", "low", "close", "volume"]]
    put = pd.read_parquet(root / "data/processed/nifty_atm_put_rolling_1min.parquet")
    results = []
    for strategy in STRATEGIES:
        for direction, option, side in ((1, call, "call"), (-1, put, "put")):
            destination = root / "models/multistrategy_paper_v1" / f"{strategy.lower()}_{side}"
            results.append(train_candidate(features, option, strategy, direction, destination))
    report = root / "reports/multistrategy_retraining_v1"
    report.mkdir(parents=True, exist_ok=True)
    (report / "validation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    for item in train_all(Path(__file__).resolve().parents[1]):
        print(item["direction"], item["strategy"], item["status"], item["selected_threshold"])
