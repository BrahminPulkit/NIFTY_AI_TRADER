"""Step 20C: threshold research using saved OOF probabilities only."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


THRESHOLDS = tuple([round(value, 2) for value in np.arange(0.50, 1.00, 0.05)] + [0.97, 0.99])
MODEL_PROBABILITIES = {
    "CATBOOST": "catboost_predicted_probability",
    "XGBOOST": "xgboost_predicted_probability",
}
GROUP_COLUMNS = {
    "year": "year",
    "market_regime": "market_regime",
    "session_phase": "session",
    "strategy_family": "strategy_family",
    "fold_id": "fold_id",
}


def validate_joined_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "timestamp", "fold_id", "true_label", *MODEL_PROBABILITIES.values(),
        "terminal_premium_return", "premium_expansion", "premium_drawdown",
        "holding_time_minutes", "strategy_family", "market_regime", "session_phase",
    }
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"Joined probability dataset missing: {missing}")
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(
        data.timestamp, utc=True, errors="raise"
    ).dt.tz_convert("Asia/Kolkata")
    if data.timestamp.duplicated().any() or not data.timestamp.is_monotonic_increasing:
        raise ValueError("Joined timestamps must be unique and chronological")
    if not data.true_label.isin([0, 1]).all():
        raise ValueError("True labels must be binary")
    for column in MODEL_PROBABILITIES.values():
        if not data[column].between(0, 1).all():
            raise ValueError(f"Invalid saved OOF probability: {column}")
    data["year"] = data.timestamp.dt.year
    return data


def _distribution(group: pd.DataFrame, column: str) -> str:
    values = group[column].value_counts(normalize=True, dropna=False).sort_index()
    return json.dumps({str(key): float(value) for key, value in values.items()}, sort_keys=True)


def threshold_metrics(
    data: pd.DataFrame,
    probability_column: str,
    threshold: float,
    include_distributions: bool = True,
) -> dict:
    probability = data[probability_column].to_numpy(float)
    prediction = probability >= threshold
    selected = data.loc[prediction]
    trades = len(selected)
    total = len(data)
    wins = int(selected.true_label.sum())
    losses = trades - wins
    if trades:
        winner_returns = selected.loc[selected.true_label.eq(1), "terminal_premium_return"]
        loser_returns = selected.loc[selected.true_label.eq(0), "terminal_premium_return"]
        average_winner = float(winner_returns.mean()) if len(winner_returns) else np.nan
        average_loser = float(loser_returns.mean()) if len(loser_returns) else np.nan
        expected_value = float(selected.terminal_premium_return.mean())
        profit_expectancy = (
            (wins / trades) * (average_winner if np.isfinite(average_winner) else 0)
            + (losses / trades) * (average_loser if np.isfinite(average_loser) else 0)
        )
    else:
        average_winner = average_loser = expected_value = profit_expectancy = np.nan
    both_classes = data.true_label.nunique() == 2
    row = {
        "threshold": threshold,
        "total_observations": total,
        "trades": trades,
        "trades_skipped": total - trades,
        "trade_frequency": trades / total if total else np.nan,
        "coverage": trades / total if total else np.nan,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / trades if trades else np.nan,
        "loss_rate": losses / trades if trades else np.nan,
        "average_terminal_premium_return": expected_value,
        "average_premium_expansion": float(selected.premium_expansion.mean()) if trades else np.nan,
        "average_premium_drawdown": float(selected.premium_drawdown.mean()) if trades else np.nan,
        "expected_value_per_trade": expected_value,
        "profit_expectancy": profit_expectancy,
        "average_winner_return": average_winner,
        "average_loser_return": average_loser,
        "average_holding_time": float(selected.holding_time_minutes.mean()) if trades else np.nan,
        "average_confidence": float(selected[probability_column].mean()) if trades else np.nan,
        "precision": float(precision_score(data.true_label, prediction, zero_division=0)),
        "recall": float(recall_score(data.true_label, prediction, zero_division=0)),
        "f1": float(f1_score(data.true_label, prediction, zero_division=0)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(data.true_label, prediction))
            if both_classes else np.nan
        ),
    }
    if include_distributions:
        row.update({
            "strategy_distribution": _distribution(selected, "strategy_family") if trades else "{}",
            "regime_distribution": _distribution(selected, "market_regime") if trades else "{}",
            "session_distribution": _distribution(selected, "session_phase") if trades else "{}",
        })
    return row


def evaluate_thresholds(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, column in MODEL_PROBABILITIES.items():
        for threshold in THRESHOLDS:
            rows.append({
                "model": model,
                **threshold_metrics(data, column, threshold),
            })
    return pd.DataFrame(rows)


def grouped_threshold_statistics(data: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    output_name = GROUP_COLUMNS[group_column]
    for group_value, group in data.groupby(group_column, sort=True, dropna=False):
        for model, probability_column in MODEL_PROBABILITIES.items():
            for threshold in THRESHOLDS:
                rows.append({
                    output_name: group_value,
                    "model": model,
                    **threshold_metrics(
                        group, probability_column, threshold,
                        include_distributions=False,
                    ),
                })
    return pd.DataFrame(rows)


def pareto_frontier(statistics: pd.DataFrame) -> pd.DataFrame:
    """Return nondominated rows maximizing frequency, EV and precision."""
    rows = []
    for model, model_rows in statistics.groupby("model", sort=True):
        candidates = model_rows.dropna(
            subset=["trade_frequency", "expected_value_per_trade", "precision"]
        )
        for index, candidate in candidates.iterrows():
            dominates = (
                (candidates.trade_frequency >= candidate.trade_frequency)
                & (candidates.expected_value_per_trade >= candidate.expected_value_per_trade)
                & (candidates.precision >= candidate.precision)
                & (
                    (candidates.trade_frequency > candidate.trade_frequency)
                    | (candidates.expected_value_per_trade > candidate.expected_value_per_trade)
                    | (candidates.precision > candidate.precision)
                )
            )
            if not dominates.any():
                rows.append(candidate)
    return pd.DataFrame(rows).sort_values(["model", "threshold"]).reset_index(drop=True)


def candidate_thresholds(statistics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in statistics.groupby("model", sort=True):
        positive = group.loc[group.expected_value_per_trade.gt(0)]
        aggressive = (
            positive.sort_values(["threshold", "coverage"], ascending=[True, False]).iloc[0]
            if not positive.empty else group.sort_values("threshold").iloc[0]
        )
        eligible = positive.loc[positive.trades.ge(100)]
        balanced = (
            eligible.sort_values(["f1", "expected_value_per_trade"], ascending=False).iloc[0]
            if not eligible.empty else aggressive
        )
        conservative = (
            eligible.sort_values(
                ["precision", "expected_value_per_trade", "threshold"],
                ascending=[False, False, False],
            ).iloc[0]
            if not eligible.empty else aggressive
        )
        for profile, row in (
            ("AGGRESSIVE", aggressive),
            ("BALANCED", balanced),
            ("CONSERVATIVE", conservative),
        ):
            rows.append({
                "model": model, "profile": profile,
                "threshold": row.threshold, "trades": row.trades,
                "coverage": row.coverage, "precision": row.precision,
                "recall": row.recall, "f1": row.f1,
                "expected_value_per_trade": row.expected_value_per_trade,
            })
    return pd.DataFrame(rows)


def equity_curves(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, probability_column in MODEL_PROBABILITIES.items():
        for threshold in THRESHOLDS:
            selected = data.loc[data[probability_column].ge(threshold)].copy()
            if selected.empty:
                continue
            selected["model"] = model
            selected["threshold"] = threshold
            selected["trade_number"] = np.arange(1, len(selected) + 1)
            selected["arithmetic_cumulative_return"] = (
                selected.terminal_premium_return.cumsum()
            )
            selected["compounded_equity"] = (
                (1 + selected.terminal_premium_return).cumprod()
            )
            rows.append(selected[[
                "timestamp", "model", "threshold", "trade_number",
                "terminal_premium_return", "arithmetic_cumulative_return",
                "compounded_equity",
            ]])
    return pd.concat(rows, ignore_index=True)


def confidence_bands(data: pd.DataFrame) -> pd.DataFrame:
    edges = [0.0, 0.50, 0.60, 0.70, 0.80, 0.90, 1.000001]
    rows = []
    for model, probability_column in MODEL_PROBABILITIES.items():
        bands = pd.cut(
            data[probability_column], bins=edges, right=False, include_lowest=True
        )
        for band, group in data.groupby(bands, observed=True):
            rows.append({
                "model": model, "confidence_band": str(band),
                "observations": len(group),
                "win_rate": float(group.true_label.mean()),
                "average_probability": float(group[probability_column].mean()),
                "average_terminal_return": float(group.terminal_premium_return.mean()),
                "average_expansion": float(group.premium_expansion.mean()),
                "average_drawdown": float(group.premium_drawdown.mean()),
            })
    return pd.DataFrame(rows)

