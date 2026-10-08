"""Phase 11 fixed-rule strategy robustness selection without ML or tuning."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ce_pe_feature_preview import feature_columns  # noqa: E402
from src.scalping_candidate_engine import STRATEGIES  # noqa: E402


OUTPUT = ROOT / "reports/phase11_strategy_selection"
EXPIRY = ROOT / "reports/phase9_label_behavior_audit/trade_path_audit.parquet"
NON_EXPIRY = ROOT / "reports/phase10_label_suitability/non_expiry_trade_path_audit.parquet"


def profit_factor(values: pd.Series) -> float | None:
    data = pd.to_numeric(values, errors="coerce").dropna()
    gains, losses = data[data > 0].sum(), -data[data < 0].sum()
    if losses == 0:
        return None if gains == 0 else float("inf")
    return float(gains / losses)


def maximum_drawdown(values: pd.Series) -> float:
    equity = pd.to_numeric(values, errors="coerce").fillna(0).cumsum()
    return float((equity.cummax() - equity).max()) if len(equity) else 0.0


def summarize(group: pd.DataFrame) -> dict:
    ordered = group.sort_values(["timestamp", "strategy"], kind="stable")
    executable = ordered.loc[ordered.actual_exit_reason.ne("AMBIGUOUS")]
    session_pnl = executable.groupby("session_date").net_pnl.sum()
    return {
        "total_trades": len(ordered), "eligible_trades": len(executable),
        "sessions": int(ordered.session_date.nunique()),
        "ce_trades": int(ordered.option_type.eq("CE").sum()),
        "pe_trades": int(ordered.option_type.eq("PE").sum()),
        "wins": int(ordered.actual_exit_reason.eq("WIN").sum()),
        "losses": int(ordered.actual_exit_reason.eq("LOSS").sum()),
        "ambiguous": int(ordered.actual_exit_reason.eq("AMBIGUOUS").sum()),
        "win_rate_eligible": float(executable.actual_exit_reason.eq("WIN").mean()) if len(executable) else None,
        "after_cost_expectancy": float(executable.net_pnl.mean()) if len(executable) else None,
        "profit_factor": profit_factor(executable.net_pnl),
        "maximum_drawdown": maximum_drawdown(executable.net_pnl),
        "average_holding_minutes": float(ordered.holding_minutes.mean()) if len(ordered) else None,
        "positive_session_rate": float(session_pnl.gt(0).mean()) if len(session_pnl) else None,
        "best_session": str(session_pnl.idxmax()) if len(session_pnl) else None,
        "best_session_pnl": float(session_pnl.max()) if len(session_pnl) else None,
        "worst_session": str(session_pnl.idxmin()) if len(session_pnl) else None,
        "worst_session_pnl": float(session_pnl.min()) if len(session_pnl) else None,
    }


def grouped(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for key, frame in data.groupby(columns, dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(columns, key)), **summarize(frame)})
    return pd.DataFrame(rows)


def walk_forward(data: pd.DataFrame, folds: int = 5) -> pd.DataFrame:
    dates = np.array(sorted(data.session_date.unique()))
    rows = []
    for fold, test_dates in enumerate(np.array_split(dates, folds), start=1):
        for strategy in STRATEGIES:
            test = data.loc[data.session_date.isin(test_dates) & data.strategy.eq(strategy)]
            if test.empty:
                continue
            rows.append({"fold": fold, "test_start": str(test_dates[0]),
                         "test_end": str(test_dates[-1]), "strategy": strategy,
                         **summarize(test)})
    return pd.DataFrame(rows)


def selection_gates(overall: dict, expiry: dict, non_expiry: dict,
                    fold_rows: pd.DataFrame) -> dict:
    fold_expectancy = pd.to_numeric(fold_rows.after_cost_expectancy, errors="coerce").dropna()
    return {
        "minimum_500_eligible_trades": overall["eligible_trades"] >= 500,
        "minimum_100_ce_trades": overall["ce_trades"] >= 100,
        "minimum_100_pe_trades": overall["pe_trades"] >= 100,
        "minimum_25_sessions": overall["sessions"] >= 25,
        "combined_expectancy_positive": (overall["after_cost_expectancy"] or 0) > 0,
        "non_expiry_expectancy_positive": (non_expiry["after_cost_expectancy"] or 0) > 0,
        "combined_profit_factor_above_one": (overall["profit_factor"] or 0) > 1,
        "positive_session_rate_at_least_55pct": (overall["positive_session_rate"] or 0) >= .55,
        "five_oos_folds_available": len(fold_expectancy) == 5,
        "median_oos_expectancy_positive": bool(len(fold_expectancy) and fold_expectancy.median() > 0),
        "at_least_three_positive_oos_folds": int(fold_expectancy.gt(0).sum()) >= 3,
        "expiry_result_reported": expiry["total_trades"] >= 0,
    }


def main() -> None:
    expiry = pd.read_parquet(EXPIRY)
    non_expiry = pd.read_parquet(NON_EXPIRY)
    data = pd.concat([expiry, non_expiry], ignore_index=True)
    assert data.entry_contract_locked.all() and data.exit_contract_matches_entry.all()
    expiry_features = pd.read_parquet(
        ROOT / "data/normalized/historical_options/validated20/feature_dataset_49.parquet",
        columns=["timestamp"])
    non_expiry_features = pd.read_parquet(
        ROOT / "data/normalized/historical_options/non_expiry15/feature_dataset_49.parquet",
        columns=["timestamp"])
    feature_timestamps = set(pd.concat([expiry_features, non_expiry_features]).timestamp)
    if not set(data.timestamp).issubset(feature_timestamps):
        raise AssertionError("A strategy trade is outside the validated Phase 6 feature timeline")
    if len(feature_columns()) != 49:
        raise AssertionError("Phase 6 feature contract changed")

    overall_table = grouped(data, ["strategy"])
    day_type = grouped(data, ["strategy", "expiry_day_status"])
    side = grouped(data, ["strategy", "option_type"])
    regime = grouped(data, ["strategy", "trend_regime"])
    sessions = grouped(data, ["strategy", "session_date"])
    folds = walk_forward(data)
    decisions = []
    for strategy in STRATEGIES:
        overall = summarize(data.loc[data.strategy.eq(strategy)])
        expiry_result = summarize(expiry.loc[expiry.strategy.eq(strategy)])
        non_expiry_result = summarize(non_expiry.loc[non_expiry.strategy.eq(strategy)])
        strategy_folds = folds.loc[folds.strategy.eq(strategy)]
        gates = selection_gates(overall, expiry_result, non_expiry_result, strategy_folds)
        decisions.append({"strategy": strategy, "passes": all(gates.values()),
                          "gates": gates, "overall": overall,
                          "expiry": expiry_result, "non_expiry": non_expiry_result,
                          "positive_oos_folds": int(strategy_folds.after_cost_expectancy.gt(0).sum()),
                          "median_oos_expectancy": float(strategy_folds.after_cost_expectancy.median())
                              if len(strategy_folds) else None})
    passed = [row["strategy"] for row in decisions if row["passes"]]
    recommendation = "A" if len(passed) == 1 else "B"
    result = {
        "status": "PHASE11_STRATEGY_SELECTION_COMPLETE",
        "recommendation": recommendation,
        "recommendation_text": (f"ONE strategy passes: {passed[0]}" if len(passed) == 1
                                else "NO strategy passes"),
        "provisional_final_scalping_strategy": passed[0] if len(passed) == 1 else None,
        "strategies_passing": passed, "decisions": decisions,
        "sessions": int(data.session_date.nunique()), "expiry_sessions": 20,
        "non_expiry_sessions": 15, "feature_count": 49,
        "catboost_trained": False, "production_changed": False,
        "strategy_selected_in_production": False,
        "limitations": [
            "Signals are evaluated independently and may overlap; drawdown is setup-level, not portfolio execution.",
            "VWAP Reclaim has no trades because NIFTY cash-index VWAP is unavailable.",
            "Ambiguous trades remain excluded from expectancy and are never forced to WIN or LOSS.",
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    overall_table.to_csv(OUTPUT / "strategy_overall.csv", index=False)
    day_type.to_csv(OUTPUT / "strategy_expiry_non_expiry.csv", index=False)
    side.to_csv(OUTPUT / "strategy_side.csv", index=False)
    regime.to_csv(OUTPUT / "strategy_regime.csv", index=False)
    sessions.to_csv(OUTPUT / "strategy_session.csv", index=False)
    folds.to_csv(OUTPUT / "walk_forward_folds.csv", index=False)
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
