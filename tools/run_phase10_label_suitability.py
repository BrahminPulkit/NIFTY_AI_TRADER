"""Compare unchanged 5%/3% labels on expiry and non-expiry sessions."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.expired_option_dataset import load_selected_contract_panel  # noqa: E402
from src.feature_pipeline import build_features  # noqa: E402
from src.scalping_candidate_engine import STRATEGIES, generate_scalping_candidates  # noqa: E402
from tools.run_label_behavior_audit import HORIZONS, breakdown, distribution, path_metrics  # noqa: E402
from tools.run_validated20_strategy_validation import context, simulate_contract_locked  # noqa: E402


SOURCE = ROOT / "data/normalized/historical_options/non_expiry15"
OUTPUT = ROOT / "reports/phase10_label_suitability"
EXPIRY_AUDIT = ROOT / "reports/phase9_label_behavior_audit/trade_path_audit.parquet"


def main() -> None:
    candidates, trades, panels = [], [], {}
    for session in sorted(path for path in SOURCE.iterdir() if path.is_dir()):
        manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
        index = pd.read_parquet(session / "nifty.parquet")
        ce, pe = pd.read_parquet(session / "ce.parquet"), pd.read_parquet(session / "pe.parquet")
        selected = pd.concat([ce, pe], ignore_index=True)
        archive = ROOT / "data/external/shoonyatrader/validated20_raw" / manifest["source_archive"]
        panel = load_selected_contract_panel(archive, ce, pe, session_date=manifest["date"])
        for key, group in panel.groupby(["expiry", "strike", "option_type"], sort=False):
            panels[(str(key[0]), float(key[1]), str(key[2]), manifest["date"])] = group
        session_candidates = generate_scalping_candidates(build_features(index))
        session_candidates = session_candidates.merge(context(index), on="timestamp", how="left")
        session_candidates["session_date"] = manifest["date"]
        session_candidates["expiry_group"] = manifest["expiry"]
        session_trades = simulate_contract_locked(session_candidates, selected, panel)
        session_trades = session_trades.merge(
            session_candidates[["timestamp", "strategy", "trend_regime", "session_date", "expiry_group"]],
            on=["timestamp", "strategy"], how="left", validate="many_to_one")
        candidates.append(session_candidates)
        trades.append(session_trades)
    candidates = pd.concat(candidates, ignore_index=True)
    trades = pd.concat(trades, ignore_index=True)
    metrics = []
    for trade in trades.itertuples(index=False):
        key = (str(trade.expiry), float(trade.strike), str(trade.option_type), trade.session_date)
        metrics.append(path_metrics(pd.Series(trade._asdict()), panels[key]))
    non_expiry = pd.concat([trades.reset_index(drop=True), pd.DataFrame(metrics)], axis=1)
    non_expiry = non_expiry.rename(columns={"outcome": "actual_exit_reason"})
    non_expiry["expiry_day_status"] = "NON_EXPIRY_DAY"
    expiry = pd.read_parquet(EXPIRY_AUDIT)
    comparison = pd.concat([expiry, non_expiry], ignore_index=True, sort=False)

    status = breakdown(comparison, ["expiry_day_status"])
    status["ambiguous_pct"] = status.ambiguous / status.trades * 100
    status["one_minute_exits"] = comparison.groupby("expiry_day_status").apply(
        lambda x: int(x.holding_minutes.eq(1).sum()), include_groups=False).reindex(
            status.expiry_day_status).to_numpy()
    status["one_minute_exit_pct"] = status.one_minute_exits / status.trades * 100
    status["first_target_pct"] = status.first_target_touched / status.trades * 100
    status["first_stop_pct"] = status.first_stop_touched / status.trades * 100

    metric_columns = ["return_1m_pct", "high_excursion_1m_pct", "low_excursion_1m_pct",
                      *[f"mfe_{h}m_pct" for h in (5, 10, 15)],
                      *[f"mae_{h}m_pct" for h in (5, 10, 15)]]
    distributions = {
        group: {column: distribution(frame[column]) for column in metric_columns}
        for group, frame in comparison.groupby("expiry_day_status", sort=False)
    }
    non_ambiguous = non_expiry.loc[non_expiry.actual_exit_reason.ne("AMBIGUOUS")]
    ambiguous_pct = float(non_expiry.actual_exit_reason.eq("AMBIGUOUS").mean() * 100)
    one_minute_pct = float(non_expiry.holding_minutes.eq(1).mean() * 100)
    class_counts = non_ambiguous.actual_exit_reason.value_counts().to_dict()
    # Suitability gates assess label resolution, not strategy profitability.
    gates = {
        "at_least_10_sessions": non_expiry.session_date.nunique() >= 10,
        "at_least_100_wins": class_counts.get("WIN", 0) >= 100,
        "at_least_100_losses": class_counts.get("LOSS", 0) >= 100,
        "ambiguity_at_most_10pct": ambiguous_pct <= 10,
        "one_minute_exits_at_most_75pct": one_minute_pct <= 75,
    }
    verdict = "A" if all(gates.values()) else "B"
    result = {
        "status": "PHASE10_LABEL_SUITABILITY_COMPLETE",
        "verdict": verdict,
        "verdict_text": ("Current label is suitable for model training." if verdict == "A" else
                         "Current label is unsuitable and needs a research-only label redesign."),
        "target_pct": 5, "stop_pct": 3, "labels_changed": False,
        "strategy_selected": False, "catboost_trained": False, "production_changed": False,
        "non_expiry_sessions_validated": int(non_expiry.session_date.nunique()),
        "expiry_sessions_compared": int(expiry.session_date.nunique()),
        "non_expiry_trades": len(non_expiry), "non_expiry_exit_counts": class_counts,
        "non_expiry_ambiguous": int(non_expiry.actual_exit_reason.eq("AMBIGUOUS").sum()),
        "non_expiry_ambiguous_pct": ambiguous_pct,
        "non_expiry_one_minute_exit_pct": one_minute_pct,
        "suitability_gates": gates,
        "distributions": distributions,
        "limitations": [
            "This assesses label resolution and class usability, not profitability.",
            "One-minute OHLC cannot resolve intrabar order when both barriers are touched.",
            "VWAP Reclaim remains unavailable because NIFTY cash-index VWAP is absent.",
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    non_expiry.to_parquet(OUTPUT / "non_expiry_trade_path_audit.parquet", index=False)
    comparison.to_parquet(OUTPUT / "expiry_non_expiry_trade_path_audit.parquet", index=False)
    status.to_csv(OUTPUT / "expiry_vs_non_expiry.csv", index=False)
    breakdown(comparison, ["expiry_day_status", "option_type"]).to_csv(
        OUTPUT / "comparison_side.csv", index=False)
    breakdown(comparison, ["expiry_day_status", "strategy"]).to_csv(
        OUTPUT / "comparison_strategy.csv", index=False)
    breakdown(comparison, ["expiry_day_status", "trend_regime"]).to_csv(
        OUTPUT / "comparison_regime.csv", index=False)
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
