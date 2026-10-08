"""Read-only label/exit behavior audit for contract-locked research trades."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.expired_option_dataset import load_selected_contract_panel  # noqa: E402


TRADES = ROOT / "reports/phase8_strategy_validation/trades.parquet"
SESSIONS = ROOT / "data/normalized/historical_options/validated20"
RAW = ROOT / "data/external/shoonyatrader/validated20_raw"
OUTPUT = ROOT / "reports/phase9_label_behavior_audit"
HORIZONS = (1, 2, 3, 5, 10, 15)


def path_metrics(entry: pd.Series, contract: pd.DataFrame) -> dict:
    timestamp, premium = pd.Timestamp(entry.timestamp), float(entry.entry_premium)
    future = contract.loc[contract.timestamp.gt(timestamp)].sort_values("timestamp", kind="stable")
    first = future.iloc[0] if len(future) else None
    result = {
        "first_candle_target_touched": bool(first is not None and first.high >= entry.target),
        "first_candle_stop_touched": bool(first is not None and first.low <= entry.stop),
        "first_candle_both_touched": bool(
            first is not None and first.high >= entry.target and first.low <= entry.stop),
        "return_1m_pct": float((first.close / premium - 1) * 100) if first is not None else np.nan,
        "high_excursion_1m_pct": float((first.high / premium - 1) * 100) if first is not None else np.nan,
        "low_excursion_1m_pct": float((first.low / premium - 1) * 100) if first is not None else np.nan,
        "first_candle_range_pct": float((first.high - first.low) / premium * 100)
            if first is not None else np.nan,
        "target_distance_pct": float((entry.target / premium - 1) * 100),
        "stop_distance_pct": float((entry.stop / premium - 1) * 100),
    }
    for horizon in HORIZONS:
        end = timestamp + pd.Timedelta(minutes=horizon)
        path = future.loc[future.timestamp.le(end)]
        result[f"mfe_{horizon}m_pct"] = float((path.high.max() / premium - 1) * 100) \
            if len(path) else np.nan
        result[f"mae_{horizon}m_pct"] = float((path.low.min() / premium - 1) * 100) \
            if len(path) else np.nan
    return result


def distribution(values: pd.Series) -> dict:
    data = pd.to_numeric(values, errors="coerce").dropna()
    if data.empty:
        return {"count": 0}
    quantiles = data.quantile([.01, .05, .25, .5, .75, .95, .99])
    return {
        "count": len(data), "mean": float(data.mean()), "std": float(data.std(ddof=0)),
        "min": float(data.min()), "p01": float(quantiles.loc[.01]),
        "p05": float(quantiles.loc[.05]), "p25": float(quantiles.loc[.25]),
        "median": float(quantiles.loc[.5]), "p75": float(quantiles.loc[.75]),
        "p95": float(quantiles.loc[.95]), "p99": float(quantiles.loc[.99]),
        "max": float(data.max()),
    }


def breakdown(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in data.groupby(columns, dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(columns, key))
        row.update({
            "trades": len(group), "wins": int(group.actual_exit_reason.eq("WIN").sum()),
            "losses": int(group.actual_exit_reason.eq("LOSS").sum()),
            "timeouts": int(group.actual_exit_reason.eq("TIMEOUT").sum()),
            "ambiguous": int(group.actual_exit_reason.eq("AMBIGUOUS").sum()),
            "first_target_touched": int(group.first_candle_target_touched.sum()),
            "first_stop_touched": int(group.first_candle_stop_touched.sum()),
            "first_both_touched": int(group.first_candle_both_touched.sum()),
            "average_holding_minutes": float(group.holding_minutes.mean()),
            "average_return_1m_pct": float(group.return_1m_pct.mean()),
            "average_mfe_5m_pct": float(group.mfe_5m_pct.mean()),
            "average_mae_5m_pct": float(group.mae_5m_pct.mean()),
            "gross_expectancy": float(group.gross_pnl.mean()),
            "after_cost_expectancy": float(group.net_pnl.mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    trades = pd.read_parquet(TRADES)
    panels = {}
    for session in sorted(path for path in SESSIONS.iterdir() if path.is_dir()):
        ce, pe = pd.read_parquet(session / "ce.parquet"), pd.read_parquet(session / "pe.parquet")
        panel = load_selected_contract_panel(
            RAW / f"{session.name.replace('-', '')}.zip", ce, pe)
        for key, group in panel.groupby(["expiry", "strike", "option_type"], sort=False):
            panels[(str(key[0]), float(key[1]), str(key[2]))] = group
    metrics = []
    for trade in trades.itertuples(index=False):
        key = (str(trade.expiry), float(trade.strike), str(trade.option_type))
        values = path_metrics(pd.Series(trade._asdict()), panels[key])
        metrics.append(values)
    audit = pd.concat([trades.reset_index(drop=True), pd.DataFrame(metrics)], axis=1)
    audit = audit.rename(columns={"outcome": "actual_exit_reason"})
    audit["expiry_day_status"] = np.where(
        audit.timestamp.dt.strftime("%Y-%m-%d").eq(audit.expiry), "EXPIRY_DAY", "NON_EXPIRY_DAY")

    breakdowns = {
        "side": breakdown(audit, ["option_type"]),
        "strategy": breakdown(audit, ["strategy"]),
        "expiry": breakdown(audit, ["expiry"]),
        "expiry_day": breakdown(audit, ["expiry_day_status"]),
        "regime": breakdown(audit, ["trend_regime"]),
    }
    distribution_columns = [
        "return_1m_pct", "high_excursion_1m_pct", "low_excursion_1m_pct",
        "target_distance_pct", "stop_distance_pct",
        *[f"mfe_{h}m_pct" for h in HORIZONS], *[f"mae_{h}m_pct" for h in HORIZONS],
    ]
    ambiguous = audit.loc[audit.actual_exit_reason.eq("AMBIGUOUS")]
    result = {
        "status": "LABEL_BEHAVIOR_AUDIT_COMPLETE",
        "flow_changed": False, "labels_changed": False, "strategy_selected": False,
        "trades": len(audit),
        "exit_counts": audit.actual_exit_reason.value_counts().to_dict(),
        "first_candle": {
            "target_touched": int(audit.first_candle_target_touched.sum()),
            "stop_touched": int(audit.first_candle_stop_touched.sum()),
            "both_touched": int(audit.first_candle_both_touched.sum()),
            "exited_after_one_minute": int(audit.holding_minutes.eq(1).sum()),
        },
        "ambiguity": {
            "count": len(ambiguous),
            "first_candle_both_touched": int(ambiguous.first_candle_both_touched.sum()),
            "intrabar_order_knowable_from_ohlc": False,
            "primary_cause": "Both fixed target and stop are inside one-minute OHLC range; touch order is unknowable.",
            "all_sessions_expiry_day": bool(ambiguous.expiry_day_status.eq("EXPIRY_DAY").all()),
            "mean_first_candle_range_pct": float(ambiguous.first_candle_range_pct.mean()),
        },
        "distributions": {name: distribution(audit[name]) for name in distribution_columns},
        "limitations": [
            "All 20 validated sessions are expiry days; no non-expiry-day comparison is available.",
            "One-minute OHLC cannot establish target-versus-stop order inside an ambiguous candle.",
            "VWAP Reclaim remains unavailable because NIFTY cash-index volume/VWAP is absent.",
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit.to_parquet(OUTPUT / "trade_path_audit.parquet", index=False)
    audit.to_csv(OUTPUT / "trade_path_audit.csv", index=False)
    for name, frame in breakdowns.items():
        frame.to_csv(OUTPUT / f"breakdown_{name}.csv", index=False)
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
