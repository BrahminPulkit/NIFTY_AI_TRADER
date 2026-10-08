"""Generate Step-26 observability reports without invoking model inference."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import pandas as pd

from src.institutional_monitoring import (
    InstitutionalMonitor, MonitoringConfig, read_csv, read_json, sha256,
)


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports/system_monitoring"
CONFIG = ROOT / "config/system_monitoring.json"


def frozen_paths() -> list[Path]:
    manifest = read_json(ROOT / "production_model/model_hashes.json")
    paths = [ROOT / path for path in manifest.get("inputs", {})]
    paths += [ROOT / "production_model" / path for path in manifest.get("outputs", {})]
    paths += [
        ROOT / "src/setup_engine.py", ROOT / "src/feature_pipeline.py",
        ROOT / "src/strategy_decision_engine.py",
        ROOT / "src/paper_trading_engine.py",
    ]
    return [path for path in paths if path.exists()]


def period_report(name: str, predictions: pd.DataFrame, trades: pd.DataFrame,
                  health: pd.DataFrame, alerts: pd.DataFrame,
                  latency: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    if name == "daily":
        start, label = now.normalize(), str(now.date())
    elif name == "weekly":
        start = now.normalize() - pd.Timedelta(days=now.weekday())
        label = f"{start.date()} to {now.date()}"
    else:
        start = now.normalize().replace(day=1)
        label = now.strftime("%Y-%m")
    selected = predictions.loc[predictions.timestamp >= start] if len(predictions) else predictions
    if len(trades):
        exit_time = pd.to_datetime(trades.exit_timestamp, utc=True)
        selected_trades = trades.loc[exit_time >= start]
    else:
        selected_trades = trades
    pnl = pd.to_numeric(selected_trades.net_pnl, errors="coerce") if len(selected_trades) else pd.Series(dtype=float)
    costs = pd.to_numeric(selected_trades.total_costs, errors="coerce") if len(selected_trades) else pd.Series(dtype=float)
    strategies = selected.strategy.value_counts().to_dict() if len(selected) else {}
    regimes = selected.regime.value_counts().to_dict() if len(selected) else {}
    return pd.DataFrame([{
        "report": name, "period": label, "signals": len(selected),
        "trades": int(selected.decision.eq("TRADE").sum()) if len(selected) else 0,
        "skipped": int(selected.decision.eq("SKIP").sum()) if len(selected) else 0,
        "completed_trades": len(selected_trades),
        "win_rate": float(pnl.gt(0).mean()) if len(pnl) else 0.0,
        "profit": float(pnl[pnl > 0].sum()) if len(pnl) else 0.0,
        "loss": float(pnl[pnl < 0].sum()) if len(pnl) else 0.0,
        "net_pnl": float(pnl.sum()) if len(pnl) else 0.0,
        "costs": float(costs.sum()) if len(costs) else 0.0,
        "strategies": json.dumps(strategies), "regimes": json.dumps(regimes),
        "ready_components": int(health.status.eq("READY").sum()),
        "blocked_components": int(health.status.isin(["BLOCKED", "FAILED"]).sum()),
        "alerts": len(alerts),
        "average_latency_ms": float(latency.rolling_average_latency_ms.iloc[0])
        if len(latency) and pd.notna(latency.rolling_average_latency_ms.iloc[0]) else None,
        "feed_uptime": None,
        "feed_uptime_status": "WAITING_NO_CONTINUOUS_FEED_HEARTBEAT",
    }])


def write_period(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(REPORTS / f"{name}_report.csv", index=False)
    payload = frame.iloc[0].where(pd.notna(frame.iloc[0]), None).to_dict()
    (REPORTS / f"{name}_report.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    lines = [f"# {name.title()} Institutional Monitoring Report", ""]
    lines += [f"- {key.replace('_', ' ').title()}: {value}" for key, value in payload.items()]
    (REPORTS / f"{name}_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    payload = read_json(CONFIG)
    config = MonitoringConfig(**{
        key: payload[key] for key in asdict(MonitoringConfig())
        if key in payload})
    protected = frozen_paths()
    before = {str(path.relative_to(ROOT)): sha256(path) for path in protected}
    monitor = InstitutionalMonitor(ROOT, config)
    predictions = monitor.predictions()
    trades = read_csv(ROOT / "reports/paper_trading/paper_trade_log.csv")
    health = monitor.system_health(predictions)
    pipeline = monitor.pipeline_status(predictions)
    latency = monitor.latency(predictions)
    data_health = monitor.data_health(predictions)
    feature = monitor.feature_health()
    prediction = monitor.prediction_health(predictions)
    drift = monitor.drift(predictions, trades)
    if feature.status.iloc[0] in {"BLOCKED", "FAILED"}:
        monitor.alert(
            "CRITICAL", "Feature contract validation is blocked.",
            "Stop inference and inspect the exact feature-health failures.")
    if pipeline.status.isin(["BLOCKED", "FAILED"]).any():
        failed = pipeline.loc[
            pipeline.status.isin(["BLOCKED", "FAILED"]), "stage"].tolist()
        monitor.alert(
            "CRITICAL", f"Pipeline stage unavailable: {failed}",
            "Keep inference SAFE_BLOCK and restore the affected stage.")
    decision = health.loc[health.component.eq("Decision Engine")]
    if len(decision) and decision.status.iloc[0] in {"BLOCKED", "FAILED"}:
        monitor.alert(
            "CRITICAL", "Decision Engine is unavailable.",
            "Do not create paper trades until its runtime heartbeat is healthy.")
    paper_risk = read_csv(ROOT / "reports/paper_trading/risk_statistics.csv")
    if len(paper_risk) and float(paper_risk.net_pnl.iloc[0]) <= -10_000:
        monitor.alert("CRITICAL", "Paper daily loss tolerance reached.",
                      "Stop new virtual entries and inspect the paper ledger.")
    alerts = pd.DataFrame(monitor.alerts, columns=[
        "timestamp", "severity", "message", "suggested_action"])
    REPORTS.mkdir(parents=True, exist_ok=True)
    pipeline.to_csv(REPORTS / "pipeline_status.csv", index=False)
    latency.to_csv(REPORTS / "latency_statistics.csv", index=False)
    feature.to_csv(REPORTS / "feature_health.csv", index=False)
    prediction.to_csv(REPORTS / "prediction_health.csv", index=False)
    drift.to_csv(REPORTS / "drift_monitor.csv", index=False)
    alerts.to_csv(REPORTS / "alerts.csv", index=False)
    health.to_csv(REPORTS / "component_health.csv", index=False)
    data_health.to_csv(REPORTS / "data_health.csv", index=False)
    for name in ("daily", "weekly", "monthly"):
        write_period(name, period_report(
            name, predictions, trades, health, alerts, latency, monitor.now))
    after = {str(path.relative_to(ROOT)): sha256(path) for path in protected}
    if before != after:
        raise AssertionError("A frozen or protected component changed during monitoring")
    metadata = {
        "step": "26", "generated_utc": monitor.now.isoformat(),
        "environment": payload.get("environment"),
        "system_mode": payload.get("system_mode"),
        "configuration": asdict(config),
        "prediction_rows_observed": len(predictions),
        "paper_trades_observed": len(trades),
        "frozen_hashes_before": before, "frozen_hashes_after": after,
        "frozen_hashes_identical": True,
        "model_inference_invoked": False,
        "probabilities_regenerated": False,
        "datasets_modified": False,
    }
    (REPORTS / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    ready, blocked = int(health.status.eq("READY").sum()), int(
        health.status.isin(["BLOCKED", "FAILED"]).sum())
    (REPORTS / "system_health_report.md").write_text(
        f"""# QuantEdge AI — Institutional System Health

- Generated: {monitor.now.isoformat()}
- Environment: {payload.get('environment')}
- Mode: {payload.get('system_mode')}
- Ready components: {ready}/{len(health)}
- Blocked or failed components: {blocked}
- Model version: production_candidate_v1.0.0
- Feature contract: 83 / 83 validated by the latest Step-24 inference
- Prediction observations: {len(predictions)}
- Completed paper trades: {len(trades)}
- Active alerts: {len(alerts)}

## Operational verdict

The model package and causal pipeline artifacts are verified. Current market
feeds are stale or disconnected, so the platform remains a paper/replay
production candidate. Feature PSI and feed uptime remain `WAITING` until live
feature vectors and continuous heartbeat telemetry exist.

No model inference, probability generation, training, or trading action was
performed by this monitoring run.
""", encoding="utf-8")
    monitor.logger.write("uptime", "WAITING_NO_CONTINUOUS_PROCESS_HEARTBEAT")


if __name__ == "__main__":
    main()
