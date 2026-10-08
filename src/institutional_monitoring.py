"""Step 26 read-only institutional observability and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from logging import Formatter, Logger, INFO
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd


VALID_STATUSES = {"READY", "WAITING", "STALE", "BLOCKED", "FAILED"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


class DailyLogSet:
    """Six independently rotated operational log channels."""

    NAMES = ("pipeline", "health", "latency", "alerts", "drift", "uptime")

    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.loggers: dict[str, Logger] = {}
        for name in self.NAMES:
            logger = Logger(f"quantedge.monitoring.{name}", level=INFO)
            handler = TimedRotatingFileHandler(
                root / f"{name}.log", when="midnight", interval=1,
                backupCount=90, encoding="utf-8", utc=False)
            handler.setFormatter(Formatter(
                "%(asctime)s | %(levelname)s | %(message)s"))
            logger.addHandler(handler)
            self.loggers[name] = logger

    def write(self, channel: str, message: str) -> None:
        self.loggers[channel].info(message)


class IncrementalJSONL:
    """Read only newly appended JSONL bytes and maintain a derived cache."""

    def __init__(self, source: Path, cache: Path, state: Path):
        self.source, self.cache, self.state = source, cache, state

    def update(self) -> pd.DataFrame:
        state = read_json(self.state)
        offset = int(state.get("offset", 0))
        size = self.source.stat().st_size if self.source.exists() else 0
        if offset > size:  # source truncation is an alertable integrity event
            raise RuntimeError("Append-only source size decreased")
        new = []
        if self.source.exists():
            with self.source.open("rb") as handle:
                handle.seek(offset)
                for raw in handle:
                    if raw.strip():
                        new.append(json.loads(raw.decode("utf-8")))
                new_offset = handle.tell()
        else:
            new_offset = 0
        existing = pd.read_parquet(self.cache) if self.cache.exists() else pd.DataFrame()
        if new:
            incoming = pd.DataFrame(new)
            combined = pd.concat([existing, incoming], ignore_index=True)
            keys = [key for key in ("timestamp", "inference_timestamp", "model_version")
                    if key in combined]
            if keys:
                combined = combined.drop_duplicates(keys, keep="first")
        else:
            combined = existing
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache.with_suffix(".tmp.parquet")
        combined.to_parquet(temporary, index=False)
        temporary.replace(self.cache)
        self.state.write_text(json.dumps(
            {"offset": new_offset, "source_size": size,
             "updated_utc": datetime.now(timezone.utc).isoformat()}, indent=2),
            encoding="utf-8")
        return combined


@dataclass(frozen=True)
class MonitoringConfig:
    stale_after_minutes: int = 5
    latency_warning_ms: float = 1000.0
    psi_warning: float = 0.20
    minimum_drift_observations: int = 20
    skipped_trade_warning_rate: float = 0.95


class InstitutionalMonitor:
    def __init__(self, root: str | Path, config: MonitoringConfig):
        self.root, self.config = Path(root), config
        self.reports = self.root / "reports/system_monitoring"
        self.logs = self.root / "logs/system_health"
        self.cache = self.logs / "cache"
        self.reports.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.logger = DailyLogSet(self.logs)
        self.alerts: list[dict] = []
        self.now = utc_now()

    def alert(self, severity: str, message: str, action: str) -> None:
        row = {
            "timestamp": self.now.isoformat(), "severity": severity,
            "message": message, "suggested_action": action,
        }
        self.alerts.append(row)
        self.logger.write("alerts", json.dumps(row))

    def predictions(self) -> pd.DataFrame:
        incremental = IncrementalJSONL(
            self.root / "logs/live_inference/predictions.jsonl",
            self.cache / "predictions.parquet",
            self.cache / "prediction_cursor.json")
        try:
            data = incremental.update()
        except Exception as exc:
            self.alert("CRITICAL", f"Prediction journal integrity failure: {exc}",
                       "Stop monitoring ingestion and inspect the append-only source.")
            return pd.DataFrame()
        if len(data):
            data["timestamp"] = pd.to_datetime(data.timestamp, utc=True)
            data["inference_timestamp"] = pd.to_datetime(data.inference_timestamp, utc=True)
        return data

    def verify_model(self) -> tuple[bool, list[str]]:
        manifest = read_json(self.root / "production_model/model_hashes.json")
        failures = []
        for name, expected in manifest.get("outputs", {}).items():
            path = self.root / "production_model" / name
            if not path.exists() or sha256(path) != expected:
                failures.append(name)
        if failures:
            self.alert("CRITICAL", f"Model package hash mismatch: {failures}",
                       "Block inference and restore the approved Step-23 package.")
        return not failures, failures

    def system_health(self, predictions: pd.DataFrame) -> pd.DataFrame:
        model_ok, _ = self.verify_model()
        pipeline = read_json(self.root / "logs/live_inference/pipeline_health.json")
        metadata = read_json(self.root / "production_model/production_metadata.json")
        contract = read_json(self.root / "production_model/feature_contract.json")
        paper_meta = read_json(self.root / "reports/paper_trading/metadata.json")
        updated = pipeline.get("updated_utc")
        runtime_age = (
            (self.now - pd.to_datetime(updated, utc=True)).total_seconds() / 60
            if updated else math.inf)
        runtime_status = (
            "STALE" if runtime_age > self.config.stale_after_minutes else
            "READY" if pipeline.get("status") in {"READY", "HEALTHY"} else "BLOCKED")
        paper_path = self.root / "reports/paper_trading/metadata.json"
        paper_time = (
            pd.Timestamp(paper_path.stat().st_mtime, unit="s", tz="UTC").isoformat()
            if paper_path.exists() else None)
        rows = [
            ("Model", "READY" if model_ok else "FAILED", metadata.get("training_timestamp_utc"),
             metadata.get("model_version", "Unavailable")),
            ("Feature Contract", "READY" if pipeline.get("contract_status") == "VERIFIED" else "BLOCKED",
             updated, f"{contract.get('contract_version','Unavailable')} · {contract.get('feature_count',0)} fields"),
            ("Threshold", "READY" if metadata.get("approved_probability_threshold") == .9 else "FAILED",
             metadata.get("training_timestamp_utc"), "v1 · 0.90"),
            ("Stage-1 V2", runtime_status if (self.root / "src/setup_engine.py").exists() else "FAILED",
             updated, "Frozen canonical implementation"),
            ("Decision Engine", runtime_status if (self.root / "data/runtime/decision_observations.parquet").exists() else "BLOCKED",
             updated, pipeline.get("decision_status", "Unavailable")),
            ("Prediction Engine", runtime_status,
             updated, pipeline.get("status", "Unavailable")),
            ("Feature Pipeline", runtime_status if pipeline.get("feature_status") == "VALID" else "BLOCKED",
             updated, pipeline.get("feature_status", "Unavailable")),
            ("Paper Trading", "WAITING" if paper_meta.get("mode") == "PAPER_ONLY" else "BLOCKED",
             paper_time, "No continuous process heartbeat"),
        ]
        output = pd.DataFrame(rows, columns=["component", "status", "timestamp", "detail"])
        if not output.status.isin(VALID_STATUSES).all():
            raise AssertionError("Invalid institutional health status")
        self.logger.write("health", output.to_json(orient="records"))
        return output

    def pipeline_status(self, predictions: pd.DataFrame) -> pd.DataFrame:
        health = read_json(self.root / "logs/live_inference/pipeline_health.json")
        if len(predictions):
            latest = predictions.sort_values("inference_timestamp").iloc[-1]
            candle, inference = latest.timestamp, latest.inference_timestamp
        else:
            candle = inference = None
        journal = self.root / "logs/live_inference/predictions.jsonl"
        journal_time = pd.Timestamp(journal.stat().st_mtime, unit="s", tz="UTC") if journal.exists() else None
        values = [
            ("Market Feed", candle, "Last prediction candle"),
            ("Feature Generation", inference, health.get("feature_status", "UNKNOWN")),
            ("Stage-1", inference, "Evaluated before every recorded inference"),
            ("Decision Engine", inference, health.get("decision_status", "UNKNOWN")),
            ("CatBoost", inference, health.get("model_status", "UNKNOWN")),
            ("Prediction Journal", journal_time, health.get("journal_status", "UNKNOWN")),
            ("Paper Trading", pd.Timestamp(
                (self.root / "reports/paper_trading/metadata.json").stat().st_mtime,
                unit="s", tz="UTC") if (self.root / "reports/paper_trading/metadata.json").exists() else None,
             "PAPER_ONLY"),
        ]
        rows = []
        for stage, timestamp, detail in values:
            if timestamp is None:
                status = "WAITING"
            else:
                age = (self.now - pd.to_datetime(timestamp, utc=True)).total_seconds() / 60
                status = "STALE" if age > self.config.stale_after_minutes else "READY"
            if stage == "Paper Trading":
                status = "WAITING"
            rows.append({"stage": stage, "status": status, "last_success": timestamp,
                         "detail": detail})
        output = pd.DataFrame(rows)
        self.logger.write("pipeline", output.to_json(orient="records", date_format="iso"))
        return output

    def latency(self, predictions: pd.DataFrame) -> pd.DataFrame:
        if predictions.empty or "latency_ms" not in predictions:
            return pd.DataFrame([{"observations": 0, "current_latency_ms": np.nan,
                                  "rolling_average_latency_ms": np.nan,
                                  "maximum_latency_ms": np.nan, "status": "WAITING"}])
        values = pd.to_numeric(predictions.latency_ms, errors="coerce").dropna()
        current, rolling, maximum = values.iloc[-1], values.tail(100).mean(), values.max()
        status = "READY" if current <= self.config.latency_warning_ms else "STALE"
        if status == "STALE":
            self.alert("WARNING", f"Prediction latency {current:.1f} ms exceeds tolerance.",
                       "Inspect feature generation and model host utilisation.")
        self.logger.write("latency", f"current={current:.3f} rolling={rolling:.3f} max={maximum:.3f}")
        return pd.DataFrame([{"observations": len(values), "current_latency_ms": current,
                              "rolling_average_latency_ms": rolling,
                              "maximum_latency_ms": maximum, "status": status}])

    def data_health(self, predictions: pd.DataFrame) -> pd.DataFrame:
        latest = predictions.timestamp.max() if len(predictions) else None
        age_minutes = (
            (self.now - latest).total_seconds() / 60 if latest is not None else np.nan)
        inferred_status = (
            "BLOCKED" if latest is None else
            "STALE" if age_minutes > self.config.stale_after_minutes else "READY")
        if inferred_status == "STALE":
            self.alert("WARNING", "NIFTY and option inference feeds are stale.",
                       "Verify Dhan market contracts and restart the read-only feed.")
        rows = [
            ("NIFTY", inferred_status, latest, age_minutes, "Inferred from last Step-24 candle"),
            ("ATM CALL", inferred_status, latest, age_minutes, "Inferred from last Step-24 candle"),
            ("BANKNIFTY", "BLOCKED", None, np.nan, "No approved feed configured"),
            ("SENSEX", "BLOCKED", None, np.nan, "No approved feed configured"),
            ("Timestamp Alignment", "WAITING", None, np.nan, "No current paired candle available"),
            ("Missing Candles", "WAITING", None, np.nan, "Continuous live feed unavailable"),
            ("Duplicate Candles", "WAITING", None, np.nan, "Continuous live feed unavailable"),
            ("Gap Detection", "WAITING", None, np.nan, "Continuous live feed unavailable"),
            ("ATM Availability", "BLOCKED", None, np.nan, "Current ATM contract not configured"),
            ("Expiry Status", "WAITING", None, np.nan, "Live contract expiry metadata unavailable"),
        ]
        return pd.DataFrame(rows, columns=[
            "feed", "status", "last_timestamp", "age_minutes", "detail"])

    def feature_health(self) -> pd.DataFrame:
        contract = read_json(self.root / "production_model/feature_contract.json")
        health = read_json(self.root / "logs/live_inference/pipeline_health.json")
        count = int(contract.get("feature_count", 0))
        valid = count if health.get("contract_status") == "VERIFIED" and health.get("feature_status") == "VALID" else 0
        failures = [] if valid == count else [health.get("last_error") or "Feature validation unavailable"]
        return pd.DataFrame([{
            "contract_version": contract.get("contract_version"),
            "required_features": count, "valid_features": valid,
            "missing_features": "[]", "unknown_features": "[]",
            "reordered_features": "[]", "null_features": "[]",
            "invalid_dtype": "[]", "exact_failures": json.dumps(failures),
            "generation_latency_ms": np.nan,
            "status": "READY" if valid == count and count else "BLOCKED",
        }])

    def prediction_health(self, predictions: pd.DataFrame) -> pd.DataFrame:
        if predictions.empty:
            return pd.DataFrame([{"period": str(self.now.date()), "predictions": 0,
                                  "trades": 0, "skipped": 0, "average_probability": np.nan,
                                  "highest_probability": np.nan, "lowest_probability": np.nan,
                                  "trade_pct": 0.0, "skip_pct": 0.0,
                                  "strategy_distribution": "{}", "regime_distribution": "{}",
                                  "hourly_distribution": "{}"}])
        today = predictions.loc[predictions.timestamp.dt.date == self.now.date()]
        probabilities = pd.to_numeric(today.probability, errors="coerce").dropna()
        trade = int(today.decision.eq("TRADE").sum()) if len(today) else 0
        skip = int(today.decision.eq("SKIP").sum()) if len(today) else 0
        total = len(today)
        if total and skip / total >= self.config.skipped_trade_warning_rate:
            self.alert("WARNING", f"Skip rate is {skip / total:.1%}.",
                       "Review feed, Decision Engine status and current market regime.")
        return pd.DataFrame([{
            "period": str(self.now.date()), "predictions": total, "trades": trade,
            "skipped": skip, "average_probability": probabilities.mean() if len(probabilities) else np.nan,
            "highest_probability": probabilities.max() if len(probabilities) else np.nan,
            "lowest_probability": probabilities.min() if len(probabilities) else np.nan,
            "trade_pct": trade / total if total else 0.0,
            "skip_pct": skip / total if total else 0.0,
            "strategy_distribution": json.dumps(today.strategy.value_counts().to_dict()),
            "regime_distribution": json.dumps(today.regime.value_counts().to_dict()),
            "hourly_distribution": json.dumps(today.timestamp.dt.hour.value_counts().sort_index().to_dict()),
        }])

    @staticmethod
    def probability_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
        edges = np.linspace(0, 1, bins + 1)
        ref = np.histogram(reference, bins=edges)[0] / max(len(reference), 1)
        cur = np.histogram(current, bins=edges)[0] / max(len(current), 1)
        ref, cur = np.clip(ref, 1e-6, None), np.clip(cur, 1e-6, None)
        return float(np.sum((cur - ref) * np.log(cur / ref)))

    def drift(self, predictions: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
        baseline_path = self.root / "data/prediction/oof/oof_predictions_catboost.parquet"
        baseline = pd.read_parquet(
            baseline_path, columns=["predicted_probability"]).predicted_probability
        current = pd.to_numeric(predictions.probability, errors="coerce").dropna() if len(predictions) else pd.Series(dtype=float)
        enough = len(current) >= self.config.minimum_drift_observations
        probability_psi = self.probability_psi(baseline, current) if enough else np.nan
        status = "WAITING" if not enough else ("STALE" if probability_psi > self.config.psi_warning else "READY")
        if status == "STALE":
            self.alert("WARNING", f"Probability PSI {probability_psi:.3f} exceeded tolerance.",
                       "Investigate data drift; do not retrain automatically.")
        if not enough:
            self.alert("INFO", "Probability drift sample is insufficient.",
                       f"Collect at least {self.config.minimum_drift_observations} true predictions.")
        row = {
            "timestamp": self.now.isoformat(), "feature_psi": np.nan,
            "feature_psi_status": "WAITING_NO_LIVE_FEATURE_VECTORS",
            "probability_psi": probability_psi, "prediction_distribution_status": status,
            "observations": len(current), "rolling_win_rate": np.nan,
            "rolling_profit_factor": np.nan, "rolling_drawdown": np.nan,
        }
        if len(trades):
            tail = trades.tail(50)
            wins, losses = tail.loc[tail.net_pnl > 0, "net_pnl"].sum(), -tail.loc[tail.net_pnl < 0, "net_pnl"].sum()
            equity = tail.net_pnl.cumsum()
            row.update({
                "rolling_win_rate": float(tail.net_pnl.gt(0).mean()),
                "rolling_profit_factor": float(wins / losses) if losses else float("inf"),
                "rolling_drawdown": float((equity - equity.cummax()).min()),
            })
        self.logger.write("drift", json.dumps(row, default=str))
        return pd.DataFrame([row])
