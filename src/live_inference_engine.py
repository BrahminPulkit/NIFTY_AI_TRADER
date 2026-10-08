"""True native-CatBoost inference with strict fail-closed orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import json
import threading

import pandas as pd

from src.live_feature_generator import build_live_contract_row
from src.live_pipeline_monitor import LivePipelineMonitor
from src.production_predictor import ContractViolation, ProductionPredictor


class SafeBlock(RuntimeError):
    pass


@dataclass(frozen=True)
class LiveInferenceResult:
    timestamp: str
    index: str
    option: str
    probability: float | None
    prediction: int
    confidence: float | None
    decision_engine_output: str
    decision: str
    risk_status: str
    reason: str
    strategy: str
    regime: str
    latency_ms: float
    model_version: str
    inference_timestamp: str


class AppendOnlyLiveJournal:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, result: LiveInferenceResult) -> None:
        payload = asdict(result)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")


class TrueLiveInferenceEngine:
    def __init__(self, package_dir: str | Path, journal_path: str | Path,
                 monitor: LivePipelineMonitor):
        self.monitor = monitor
        try:
            self.predictor = ProductionPredictor(package_dir)
        except Exception as exc:
            monitor.safe_block(exc)
            raise SafeBlock(f"Frozen model package failed verification: {exc}") from exc
        self.journal = AppendOnlyLiveJournal(journal_path)
        self.threshold = float(self.predictor.metadata["approved_probability_threshold"])
        if self.threshold != 0.9:
            raise SafeBlock("Approved production threshold must equal 0.90")
        monitor.update(status="READY", model_status="VERIFIED",
                       contract_status="VERIFIED", journal_status="READY")

    def infer_contract_row(self, row: pd.DataFrame, *, timestamp,
                           index_name="NIFTY", option_name="ATM_CALL",
                           stage1_allowed=True, stage1_reason="ALLOWED") -> LiveInferenceResult:
        start = perf_counter()
        inference_time = datetime.now(timezone.utc).isoformat()
        try:
            if not stage1_allowed:
                result = LiveInferenceResult(
                    str(timestamp), index_name, option_name, None, 0, None,
                    "NOT_RUN", "SKIP", "SAFE", stage1_reason, "NONE", "NONE",
                    (perf_counter() - start) * 1000,
                    self.predictor.metadata["model_version"], inference_time)
            else:
                output = self.predictor.predict(row)
                probability = float(output["probability"])
                decision_engine = str(row.iloc[0]["recommendation"])
                if decision_engine != "TRADE":
                    decision, reason = "SKIP", f"DECISION_ENGINE_{row.iloc[0]['decision_reason']}"
                elif probability < self.threshold:
                    decision, reason = "SKIP", "PROBABILITY_BELOW_APPROVED_THRESHOLD"
                else:
                    decision, reason = "TRADE", "ALL_FROZEN_GATES_PASSED"
                result = LiveInferenceResult(
                    str(timestamp), index_name, option_name, probability,
                    int(probability >= .5), probability, decision_engine, decision,
                    "APPROVED" if decision == "TRADE" else "SAFE",
                    reason, str(row.iloc[0]["current_strategy"]),
                    str(row.iloc[0]["market_state"]), (perf_counter() - start) * 1000,
                    output["model_version"], inference_time)
            self.journal.append(result)
            self.monitor.update(
                status="HEALTHY", feature_status="VALID", decision_status=result.decision_engine_output,
                journal_status="APPENDED", last_timestamp=str(timestamp),
                last_latency_ms=result.latency_ms, last_error=None)
            return result
        except (ContractViolation, Exception) as exc:
            self.monitor.safe_block(exc)
            raise SafeBlock(str(exc)) from exc

    def infer_histories(self, index_history: pd.DataFrame, option_history: pd.DataFrame,
                        decision_observations: pd.DataFrame) -> LiveInferenceResult:
        self.monitor.update(data_status="SYNCHRONISING", feature_status="BUILDING",
                            decision_status="BUILDING")
        expected = self.predictor.contract["feature_order"]
        row, audit = build_live_contract_row(
            index_history, option_history, decision_observations, expected)
        if row is None:
            return self.infer_contract_row(
                pd.DataFrame(columns=expected), timestamp=audit["timestamp"],
                stage1_allowed=False, stage1_reason=audit["reason"])
        return self.infer_contract_row(row, timestamp=audit["timestamp"])
