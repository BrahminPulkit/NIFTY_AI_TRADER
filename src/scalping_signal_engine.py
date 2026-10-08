"""Unified CE/PE paper-signal approval with fail-closed model contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from catboost import CatBoostClassifier
import joblib
import pandas as pd

from src.feature_pipeline import FEATURE_VERSION, build_features
from src.option_contract_pipeline import (
    normalize_option_contracts,
    select_option_contract,
    validate_artifact_manifest,
)
from src.option_feature_pipeline import OPTION_FEATURES, OPTION_FEATURE_VERSION, build_option_features
from src.risk_manager import RiskManager
from src.scalping_candidate_engine import CandidateConfig, generate_scalping_candidates

SCALPING_FEATURE_VERSION = f"{FEATURE_VERSION}|{OPTION_FEATURE_VERSION}"


@dataclass(frozen=True)
class SignalConfig:
    minimum_strategy_score: int = 50
    minimum_model_probability: float = 0.50
    maximum_data_age_seconds: int = 120
    option_stop_pct: float = 0.03


class ValidatedSideModel:
    def __init__(self, root: str | Path, expected_side: str):
        root = Path(root)
        contract = json.loads((root / "feature_contract.json").read_text(encoding="utf-8"))
        if contract.get("option_type") != expected_side:
            raise ValueError("MODEL_VALIDATION_BLOCKED: model side mismatch")
        self.feature_order = contract["feature_order"]
        validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
        if validation.get("status") != "FORWARD_PAPER_CANDIDATE":
            raise ValueError("MODEL_VALIDATION_BLOCKED: candidate has not passed walk-forward validation")
        self.manifest = validate_artifact_manifest(
            root / "artifact_manifest.json", feature_order=self.feature_order,
            feature_version=SCALPING_FEATURE_VERSION)
        self.preprocessor = joblib.load(root / "preprocessor.joblib")
        self.model = CatBoostClassifier(); self.model.load_model(root / "model.cbm")

    def probability(self, values: dict) -> float:
        missing = [name for name in self.feature_order if name not in values]
        if missing:
            raise ValueError(f"MODEL_VALIDATION_BLOCKED: missing inference features {missing}")
        row = pd.DataFrame([[values[name] for name in self.feature_order]], columns=self.feature_order)
        return float(self.model.predict_proba(self.preprocessor.transform(row))[0, 1])


class ScalpingSignalEngine:
    def __init__(self, models: dict[str, object], *, config: SignalConfig = SignalConfig(),
                 candidate_config: CandidateConfig = CandidateConfig(), risk: RiskManager | None = None,
                 journal_path: str | Path | None = None):
        self.models, self.config, self.candidate_config = models, config, candidate_config
        self.risk = risk or RiskManager()
        self.journal_path = Path(journal_path) if journal_path else None

    def evaluate(self, index: pd.DataFrame, options: pd.DataFrame, *, now=None) -> list[dict]:
        features = build_features(index)
        contracts = normalize_option_contracts(options)
        candidates = generate_scalping_candidates(features, self.candidate_config)
        if candidates.empty:
            return self._record([self._reject(features.timestamp.iloc[-1], "NO_CANDIDATE")])
        current_time = pd.Timestamp(features.timestamp.iloc[-1])
        now = current_time if now is None else pd.Timestamp(now)
        now = now.tz_localize("Asia/Kolkata") if now.tzinfo is None else now.tz_convert("Asia/Kolkata")
        current = candidates.loc[candidates.timestamp.eq(current_time)]
        if current.empty:
            return self._record([self._reject(current_time, "NO_CANDIDATE")])
        results = []
        for candidate in current.itertuples(index=False):
            side = candidate.option_type
            try:
                option = select_option_contract(
                    contracts, timestamp=current_time, option_type=side,
                    underlying_price=candidate.index_price,
                )
            except LookupError:
                results.append(self._reject(current_time, "OPTION_CONTRACT_UNAVAILABLE", candidate))
                continue
            index_age = (now - current_time).total_seconds()
            option_age = (now - option.timestamp).total_seconds()
            if current_time != option.timestamp:
                results.append(self._reject(current_time, "TIMESTAMP_MISMATCH", candidate))
                continue
            if index_age > self.config.maximum_data_age_seconds or option_age > self.config.maximum_data_age_seconds:
                results.append(self._reject(current_time, "STALE_DATA", candidate, option, index_age, option_age))
                continue
            option_confirmation = 15 if float(option.close) > float(option.open) else 0
            risk_quality = 10 if float(option.close) > 0 and self.config.option_stop_pct > 0 else 0
            score = min(100, int(candidate.base_strategy_score) + option_confirmation + risk_quality)
            if score < self.config.minimum_strategy_score:
                results.append(self._reject(current_time, "STRATEGY_SCORE_LOW", candidate, option,
                                            index_age, option_age, score))
                continue
            model = self.models.get(side)
            if model is None:
                results.append(self._reject(current_time, "MODEL_VALIDATION_BLOCKED", candidate, option,
                                            index_age, option_age, score))
                continue
            option_history = contracts.loc[
                contracts.contract_segment_id.eq(option.contract_segment_id)
                & contracts.timestamp.le(current_time)]
            option_features = build_option_features(option_history).iloc[-1]
            values = features.iloc[-1].to_dict()
            values.update({f"option_{name}": option_features[name] for name in OPTION_FEATURES})
            values.update({"strategy": candidate.strategy, "strategy_score": score,
                           "direction": candidate.direction})
            probability = float(model.probability(values))
            if probability < self.config.minimum_model_probability:
                results.append(self._reject(current_time, "MODEL_PROBABILITY_LOW", candidate, option,
                                            index_age, option_age, score, probability))
                continue
            plan, risk_reason = self.risk.create_plan(
                1, float(option.close), float(option.close) * (1 - self.config.option_stop_pct),
                current_time.date())
            if plan is None:
                results.append(self._reject(current_time, f"RISK_{risk_reason.upper().replace(' ', '_')}",
                                            candidate, option, index_age, option_age, score, probability))
                continue
            results.append({
                "timestamp": current_time.isoformat(), "strategy": candidate.strategy,
                "direction": "BULLISH" if candidate.direction == 1 else "BEARISH",
                "candidate_created": True, "strategy_score": score, "model_probability": probability,
                "score_components": {
                    **dict(candidate.score_components),
                    "option_confirmation": option_confirmation,
                    "risk_quality": risk_quality,
                },
                "required_probability": self.config.minimum_model_probability,
                "selected_option": self._option(option), "risk_status": "APPROVED",
                "final_decision": f"BUY {side}", "primary_rejection_reason": None,
                "index_candle_timestamp": current_time.isoformat(),
                "option_candle_timestamp": option.timestamp.isoformat(),
                "index_data_age_seconds": index_age, "option_data_age_seconds": option_age,
            })
        return self._record(results)

    def _record(self, results: list[dict]) -> list[dict]:
        if self.journal_path:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            with self.journal_path.open("a", encoding="utf-8") as stream:
                for result in results:
                    stream.write(json.dumps(result, sort_keys=True, default=str) + "\n")
        return results

    @staticmethod
    def _option(row) -> dict:
        return {name: row[name] for name in ("underlying", "option_type", "strike", "expiry",
                                              "security_id", "contract_segment_id")}

    def _reject(self, timestamp, reason, candidate=None, option=None, index_age=None,
                option_age=None, score=None, probability=None) -> dict:
        return {
            "timestamp": pd.Timestamp(timestamp).isoformat(),
            "strategy": getattr(candidate, "strategy", None),
            "direction": ("BULLISH" if getattr(candidate, "direction", 0) == 1 else
                          "BEARISH" if getattr(candidate, "direction", 0) == -1 else "NEUTRAL"),
            "candidate_created": candidate is not None, "strategy_score": score,
            "model_probability": probability, "required_probability": self.config.minimum_model_probability,
            "selected_option": None if option is None else self._option(option),
            "risk_status": "NOT_RUN", "final_decision": "NO TRADE",
            "primary_rejection_reason": reason,
            "index_candle_timestamp": pd.Timestamp(timestamp).isoformat(),
            "option_candle_timestamp": None if option is None else option.timestamp.isoformat(),
            "index_data_age_seconds": index_age, "option_data_age_seconds": option_age,
        }
