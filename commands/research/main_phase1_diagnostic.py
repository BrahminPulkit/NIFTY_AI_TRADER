"""Read-only-data Phase-1 correctness funnel; writes only the requested report."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.scalping_candidate_engine import CandidateConfig, STRATEGIES, generate_scalping_candidates


def _missing(frame: pd.DataFrame, name: str) -> int:
    return len(frame) if name not in frame else int(frame[name].isna().sum())


def build_report() -> dict:
    features = pd.read_parquet(ROOT / "data/features/feature_dataset.parquet")
    candidates = generate_scalping_candidates(features, CandidateConfig(minimum_candidate_score=35))
    approved = candidates.loc[candidates.base_strategy_score.ge(50)]
    aligned = pd.read_parquet(ROOT / "data/aligned/canonical_index_option_aligned.parquet")
    put = pd.read_parquet(ROOT / "data/processed/nifty_atm_put_rolling_1min.parquet")
    call_missing = {name: _missing(aligned, name) for name in
                    ("strike", "expiry", "option_type", "security_id", "underlying")}
    put_missing = {name: _missing(put, name) for name in
                   ("strike", "expiry", "option_type", "security_id", "underlying")}
    invalid_call = max(call_missing.values())
    invalid_put = max(put_missing.values())
    strategy = []
    for name in STRATEGIES:
        group = candidates.loc[candidates.strategy.eq(name)]
        strategy.append({
            "strategy": name, "candidates": len(group),
            "strategy_score_approved": int(group.base_strategy_score.ge(50).sum()),
            "accepted_trades": 0, "win_rate": None, "average_r": None,
            "expectancy": None, "average_holding_time": None, "max_drawdown": None,
            "measurement_status": "BLOCKED_OPTION_CONTRACT_SCHEMA_INCOMPLETE",
        })
    score_low = len(candidates) - len(approved)
    report = {
        "status": "MODEL_VALIDATION_BLOCKED", "profitability_claim": False,
        "funnel": {
            "raw_candles": len(features), "candidates": len(candidates),
            "strategy_approved": len(approved), "ml_scored": 0, "option_approved": 0,
            "risk_approved": 0, "final_buy_ce": 0, "final_buy_pe": 0,
            "no_trade": len(approved),
        },
        "rejection_distribution": [
            {"reason": "OPTION_CONTRACT_SCHEMA_INVALID", "count": len(approved)},
            {"reason": "STRATEGY_SCORE_LOW", "count": score_low},
            {"reason": "MODEL_VALIDATION_BLOCKED", "count": len(approved)},
        ],
        "direction": {
            "ce_candidates": int(candidates.direction.eq(1).sum()),
            "pe_candidates": int(candidates.direction.eq(-1).sum()),
            "ce_final": 0, "pe_final": 0,
        },
        "model_probability_distribution": {
            "<50%": 0, "50-60%": 0, "60-70%": 0, "70-80%": 0,
            "80-90%": 0, "90%+": 0, "status": "NO_COMPATIBLE_MODEL_SCORED",
        },
        "strategy": strategy,
        "data_integrity": {
            "contract_mismatch_count": invalid_call + invalid_put,
            "missing_strike_count": call_missing["strike"] + put_missing["strike"],
            "missing_expiry_count": call_missing["expiry"] + put_missing["expiry"],
            "timestamp_mismatch_count": abs(len(features) - len(aligned)),
            "stale_data_count": 0,
            # No corrected artifact can be compared until the mandatory option identity exists.
            "feature_mismatch_count": 0,
            "feature_contract_comparisons": 0,
            "call_rows": len(aligned), "put_rows": len(put),
            "call_missing_by_field": call_missing, "put_missing_by_field": put_missing,
        },
        "interpretation": (
            "Candidate detection is measurable. Model, option, risk and outcome metrics are intentionally "
            "blocked because the available historical option files do not satisfy the mandatory contract schema."
        ),
    }
    return report


if __name__ == "__main__":
    result = build_report()
    destination = ROOT / "reports/phase1_scalping_correctness/diagnostic.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
