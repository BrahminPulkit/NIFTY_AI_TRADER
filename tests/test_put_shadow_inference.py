import json

from src.put_shadow_inference import PutShadowInferenceEngine


def test_put_shadow_package_is_paper_only_and_temporally_validated():
    validation = json.loads(open(
        "models/stage2_put_candidate_v1/threshold_validation.json", encoding="utf-8").read())
    engine = PutShadowInferenceEngine("models/stage2_put_candidate_v1")
    assert validation["status"] == "FORWARD_PAPER_CANDIDATE"
    assert validation["live_orders_enabled"] is False
    assert engine.threshold == .85


def test_put_shadow_skip_never_claims_broker_approval():
    engine = PutShadowInferenceEngine("models/stage2_put_candidate_v1")
    result = engine._skip("2026-08-07 10:00+05:30", "NO_SETUP")
    assert result["action"] == "NO TRADE"
    assert result["paper_only"] is True
    assert result["instrument_scope"] == "ATM_PUT_PAPER_ONLY"
