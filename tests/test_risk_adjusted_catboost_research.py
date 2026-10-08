from tools.run_risk_adjusted_catboost_research import FEATURES, load_data


def test_corrected_research_target_and_leakage_contract():
    data, leakage = load_data()
    assert set(data.RISK_ADJUSTED_5M.unique()) == {"WIN", "LOSS", "NEUTRAL"}
    assert data.label.equals(data.RISK_ADJUSTED_5M.eq("WIN").astype("int8"))
    assert len(FEATURES) == 48
    assert "nifty_vwap_distance" not in FEATURES
    assert leakage["all_prefix_invariant"]
    assert not leakage["label_columns_in_model_features"]
    assert leakage["future_data_contamination_count"] == 0
