import pandas as pd

from tools.run_corrected_scalping_label_audit import build_risk_adjusted_label


def observation(**overrides):
    values = {
        "option_type": "CE", "ce_atr_pct": 4.0, "pe_atr_pct": 4.0,
        "nifty_atr_pct": 0.1, "selected_return_5m_pct": 4.0,
        "opposite_return_5m_pct": -2.0, "nifty_directional_return_5m_pct": 0.2,
        "mfe_5m_pct": 6.0, "mae_5m_pct": -1.0,
    }
    values.update(overrides)
    return pd.DataFrame([values])


def test_confirmed_risk_adjusted_opportunity_is_win():
    result = build_risk_adjusted_label(observation(), 5)
    assert result.loc[0, "RISK_ADJUSTED_5M"] == "WIN"


def test_conflicting_relative_response_is_neutral_not_win():
    result = build_risk_adjusted_label(
        observation(opposite_return_5m_pct=8.0, nifty_directional_return_5m_pct=-0.2), 5)
    assert result.loc[0, "RISK_ADJUSTED_5M"] == "NEUTRAL"


def test_missing_horizon_path_remains_ambiguous():
    result = build_risk_adjusted_label(observation(selected_return_5m_pct=float("nan")), 5)
    assert result.loc[0, "RISK_ADJUSTED_5M"] == "AMBIGUOUS"
