import pandas as pd

from tools.run_relationship_feature_audit import relationship_features


def test_relationship_features_are_current_row_formulas_without_future_shift():
    frame = pd.DataFrame({
        **{f"{p}_return_{h}m": [v] for p, v in (("nifty", 1.), ("ce", 2.), ("pe", -1.)) for h in (1, 3, 5)},
        **{f"{p}_momentum_{h}m": [v] for p, v in (("nifty", 1.), ("ce", 2.), ("pe", -1.)) for h in (1, 3, 5)},
        "ce_volatility": [2.], "pe_volatility": [1.], "ce_atr_pct": [3.],
        "pe_atr_pct": [1.], "ce_volume_ratio": [2.], "pe_volume_ratio": [1.],
    })
    result = relationship_features(frame).iloc[0]
    assert result.ce_response_vs_nifty_1m == 1
    assert result.pe_response_vs_nifty_1m == 0
    assert result.ce_pe_return_spread_1m == 3
    assert result.ce_pe_volume_imbalance == 1 / 3
