from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from institutional_hybrid.experiments.phase1_engine import (
    EXPERIMENTS, build_signals, load_common, rules_hash,
)


def test_six_ablation_experiments_and_fixed_rules():
    assert len(EXPERIMENTS) == 6
    common = load_common()
    assert common["risk"]["stop_pct"] == .04
    assert common["risk"]["target_pct"] == .08
    assert len(rules_hash(common)) == 64


def test_filters_are_conjunctive_and_rejections_measured():
    idx = pd.date_range("2025-01-01 09:20", periods=3, freq="min")
    frame = pd.DataFrame({
        "ema_pullback": [True, True, False], "trend_direction": [1, 1, 1],
        "false_break": [True, False, True], "market_structure_shift": [0, 0, 0],
        "atr_expansion": [1.2, 1.2, 1.2], "range_expansion": [1.3, 1.3, 1.3],
        "breakout_direction": [1, 1, 1], "compression_breakout": [False] * 3,
        "relative_volume": [2.0, 2.0, 2.0], "premium_expansion_score": [70] * 3,
    }, index=idx)
    out = build_signals(frame, ["ema_pullback", "liquidity"], load_common())
    assert out["signal"].tolist() == [1, 0, 0]
    assert out["rejection_reason"].tolist() == ["", "liquidity", ""]
