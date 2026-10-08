from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from institutional_hybrid.backtests import ResearchBacktester
from institutional_hybrid.indicators.engines import liquidity_features
from institutional_hybrid.strategies import InstitutionalHybridStrategy
from institutional_hybrid.utils.io import LAB_ROOT, assert_lab_output


def frames(n=260):
    idx = pd.date_range("2025-01-01 09:15", periods=n, freq="min")
    close = 22000 + np.linspace(0, 150, n) + np.sin(np.arange(n) / 5) * 8
    futures = pd.DataFrame({
        "open": close - 1, "high": close + 4, "low": close - 4,
        "close": close, "volume": 1000 + (np.arange(n) % 17) * 75,
    }, index=idx)
    options = pd.DataFrame({
        "bid": 99.5, "ask": 100.5, "volume": 1500 + np.arange(n),
        "open_interest": 10000 + np.arange(n) * 5, "delta": .5,
        "gamma": .02, "theta": -.1, "iv": .14,
        "premium": 100 + np.sin(np.arange(n) / 7) * 6,
    }, index=idx)
    return futures, options


def test_no_lookahead_in_prior_high():
    futures, _ = frames(30)
    result = liquidity_features(futures, 5)
    assert result["prior_high"].iloc[10] == futures["high"].iloc[5:10].max()


def test_score_and_backtest_contract():
    futures, options = frames()
    strategy = InstitutionalHybridStrategy(threshold=35)
    scored = strategy.score(strategy.build_features(futures, options))
    assert scored["institutional_score"].dropna().between(0, 100).all()
    trades, metrics = ResearchBacktester().run(scored)
    assert set(["win_rate", "profit_factor", "maximum_drawdown"]).issubset(metrics)
    assert isinstance(trades, pd.DataFrame)


def test_output_guard():
    assert assert_lab_output(LAB_ROOT / "reports") == (LAB_ROOT / "reports").resolve()
    try:
        assert_lab_output(LAB_ROOT.parent / "forbidden.csv")
    except ValueError:
        pass
    else:
        raise AssertionError("output guard allowed path outside lab")
