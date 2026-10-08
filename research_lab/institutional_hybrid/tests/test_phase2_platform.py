from __future__ import annotations

import ast
import importlib
from pathlib import Path
import sys
import numpy as np
import pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parent))

from institutional_hybrid.traders import TRADERS
from institutional_hybrid.traders.core import FixedExecution, backtest, frame_hash, guard_output, metrics


def fixture(n=80):
    idx=pd.date_range("2025-01-01 09:20",periods=n,freq="min")
    ones=pd.Series(1.0,index=idx); flags=pd.Series(False,index=idx)
    x=pd.DataFrame(index=idx)
    x["premium"]=100+np.sin(np.arange(n)/4)*5+np.arange(n)*.05
    x["atr_expansion"]=1.2; x["strategy_score"]=60.; x["regime"]="bull_trend"
    x["volume"]=1000.; x["open_interest"]=5000.; x["ema_pullback"]=flags
    x.loc[idx[::10],"ema_pullback"]=True
    x["trend_direction"]=1; x["continuation"]=True; x["trend_bar"]=True
    x["pullback_bar"]=False; x.loc[idx[1::10],"pullback_bar"]=True
    x["false_break"]=False; x["relative_volume"]=1.6; x["ema_spread_atr"]=.7
    x["range_expansion"]=1.3; x["momentum_3_atr"]=1.1
    x["compression_breakout"]=False; x.loc[idx[::15],"compression_breakout"]=True
    x["breakout_direction"]=1; x["market_structure_shift"]=0
    x["close"]=22000+np.arange(n); x["ema20"]=x["close"]-2
    for c in ("score_trend","score_regime","score_momentum","score_volume","score_breakout","score_liquidity"):
        x[c]=15.
    return x


def test_every_trader_has_required_files_and_is_deterministic():
    required={"strategy.py","rules.py","filters.py","backtest.py","analytics.py","charts.py","config.yaml","README.md"}
    x=fixture()
    for trader in TRADERS:
        folder=LAB/"traders"/trader
        assert required.issubset({p.name for p in folder.iterdir()})
        module=importlib.import_module(f"institutional_hybrid.traders.{trader}.strategy")
        first=module.signal(x.copy()); second=module.signal(x.copy())
        assert frame_hash(first)==frame_hash(second)


def test_signals_are_causal_under_future_mutation():
    x=fixture()
    cut=40
    for trader in TRADERS:
        module=importlib.import_module(f"institutional_hybrid.traders.{trader}.strategy")
        first=module.signal(x.copy()).iloc[:cut]
        changed=x.copy()
        changed.iloc[cut:,changed.columns.get_loc("premium")]=99999
        changed.iloc[cut:,changed.columns.get_loc("close")]=-99999
        second=module.signal(changed).iloc[:cut]
        pd.testing.assert_series_equal(first.signal,second.signal)


def test_no_production_imports_in_phase2_python():
    banned={"src","production_model","pages","app","paper_trading","live_inference"}
    for root in (LAB/"traders",LAB/"comparison_center",LAB/"hybrid_builder"):
        for path in root.rglob("*.py"):
            tree=ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):
                    assert not {n.name.split(".")[0] for n in node.names}&banned
                if isinstance(node,ast.ImportFrom) and node.module:
                    assert node.module.split(".")[0] not in banned


def test_output_guard_and_repeatable_backtest():
    x=fixture(); x["signal"]=0; x.loc[x.index[::10],"signal"]=1
    a=backtest(x,FixedExecution()); b=backtest(x,FixedExecution())
    pd.testing.assert_frame_equal(a,b)
    assert metrics(a)==metrics(b)
    try: guard_output(LAB.parent/"forbidden.csv")
    except ValueError: pass
    else: raise AssertionError("write outside lab accepted")


def test_no_outcome_proxy_leakage():
    for path in list((LAB/"traders").rglob("*.py"))+list((LAB/"hybrid_builder").rglob("*.py")):
        text=path.read_text(encoding="utf-8")
        assert "expected_premium_expansion" not in text
        assert "premium_expansion_probability" not in text
