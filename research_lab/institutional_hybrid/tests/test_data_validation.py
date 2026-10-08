from __future__ import annotations
import ast
import json
from pathlib import Path
import sys
import pandas as pd

LAB=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(LAB.parent))

from institutional_hybrid.data_validation.alignment import align_option_to_index
from institutional_hybrid.data_validation.engine import InstitutionalDataValidator
from institutional_hybrid.utils.research_gate import ResearchDataGateError, require_approved_data


def test_exact_and_backward_alignment_never_use_future():
    tz="Asia/Kolkata"
    idx=pd.date_range("2025-01-01 09:15",periods=3,freq="min",tz=tz)
    index=pd.DataFrame({"close":[1,2,3]},index=idx)
    option=pd.DataFrame({"close":[10,20]},index=[idx[0],idx[1]+pd.Timedelta(seconds=30)])
    exact=align_option_to_index(index,option,0)
    assert exact.option_timestamp.notna().tolist()==[True,False,False]
    backward=align_option_to_index(index,option,40)
    assert (backward.loc[backward.option_timestamp.notna(),"option_timestamp"]<=
            backward.loc[backward.option_timestamp.notna()].index).all()
    assert backward.alignment_lag_seconds.dropna().ge(0).all()


def test_real_audit_is_deterministic_and_fails_for_exact_reasons():
    first=InstitutionalDataValidator().run()[0]
    second=InstitutionalDataValidator().run()[0]
    assert json.dumps(first,sort_keys=True)==json.dumps(second,sort_keys=True)
    assert first["status"]=="FAIL"
    assert not first["can_research_continue"]
    reasons=" ".join(first["critical_issues"])
    assert "corrupted OHLC rows=11" in reasons
    assert "IV anomalies=502" in reasons
    assert "strike farther than configured ATM limit on 1 rows" in reasons
    assert "rolling strike changes intraday 42476 times" in reasons


def test_validation_package_has_no_strategy_or_production_imports():
    banned={"strategies","traders","backtests","src","production_model","pages","app"}
    for path in (LAB/"data_validation").rglob("*.py"):
        tree=ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.module:
                assert not set(node.module.split("."))&banned
            if isinstance(node,ast.Import):
                for name in node.names: assert not set(name.name.split("."))&banned


def test_research_gate_denies_current_data():
    try:
        require_approved_data()
    except ResearchDataGateError:
        pass
    else:
        raise AssertionError("strategy gate unexpectedly approved invalid data")
