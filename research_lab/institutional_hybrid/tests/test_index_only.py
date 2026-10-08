from pathlib import Path
import sys
import pandas as pd

LAB=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(LAB.parent))

from institutional_hybrid.index_only.data import load_and_validate
from institutional_hybrid.index_only.backtest import run
from institutional_hybrid.index_only.data import config


def test_index_audit_is_read_only_and_approved_with_quarantine():
    frame,audit=load_and_validate()
    assert audit["approved"]
    assert audit["corrupted_rows"]==11
    assert frame.quarantined.sum()==11
    assert audit["synthetic_rows"]==0


def test_next_open_and_no_gap_entry():
    idx=pd.to_datetime(["2025-01-01 09:15+05:30","2025-01-01 09:16+05:30",
                        "2025-01-01 09:18+05:30"])
    x=pd.DataFrame({
        "open":[100,101,102],"high":[101,102,103],"low":[99,100,101],"close":[100,101,102],
        "signal":[1,1,0],"quarantined":[False]*3,"strategy_score":[50]*3,
        "regime":["bull_trend"]*3,
    },index=idx)
    trades=run(x,config())
    assert len(trades)==1
    assert trades.iloc[0].entry_time==idx[1]
    assert not (trades.entry_time==idx[2]).any()
