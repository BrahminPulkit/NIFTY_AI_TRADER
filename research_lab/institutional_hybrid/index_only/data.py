"""Read-only index loader, audit and causal feature construction."""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

from ..indicators.engines import (
    breakout_features, ema_pullback_features, futures_volume_features,
    liquidity_features, momentum_features, regime_features,
)

ROOT=Path(__file__).resolve().parent


def config():
    return yaml.safe_load((ROOT/"config.yaml").read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    h=sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def load_and_validate():
    cfg=config(); path=Path(cfg["source"]).resolve(); before=path.stat()
    raw=pd.read_csv(path,low_memory=False)
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise RuntimeError("Index source changed during read")
    required={"timestamp","open","high","low","close","volume"}
    missing=required-set(raw.columns)
    if missing: raise ValueError(f"Index schema missing: {sorted(missing)}")
    raw["timestamp"]=pd.to_datetime(raw.timestamp,unit="s",utc=True,errors="coerce").dt.tz_convert(cfg["timezone"])
    raw=raw.set_index("timestamp").sort_index()
    session=raw.between_time(cfg["session"]["start"],cfg["session"]["end"]).copy()
    prices=session[["open","high","low","close"]]
    corrupt=(prices.high<prices[["open","close","low"]].max(axis=1))|(
        prices.low>prices[["open","close","high"]].min(axis=1))
    corrupt|=prices.isna().any(axis=1)|prices.le(0).any(axis=1)
    session["quarantined"]=corrupt
    counts=session.groupby(session.index.normalize()).size()
    expected=375
    audit={
        "source":str(path),"sha256":file_hash(path),"raw_rows":len(raw),
        "session_rows":len(session),"start":str(session.index.min()),"end":str(session.index.max()),
        "trading_days":int(session.index.normalize().nunique()),
        "duplicate_timestamps":int(session.index.duplicated().sum()),
        "monotonic":bool(session.index.is_monotonic_increasing),
        "timezone":str(session.index.tz),"corrupted_rows":int(corrupt.sum()),
        "corrupted_pct":float(corrupt.mean()*100),
        "missing_candles":int((expected-counts.clip(upper=expected)).clip(lower=0).sum()),
        "incomplete_days":int((counts<expected).sum()),
        "zero_volume_rows":int(session.volume.eq(0).sum()),
        "synthetic_rows":0,
    }
    audit["approved"]=(
        audit["duplicate_timestamps"]==cfg["integrity"]["duplicate_tolerance"]
        and audit["monotonic"] and audit["timezone"]==cfg["timezone"]
        and audit["corrupted_pct"]<=cfg["integrity"]["maximum_corrupt_pct"]
    )
    return session,audit


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    x=regime_features(frame)
    x=ema_pullback_features(x); x=liquidity_features(x)
    x=momentum_features(x); x=breakout_features(x); x=futures_volume_features(x)
    x["score_regime"]=x.regime.isin(["bull_trend","bear_trend","expansion","gap_day"]).astype(float)*20
    x["score_trend"]=((x.trend_direction!=0)&(x.ema_pullback|x.continuation)).astype(float)*15
    x["score_liquidity"]=(x.false_break|(x.market_structure_shift!=0)).astype(float)*15
    x["score_momentum"]=((x.atr_expansion>=1.1)&(x.range_expansion>=1.2)).astype(float)*15
    x["score_breakout"]=(x.breakout_direction!=0).astype(float)*10
    x["score_volume"]=((x.relative_volume>=1.5)&x.volume.gt(0)).astype(float)*10
    x["institutional_score"]=x[[
        "score_regime","score_trend","score_liquidity","score_momentum",
        "score_breakout","score_volume",
    ]].sum(axis=1)
    return x
