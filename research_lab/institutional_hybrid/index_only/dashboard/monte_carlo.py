"""Deterministic block-bootstrap capital stress testing."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    starting_capital: float = 10.0
    simulations: int = 2000
    trades_per_path: int = 250
    block_size: int = 5
    exposure_fraction: float = 1.0
    fixed_cost_rupees: float = 0.0
    ruin_fraction: float = 0.25
    seed: int = 42


def _indices(n:int,paths:int,horizon:int,block:int,rng) -> np.ndarray:
    blocks=int(np.ceil(horizon/block))
    starts=rng.integers(0,n,size=(paths,blocks))
    offsets=np.arange(block)
    return ((starts[:,:,None]+offsets)%n).reshape(paths,-1)[:,:horizon]


def simulate(returns, cfg:SimulationConfig) -> dict:
    r=np.asarray(returns,dtype=float)
    r=r[np.isfinite(r)]
    if not len(r): raise ValueError("At least one finite historical return is required")
    if cfg.starting_capital<10: raise ValueError("Starting capital must be at least ₹10")
    if not 0<cfg.ruin_fraction<1: raise ValueError("Ruin fraction must be between 0 and 1")
    rng=np.random.default_rng(cfg.seed)
    sampled=r[_indices(len(r),cfg.simulations,cfg.trades_per_path,cfg.block_size,rng)]
    equity=np.empty((cfg.simulations,cfg.trades_per_path+1),dtype=float)
    equity[:,0]=cfg.starting_capital
    for j in range(cfg.trades_per_path):
        equity[:,j+1]=np.maximum(
            equity[:,j]*(1+sampled[:,j]*cfg.exposure_fraction)-cfg.fixed_cost_rupees,0
        )
    peaks=np.maximum.accumulate(equity,axis=1)
    drawdowns=np.divide(equity,peaks,out=np.zeros_like(equity),where=peaks!=0)-1
    ruin_level=cfg.starting_capital*cfg.ruin_fraction
    final=equity[:,-1]
    return {
        "equity":equity,"final_equity":final,
        "path_max_drawdown":drawdowns.min(axis=1),
        "risk_of_ruin":float((equity.min(axis=1)<=ruin_level).mean()),
        "probability_of_success":float((final>cfg.starting_capital).mean()),
        "probability_of_profit_10pct":float((final>=cfg.starting_capital*1.10).mean()),
        "median_final":float(np.median(final)),
        "p05_final":float(np.quantile(final,.05)),
        "p95_final":float(np.quantile(final,.95)),
        "median_max_drawdown":float(np.median(drawdowns.min(axis=1))),
    }
