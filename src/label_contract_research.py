"""Vectorized research of alternative outcome-label contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ContractSpec:
    contract_id: str
    family: str
    horizon: int
    stop_value: float
    risk_reward: float | None
    exit_type: str = "BARRIER"


class OpportunityMatrix:
    """Causal future-path matrix for frozen Stage-1 allowed entries."""

    def __init__(self, frame: pd.DataFrame, maximum_horizon: int = 20):
        data = frame.reset_index(drop=True)
        eligible = np.flatnonzero(data.trade_allowed.eq(1).to_numpy())
        timestamp = pd.to_datetime(data.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
        sessions = timestamp.dt.strftime("%Y%m%d").to_numpy()
        n = len(eligible)
        self.index = eligible
        self.entry = data.close.to_numpy(float)[eligible]
        self.direction = data.entry_signal.to_numpy(int)[eligible]
        self.atr = data.atr_14.to_numpy(float)[eligible]
        self.year = timestamp.dt.year.to_numpy()[eligible]
        self.regime = data.trend_state.astype(str).to_numpy()[eligible]
        minute = data.minutes_from_open.to_numpy(int)[eligible]
        self.session_segment = np.select([minute < 60, minute >= 315],
                                         ["OPENING", "CLOSING"], default="MID_SESSION")
        self.session_id = sessions[eligible]
        self.future_open = np.full((n, maximum_horizon), np.nan)
        self.future_high = np.full((n, maximum_horizon), np.nan)
        self.future_low = np.full((n, maximum_horizon), np.nan)
        self.future_close = np.full((n, maximum_horizon), np.nan)
        self.valid = np.zeros((n, maximum_horizon), dtype=bool)
        opens, highs = data.open.to_numpy(float), data.high.to_numpy(float)
        lows, closes = data.low.to_numpy(float), data.close.to_numpy(float)
        for offset in range(1, maximum_horizon + 1):
            future = eligible + offset
            exists = future < len(data)
            safe = np.minimum(future, len(data)-1)
            valid = exists & (sessions[safe] == sessions[eligible])
            self.valid[:, offset-1] = valid
            self.future_open[valid, offset-1] = opens[safe[valid]]
            self.future_high[valid, offset-1] = highs[safe[valid]]
            self.future_low[valid, offset-1] = lows[safe[valid]]
            self.future_close[valid, offset-1] = closes[safe[valid]]

    def evaluate(self, spec: ContractSpec) -> pd.DataFrame:
        h = spec.horizon
        valid = self.valid[:, :h]
        keep = valid.any(axis=1)
        entry, direction = self.entry.copy(), self.direction.copy()
        if spec.family in {"ATR", "ATR_TRAILING", "TIME_ATR"}:
            risk = self.atr * spec.stop_value
        else:
            risk = entry * spec.stop_value
        if not np.isfinite(risk).all() or (risk <= 0).any():
            raise ValueError("Invalid risk distance")
        n = len(entry)
        last_offset = valid.sum(axis=1)
        last_col = np.maximum(last_offset-1, 0)
        exit_price = self.future_close[np.arange(n), last_col]
        outcome = np.full(n, "TIMEOUT", dtype=object)
        holding = last_offset.copy()
        unresolved = keep.copy()
        mfe = np.zeros(n); mae = np.zeros(n)

        if spec.exit_type == "TRAILING":
            trailing = np.where(direction == 1, entry-risk, entry+risk)
            for column in range(h):
                active = unresolved & valid[:, column]
                high, low, candle_open = (self.future_high[:, column], self.future_low[:, column],
                                          self.future_open[:, column])
                favorable = np.where(direction == 1, high-entry, entry-low)
                adverse = np.where(direction == 1, entry-low, high-entry)
                mfe[active] = np.maximum(mfe[active], favorable[active])
                mae[active] = np.maximum(mae[active], adverse[active])
                hit = active & np.where(direction == 1, low <= trailing, high >= trailing)
                fill = np.where(direction == 1, np.minimum(candle_open, trailing),
                                np.maximum(candle_open, trailing))
                exit_price[hit] = fill[hit]; holding[hit] = column+1; outcome[hit] = "STOP"
                unresolved[hit] = False
                survived = active & ~hit
                trailing[survived] = np.where(
                    direction[survived] == 1,
                    np.maximum(trailing[survived], high[survived]-risk[survived]),
                    np.minimum(trailing[survived], low[survived]+risk[survived]))
        elif spec.exit_type == "TIME":
            for column in range(h):
                active = valid[:, column]
                favorable = np.where(direction == 1, self.future_high[:, column]-entry,
                                      entry-self.future_low[:, column])
                adverse = np.where(direction == 1, entry-self.future_low[:, column],
                                    self.future_high[:, column]-entry)
                mfe[active] = np.maximum(mfe[active], favorable[active])
                mae[active] = np.maximum(mae[active], adverse[active])
        else:
            target_r = float(spec.risk_reward)
            stop = np.where(direction == 1, entry-risk, entry+risk)
            target = np.where(direction == 1, entry+risk*target_r, entry-risk*target_r)
            for column in range(h):
                active = unresolved & valid[:, column]
                high, low, candle_open = (self.future_high[:, column], self.future_low[:, column],
                                          self.future_open[:, column])
                favorable = np.where(direction == 1, high-entry, entry-low)
                adverse = np.where(direction == 1, entry-low, high-entry)
                mfe[active] = np.maximum(mfe[active], favorable[active])
                mae[active] = np.maximum(mae[active], adverse[active])
                hit_stop = active & np.where(direction == 1, low <= stop, high >= stop)
                stop_fill = np.where(direction == 1, np.minimum(candle_open, stop),
                                     np.maximum(candle_open, stop))
                exit_price[hit_stop] = stop_fill[hit_stop]; holding[hit_stop] = column+1
                outcome[hit_stop] = "STOP"; unresolved[hit_stop] = False
                hit_target = unresolved & active & np.where(direction == 1, high >= target, low <= target)
                target_fill = np.where(direction == 1, np.maximum(candle_open, target),
                                       np.minimum(candle_open, target))
                exit_price[hit_target] = target_fill[hit_target]; holding[hit_target] = column+1
                outcome[hit_target] = "WIN"; unresolved[hit_target] = False

        result = pd.DataFrame({
            "contract_id": spec.contract_id, "family": spec.family, "horizon": h,
            "direction": np.where(direction == 1, "BUY_CALL", "BUY_PUT"), "year": self.year,
            "regime": self.regime, "session_segment": self.session_segment,
            "session_id": self.session_id, "outcome": outcome, "holding_candles": holding,
            "mfe_r": mfe/risk, "mae_r": mae/risk,
            "realized_r": direction*(exit_price-entry)/risk,
            "entry_price": entry, "exit_price": exit_price, "risk_points": risk,
        })
        return result.loc[keep].reset_index(drop=True)


def summarize_contract(data: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in data.groupby(groups, sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append({**dict(zip(groups, keys)), "observations": len(group),
                     "win_rate": group.realized_r.gt(0).mean(),
                     "loss_rate": group.realized_r.lt(0).mean(),
                     "stop_rate": group.outcome.eq("STOP").mean(),
                     "timeout_rate": group.outcome.eq("TIMEOUT").mean(),
                     "expected_r": group.realized_r.mean(), "average_mfe_r": group.mfe_r.mean(),
                     "average_mae_r": group.mae_r.mean(),
                     "average_holding_candles": group.holding_candles.mean()})
    return pd.DataFrame(rows)


def contract_grid() -> list[ContractSpec]:
    specs = []
    for horizon in (5, 10, 15, 20):
        for stop in (0.0010, 0.0015, 0.0020, 0.0025):
            for rr in (1.0, 1.25, 1.5, 2.0):
                specs.append(ContractSpec(f"FIXED_H{horizon}_SL{stop:.4f}_RR{rr:.2f}",
                                          "FIXED", horizon, stop, rr))
        for stop_atr in (0.5, 0.75, 1.0, 1.25):
            for rr in (1.0, 1.25, 1.5, 2.0):
                specs.append(ContractSpec(f"ATR_H{horizon}_SL{stop_atr:.2f}_RR{rr:.2f}",
                                          "ATR", horizon, stop_atr, rr))
        for trailing in (0.5, 0.75, 1.0, 1.25):
            specs.append(ContractSpec(f"TRAIL_H{horizon}_ATR{trailing:.2f}", "ATR_TRAILING",
                                      horizon, trailing, None, "TRAILING"))
        specs.append(ContractSpec(f"TIME_H{horizon}_FIXED_RISK", "TIME_FIXED", horizon,
                                  0.002, None, "TIME"))
        specs.append(ContractSpec(f"TIME_H{horizon}_ATR_RISK", "TIME_ATR", horizon,
                                  1.0, None, "TIME"))
    return specs
