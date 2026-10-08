"""Step 19F: causal adaptive strategy selection research layer."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.strategy_discovery import classify_entry_context


RESEARCH_HORIZON_MINUTES = 15
RESEARCH_EXPANSION_TARGET = 0.05
MINIMUM_HISTORY = 30
STRATEGIES = (
    "PULLBACK_BREAKOUT",
    "MOMENTUM_BREAKOUT",
    "EXTENDED_BREAKOUT",
    "CONTINUATION_BREAKOUT",
)


def build_decision_observations(
    setups: pd.DataFrame, aligned: pd.DataFrame
) -> pd.DataFrame:
    """Attach the 15-minute CALL response used only for historical learning."""
    entries = classify_entry_context(setups)
    entries["session_phase"] = np.select(
        [
            entries.minutes_from_open.lt(60),
            entries.minutes_from_open.ge(240),
        ],
        ["OPENING_DRIVE", "LATE_TREND"], default="MID_SESSION",
    )
    entries["structure_context"] = np.where(
        entries.pullback_state.eq("PULLBACK"), "PULLBACK", "BREAKOUT"
    )
    entries["state_signature"] = (
        entries.market_regime + "|" + entries.session_phase + "|"
        + entries.structure_context
    )
    market = aligned[[
        "timestamp", "option_close", "option_high", "option_low",
        "index_session_id",
    ]].copy()
    market["timestamp"] = pd.to_datetime(
        market.timestamp, utc=True, errors="raise"
    ).dt.tz_convert("Asia/Kolkata")
    market = market.sort_values("timestamp").reset_index(drop=True)
    columns = [
        "timestamp", "entry_signal", "strategy_family", "market_regime",
        "trend_regime", "volatility_regime", "session_phase",
        "structure_context", "state_signature",
    ]
    mapped = entries[columns].merge(
        market[["timestamp", "option_close", "index_session_id"]],
        on="timestamp", how="inner", validate="one_to_one",
    ).rename(columns={"option_close": "entry_premium"})
    lookup = pd.Series(market.index, index=market.timestamp)
    sessions = market.index_session_id.astype(str).to_numpy()
    rows = []
    for entry in mapped.itertuples(index=False):
        index = int(lookup.loc[entry.timestamp])
        end = entry.timestamp + pd.Timedelta(minutes=RESEARCH_HORIZON_MINUTES)
        stop = int(market.timestamp.searchsorted(end, side="right"))
        path = market.iloc[index + 1:stop]
        path = path.loc[path.index_session_id.astype(str).eq(sessions[index])]
        if path.empty or entry.entry_premium <= 0:
            continue
        premium = float(entry.entry_premium)
        returns_high = path.option_high.astype(float) / premium - 1
        target_hits = np.flatnonzero(returns_high.ge(RESEARCH_EXPANSION_TARGET).to_numpy())
        target_hit = len(target_hits) > 0
        holding = (
            int((path.timestamp.iloc[target_hits[0]] - entry.timestamp).total_seconds() / 60)
            if target_hit
            else int((path.timestamp.iloc[-1] - entry.timestamp).total_seconds() / 60)
        )
        rows.append({
            **{column: getattr(entry, column) for column in columns},
            "entry_premium": premium,
            "outcome_available_timestamp": path.timestamp.iloc[-1],
            "premium_expansion_hit": target_hit,
            "terminal_premium_return": float(path.option_close.iloc[-1]) / premium - 1,
            "premium_mfe": max(0.0, float(path.option_high.max()) / premium - 1),
            "premium_mae": max(0.0, 1 - float(path.option_low.min()) / premium),
            "holding_minutes": holding,
        })
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


@dataclass
class _Accumulator:
    count: int = 0
    expansion_hits: int = 0
    terminal_sum: float = 0.0
    terminal_sum_sq: float = 0.0
    mfe_sum: float = 0.0
    mae_sum: float = 0.0
    holding_sum: float = 0.0

    def add(self, row) -> None:
        terminal = float(row.terminal_premium_return)
        self.count += 1
        self.expansion_hits += int(row.premium_expansion_hit)
        self.terminal_sum += terminal
        self.terminal_sum_sq += terminal * terminal
        self.mfe_sum += float(row.premium_mfe)
        self.mae_sum += float(row.premium_mae)
        self.holding_sum += float(row.holding_minutes)

    def estimates(self) -> dict:
        if self.count == 0:
            return {}
        mean = self.terminal_sum / self.count
        variance = max(self.terminal_sum_sq / self.count - mean * mean, 0.0)
        standard_error = np.sqrt(variance / self.count)
        expected_mfe = self.mfe_sum / self.count
        expected_mae = self.mae_sum / self.count
        expansion_probability = self.expansion_hits / self.count
        ratio = expected_mfe / max(expected_mae, 1e-12)
        sample_confidence = min(1.0, np.sqrt(self.count / 100))
        confidence = sample_confidence * expansion_probability * min(ratio, 2.0) / 2
        return {
            "history_count": self.count,
            "premium_expansion_probability": expansion_probability,
            "expected_premium_expansion": expected_mfe,
            "expected_drawdown": expected_mae,
            "expected_holding_minutes": self.holding_sum / self.count,
            "expected_terminal_return": mean,
            "terminal_return_standard_error": standard_error,
            "terminal_return_95_lower": mean - 1.96 * standard_error,
            "expansion_drawdown_ratio": ratio,
            "strategy_confidence_score": confidence,
        }


def build_walk_forward_decisions(observations: pd.DataFrame) -> pd.DataFrame:
    """Select a strategy using only matured historical observations."""
    data = observations.sort_values("timestamp").reset_index(drop=True)
    matured = data.sort_values(
        ["outcome_available_timestamp", "timestamp"]
    ).reset_index(drop=True)
    accumulators: dict[tuple, _Accumulator] = defaultdict(_Accumulator)
    pointer = 0
    rows = []
    observed_strategies = sorted(set(data.strategy_family).intersection(STRATEGIES))
    for current in data.itertuples(index=False):
        while (
            pointer < len(matured)
            and matured.at[pointer, "outcome_available_timestamp"] <= current.timestamp
        ):
            historical = matured.iloc[pointer]
            if historical.timestamp < current.timestamp:
                for key in (
                    ("EXACT", historical.state_signature, historical.strategy_family),
                    ("REGIME", historical.market_regime, historical.strategy_family),
                    ("STRATEGY", historical.strategy_family),
                ):
                    accumulators[key].add(historical)
            pointer += 1
        candidates = []
        for strategy in observed_strategies:
            source = "NONE"
            accumulator = None
            for key in (
                ("EXACT", current.state_signature, strategy),
                ("REGIME", current.market_regime, strategy),
                ("STRATEGY", strategy),
            ):
                candidate = accumulators[key]
                if candidate.count >= MINIMUM_HISTORY:
                    source, accumulator = key[0], candidate
                    break
            if accumulator is None:
                continue
            estimates = accumulator.estimates()
            candidates.append({
                "activated_strategy": strategy,
                "estimate_source": source,
                **estimates,
            })
        if candidates:
            selected = max(
                candidates,
                key=lambda item: (
                    item["strategy_confidence_score"],
                    item["terminal_return_95_lower"],
                    item["history_count"],
                    item["activated_strategy"],
                ),
            )
            statistically_positive = (
                selected["terminal_return_95_lower"] > 0
                and selected["expansion_drawdown_ratio"] > 1
            )
            recommendation = (
                "TRADE"
                if current.strategy_family == selected["activated_strategy"]
                and statistically_positive
                else "SKIP"
            )
            reason = (
                "ACTIVE_STRATEGY_MATCH"
                if recommendation == "TRADE"
                else "CURRENT_STRATEGY_NOT_SELECTED"
                if current.strategy_family != selected["activated_strategy"]
                else "INSUFFICIENT_POSITIVE_EVIDENCE"
            )
        else:
            selected = {
                "activated_strategy": "NONE", "estimate_source": "NONE",
                "history_count": 0, "premium_expansion_probability": np.nan,
                "expected_premium_expansion": np.nan, "expected_drawdown": np.nan,
                "expected_holding_minutes": np.nan, "expected_terminal_return": np.nan,
                "terminal_return_standard_error": np.nan,
                "terminal_return_95_lower": np.nan,
                "expansion_drawdown_ratio": np.nan,
                "strategy_confidence_score": 0.0,
            }
            recommendation, reason = "SKIP", "INSUFFICIENT_HISTORY"
        rows.append({
            "timestamp": current.timestamp,
            "market_state": current.state_signature,
            "trend_regime": current.trend_regime,
            "volatility_regime": current.volatility_regime,
            "session_phase": current.session_phase,
            "structure_context": current.structure_context,
            "current_strategy": current.strategy_family,
            **selected,
            "recommendation": recommendation,
            "decision_reason": reason,
            # Research response fields are retained only for later analysis;
            # they are never inputs to the decision above.
            "realized_expansion_hit": current.premium_expansion_hit,
            "realized_terminal_return": current.terminal_premium_return,
            "realized_mfe": current.premium_mfe,
            "realized_mae": current.premium_mae,
        })
    return pd.DataFrame(rows)

