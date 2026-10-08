"""Step 19D: ATM CALL premium-expansion research over frozen LONG setups."""

from __future__ import annotations

import numpy as np
import pandas as pd


HORIZONS = (15, 30, 45, 60)
TARGETS = (0.05, 0.08, 0.10, 0.12, 0.15, 0.20)


def _timestamps(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise").dt.tz_convert("Asia/Kolkata")


def build_premium_paths(
    aligned: pd.DataFrame,
    outcomes: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """Measure causal post-entry CALL paths; production outcomes are not used."""
    required_aligned = {
        "timestamp", "option_open", "option_high", "option_low", "option_close",
        "option_volume", "index_session_id",
    }
    required_outcomes = {
        "timestamp", "direction", "trade_allowed", "trend_state",
        "session_segment", "year",
    }
    if missing := sorted(required_aligned.difference(aligned.columns)):
        raise ValueError(f"Aligned dataset missing: {missing}")
    if missing := sorted(required_outcomes.difference(outcomes.columns)):
        raise ValueError(f"Frozen outcomes missing: {missing}")
    market = aligned.copy()
    market["timestamp"] = _timestamps(market.timestamp)
    if market.timestamp.duplicated().any() or not market.timestamp.is_monotonic_increasing:
        raise ValueError("Aligned timestamps must be sorted and unique")
    allowed = outcomes.loc[
        outcomes.trade_allowed.eq(1) & outcomes.direction.eq("LONG"),
        ["timestamp", "trend_state", "session_segment", "year"],
    ].copy()
    allowed["timestamp"] = _timestamps(allowed.timestamp)
    entries = allowed.merge(
        market[["timestamp", "option_close", "index_session_id"]],
        on="timestamp", how="inner", validate="one_to_one",
    ).rename(columns={"option_close": "entry_premium"})
    sessions = market.index_session_id.astype(str).to_numpy()
    entry_lookup = pd.Series(market.index, index=market.timestamp)
    records: list[dict] = []
    for entry in entries.itertuples(index=False):
        index = int(entry_lookup.loc[entry.timestamp])
        premium = float(entry.entry_premium)
        if not np.isfinite(premium) or premium <= 0:
            raise ValueError(f"Invalid entry premium at {entry.timestamp}")
        for horizon in horizons:
            horizon_end = entry.timestamp + pd.Timedelta(minutes=horizon)
            stop_index = int(market.timestamp.searchsorted(horizon_end, side="right"))
            candidate = market.iloc[index + 1:stop_index]
            candidate = candidate.loc[
                candidate.index_session_id.astype(str).eq(sessions[index])
            ]
            if candidate.empty:
                records.append({
                    "timestamp": entry.timestamp, "horizon_minutes": horizon,
                    "entry_premium": premium, "observable_minutes": 0,
                    "full_horizon_observed": False, "terminal_return": np.nan,
                    "mfe": np.nan, "mae": np.nan, "year": int(entry.year),
                    "regime": entry.trend_state, "session": entry.session_segment,
                })
                continue
            last = candidate.iloc[-1]
            elapsed = int((last.timestamp - entry.timestamp).total_seconds() / 60)
            mfe = max(0.0, float(candidate.option_high.max()) / premium - 1)
            mae = max(0.0, 1 - float(candidate.option_low.min()) / premium)
            records.append({
                "timestamp": entry.timestamp,
                "horizon_minutes": horizon,
                "entry_premium": premium,
                "observable_minutes": elapsed,
                "full_horizon_observed": elapsed >= horizon,
                "terminal_return": float(last.option_close) / premium - 1,
                "mfe": mfe,
                "mae": mae,
                "average_future_volume": float(candidate.option_volume.mean()),
                "year": int(entry.year),
                "regime": entry.trend_state,
                "session": entry.session_segment,
            })
    result = pd.DataFrame(records)
    if len(result) != len(entries) * len(horizons):
        raise AssertionError("Every aligned LONG setup must have one row per horizon")
    return result


def expand_targets(
    paths: pd.DataFrame, targets: tuple[float, ...] = TARGETS
) -> pd.DataFrame:
    rows = []
    for target in targets:
        piece = paths.copy()
        piece["target"] = target
        piece["outcome"] = np.select(
            [piece.mfe.ge(target), piece.terminal_return.lt(0)],
            ["WIN", "LOSS"], default="TIMEOUT",
        )
        piece.loc[piece.terminal_return.isna(), "outcome"] = "UNOBSERVABLE"
        rows.append(piece)
    return pd.concat(rows, ignore_index=True)


def summarize(
    expanded: pd.DataFrame, groups: list[str]
) -> pd.DataFrame:
    rows = []
    observable = expanded.loc[expanded.outcome.ne("UNOBSERVABLE")]
    for keys, group in observable.groupby(groups, sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append({
            **dict(zip(groups, keys)),
            "setups": len(group),
            "win_count": int(group.outcome.eq("WIN").sum()),
            "win_rate": float(group.outcome.eq("WIN").mean()),
            "loss_rate": float(group.outcome.eq("LOSS").mean()),
            "timeout_rate": float(group.outcome.eq("TIMEOUT").mean()),
            "expected_premium_return": float(group.terminal_return.mean()),
            "median_premium_return": float(group.terminal_return.median()),
            "average_mfe": float(group.mfe.mean()),
            "average_mae": float(group.mae.mean()),
            "full_horizon_fraction": float(group.full_horizon_observed.mean()),
        })
    return pd.DataFrame(rows)


def build_leaderboard(expanded: pd.DataFrame) -> pd.DataFrame:
    overall = summarize(expanded, ["horizon_minutes", "target"])
    yearly = summarize(expanded, ["horizon_minutes", "target", "year"])
    stability = yearly.groupby(["horizon_minutes", "target"]).agg(
        yearly_win_rate_mean=("win_rate", "mean"),
        yearly_win_rate_std=("win_rate", "std"),
        minimum_year_win_count=("win_count", "min"),
        years_with_wins=("win_count", lambda values: int((values > 0).sum())),
        years_observed=("year", "nunique"),
    ).reset_index()
    result = overall.merge(stability, on=["horizon_minutes", "target"])
    result["yearly_win_rate_cv"] = (
        result.yearly_win_rate_std / result.yearly_win_rate_mean.replace(0, np.nan)
    )
    result["density_stability_score"] = (
        result.win_rate / (1 + result.yearly_win_rate_cv.fillna(np.inf))
    )
    # Fixed adequacy screen, declared for sample-size interpretation—not a
    # production target-selection rule.
    result["ml_sample_adequate"] = (
        result.win_count.ge(200)
        & result.win_rate.ge(0.05)
        & result.minimum_year_win_count.ge(10)
    )
    return result.sort_values(
        ["ml_sample_adequate", "density_stability_score", "win_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
