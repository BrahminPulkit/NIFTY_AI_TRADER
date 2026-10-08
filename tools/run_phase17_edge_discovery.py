"""Research-only economic edge discovery on validated CE/PE observations."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paper_trading_engine import PaperTradingConfig, trading_costs  # noqa: E402
from tools.run_phase12_feature_signal_audit import folds  # noqa: E402
from tools.run_risk_adjusted_catboost_research import load_data  # noqa: E402


OUTPUT = ROOT / "reports/phase17_edge_discovery"
LABEL_DATA = ROOT / "reports/corrected_scalping_label_audit/label_comparison.parquet"
QUANTITY = 65
SLIPPAGE = {"LOW": 1.0, "MEDIUM": 2.0, "HIGH": 5.0}
TIME_BUCKETS = [("09:15-09:30", 555, 570), ("09:30-10:00", 570, 600),
                ("10:00-11:00", 600, 660), ("11:00-12:00", 660, 720),
                ("12:00-13:00", 720, 780), ("13:00-14:00", 780, 840),
                ("14:00-15:00", 840, 900), ("FINAL_HOUR", 870, 930)]


def load_economic_context(data: pd.DataFrame) -> pd.DataFrame:
    economic = pd.read_parquet(
        ROOT / "reports/phase16_time_decay_audit/economic_observations.parquet")
    economic = economic.rename(columns={"selected_premium": "entry_premium"})
    keep = ["timestamp", *[column for column in economic.columns
                            if column != "timestamp" and column not in data.columns]]
    result = data.merge(economic[keep], on="timestamp", how="left", validate="one_to_one")
    option_frames = []
    for root in (ROOT / "data/normalized/historical_options/validated20",
                 ROOT / "data/normalized/historical_options/non_expiry15"):
        for session in (path for path in root.iterdir() if path.is_dir()):
            for side in ("ce", "pe"):
                frame = pd.read_parquet(session / f"{side}.parquet", columns=["timestamp", "oi"])
                frame = frame.rename(columns={"oi": f"{side}_oi"})
                option_frames.append((side, frame))
    ce = pd.concat([frame for side, frame in option_frames if side == "ce"]).drop_duplicates("timestamp")
    pe = pd.concat([frame for side, frame in option_frames if side == "pe"]).drop_duplicates("timestamp")
    result = result.merge(ce, on="timestamp", how="left", validate="one_to_one")
    result = result.merge(pe, on="timestamp", how="left", validate="one_to_one")
    result["selected_oi"] = np.where(result.option_type.eq("CE"), result.ce_oi, result.pe_oi)
    result["opposite_oi"] = np.where(result.option_type.eq("CE"), result.pe_oi, result.ce_oi)
    result["moneyness_bucket"] = pd.cut(
        result.signed_moneyness_pct, [-np.inf, -0.1, 0.1, np.inf], labels=["OTM", "ATM", "ITM"])
    return result


def profit_factor(values: pd.Series) -> float:
    gains, losses = values[values > 0].sum(), -values[values < 0].sum()
    return float(gains / losses) if losses > 0 else float("inf")


def maximum_drawdown(values: pd.Series) -> float:
    curve = values.cumsum(); return float((curve.cummax() - curve).max()) if len(curve) else 0.0


def add_context(data: pd.DataFrame) -> pd.DataFrame:
    nifty_paths = {}
    for root in (ROOT / "data/normalized/historical_options/validated20",
                 ROOT / "data/normalized/historical_options/non_expiry15"):
        for session in (path for path in root.iterdir() if path.is_dir()):
            nifty = pd.read_parquet(session / "nifty.parquet")
            nifty["timestamp"] = pd.to_datetime(nifty.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
            date = nifty.timestamp.iloc[0].strftime("%Y-%m-%d")
            nifty_paths[date] = nifty.set_index("timestamp").sort_index()
    labels = pd.read_parquet(LABEL_DATA)[[
        "timestamp", "mfe_5m_pct", "mae_5m_pct", "return_1m_pct",
        "first_candle_both_touched", "high_excursion_1m_pct", "low_excursion_1m_pct"]]
    data = data.merge(labels, on="timestamp", validate="one_to_one")
    rows = []
    for row in data.itertuples(index=False):
        timestamp, date = pd.Timestamp(row.timestamp), str(row.session_date)
        nifty = nifty_paths[date]
        history = nifty.loc[:timestamp]
        five, fifteen = history.tail(5), history.tail(15)
        high15, low15 = float(fifteen.high.max()), float(fifteen.low.min())
        side = row.option_type
        selected_momentum = row.ce_momentum_5m if side == "CE" else row.pe_momentum_5m
        selected_acceleration = row.ce_premium_acceleration if side == "CE" else row.pe_premium_acceleration
        relative_strength = row.ce_strength_vs_pe if side == "CE" else row.pe_strength_vs_ce
        volume_imbalance = (row.ce_volume_ratio - row.pe_volume_ratio) * (1 if side == "CE" else -1)
        selected_volatility = row.ce_volatility if side == "CE" else row.pe_volatility
        directional_momentum = row.nifty_momentum_5m * (1 if side == "CE" else -1)
        selected_return = row.ce_return_5m if side == "CE" else row.pe_return_5m
        directional_return = row.nifty_return_5m * (1 if side == "CE" else -1)
        response = selected_return / abs(directional_return) if abs(directional_return) >= 0.0001 else np.nan
        selected_oi, opposite_oi = float(row.selected_oi), float(row.opposite_oi)
        oi_imbalance = (selected_oi - opposite_oi) / max(selected_oi + opposite_oi, 1)
        range5 = (float(five.high.max()) - float(five.low.min())) / row.spot * 100
        range15 = (high15 - low15) / row.spot * 100
        range_position = (row.spot - low15) / (high15 - low15) if high15 > low15 else 0.5
        minute = timestamp.hour * 60 + timestamp.minute
        rows.append({
            "directional_nifty_momentum": directional_momentum,
            "selected_option_momentum": selected_momentum,
            "selected_acceleration": selected_acceleration,
            "selected_relative_strength": relative_strength,
            "volume_imbalance": volume_imbalance, "oi_imbalance": oi_imbalance,
            "selected_volatility": selected_volatility, "option_response_coefficient": response,
            "nifty_range_5m_pct": range5, "nifty_range_15m_pct": range15,
            "range_expansion_ratio": range5 / range15 if range15 > 0 else np.nan,
            "range_position_15m": range_position,
            "distance_recent_high_pct": (high15 - row.spot) / row.spot * 100,
            "distance_recent_low_pct": (row.spot - low15) / row.spot * 100,
            "minute_of_day": minute,
            "time_bucket_primary": next((name for name, start, end in TIME_BUCKETS[:-1]
                                         if start <= minute < end), "OUTSIDE"),
        })
    result = pd.concat([data.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    result["gross_pnl"] = result.entry_premium * result.selected_return_5m_pct / 100 * QUANTITY
    result["exit_premium_5m"] = result.entry_premium * (1 + result.selected_return_5m_pct / 100)
    for scenario, bps in SLIPPAGE.items():
        config = PaperTradingConfig(quantity=QUANTITY, slippage_bps_each_side=bps)
        result[f"net_pnl_{scenario.lower()}"] = [
            gross - trading_costs(entry, exit_, QUANTITY, config)["total_costs"]
            for gross, entry, exit_ in zip(result.gross_pnl, result.entry_premium,
                                            result.exit_premium_5m)]
    return result


def thresholds(train: pd.DataFrame) -> dict:
    columns = ["nifty_atr_pct", "directional_nifty_momentum", "selected_option_momentum",
               "selected_relative_strength", "volume_imbalance", "oi_imbalance",
               "option_response_coefficient", "range_expansion_ratio"]
    return {column: {q: float(train[column].quantile(q)) for q in (.33, .5, .67)}
            for column in columns}


def condition_masks(frame: pd.DataFrame, limits: dict) -> dict[str, pd.Series]:
    q = lambda column, value: limits[column][value]
    trend_aligned = ((frame.option_type.eq("CE") & frame.trend_regime.eq("TREND_UP"))
                     | (frame.option_type.eq("PE") & frame.trend_regime.eq("TREND_DOWN")))
    directional = frame.directional_nifty_momentum
    relative = frame.selected_relative_strength
    compression = frame.range_expansion_ratio <= q("range_expansion_ratio", .33)
    breakout = ((frame.option_type.eq("CE") & frame.range_position_15m.ge(.8))
                | (frame.option_type.eq("PE") & frame.range_position_15m.le(.2)))
    return {
        "HIGH_NIFTY_VOLATILITY": frame.nifty_atr_pct >= q("nifty_atr_pct", .67),
        "LOW_NIFTY_VOLATILITY": frame.nifty_atr_pct <= q("nifty_atr_pct", .33),
        "STRONG_DIRECTIONAL_MOMENTUM": directional >= q("directional_nifty_momentum", .67),
        "STRONG_OPTION_MOMENTUM": frame.selected_option_momentum >= q("selected_option_momentum", .67),
        "STRONG_RELATIVE_STRENGTH": relative >= q("selected_relative_strength", .67),
        "HIGH_VOLUME_IMBALANCE": frame.volume_imbalance >= q("volume_imbalance", .67),
        "HIGH_OI_IMBALANCE": frame.oi_imbalance >= q("oi_imbalance", .67),
        "HIGH_OPTION_RESPONSE": frame.option_response_coefficient >= q("option_response_coefficient", .67),
        "VOLATILITY_COMPRESSION": compression,
        "RANGE_EXPANSION": frame.range_expansion_ratio >= q("range_expansion_ratio", .67),
        "TREND_PLUS_MOMENTUM": trend_aligned & directional.ge(q("directional_nifty_momentum", .5)),
        "MOMENTUM_PLUS_RELATIVE": directional.ge(q("directional_nifty_momentum", .5))
                                    & relative.ge(q("selected_relative_strength", .5)),
        "COMPRESSION_PLUS_BREAKOUT": compression & breakout,
        "EXPIRY_STRONG_MOMENTUM": frame.expiry_day_status.eq("EXPIRY_DAY")
                                  & directional.ge(q("directional_nifty_momentum", .67)),
        "NONEXPIRY_TREND_MOMENTUM": frame.expiry_day_status.eq("NON_EXPIRY_DAY")
                                    & trend_aligned & directional.ge(q("directional_nifty_momentum", .5)),
        "OPENING_HIGH_VOLATILITY": frame.minute_of_day.lt(600)
                                   & frame.nifty_atr_pct.ge(q("nifty_atr_pct", .67)),
        "FINAL_HOUR_HIGH_VOLATILITY": frame.minute_of_day.ge(870)
                                      & frame.nifty_atr_pct.ge(q("nifty_atr_pct", .67)),
        "OPTION_VWAP_CONFIRMATION": np.where(
            frame.option_type.eq("CE"), frame.ce_vwap_distance, frame.pe_vwap_distance) > 0,
    }


def economic_metrics(group: pd.DataFrame, pnl_column: str = "net_pnl_medium") -> dict:
    if group.empty:
        return {"observations": 0}
    pnl = group[pnl_column]
    sessions = group.groupby("session_date")[pnl_column].sum()
    return {
        "observations": len(group), "sessions": int(group.session_date.nunique()),
        "win_rate": float(group.label.mean()),
        "mean_forward_return": float(group.selected_return_5m_pct.mean()),
        "median_forward_return": float(group.selected_return_5m_pct.median()),
        "mean_mfe": float(group.mfe_5m_pct.mean()), "mean_mae": float(group.mae_5m_pct.mean()),
        "gross_expectancy": float(group.gross_pnl.mean()),
        "after_cost_expectancy": float(pnl.mean()), "profit_factor": profit_factor(pnl),
        "maximum_drawdown": maximum_drawdown(pnl),
        "positive_session_pct": float(sessions.gt(0).mean() * 100),
    }


def bh_adjust(pvalues: pd.Series) -> pd.Series:
    order = np.argsort(pvalues.to_numpy()); ranked = pvalues.to_numpy()[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    result = np.empty(len(ranked)); result[order] = np.clip(adjusted, 0, 1)
    return pd.Series(result, index=pvalues.index)


def discover(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows, selected_rows, membership_rows = [], [], []
    for fold_id, (train_dates, test_dates) in enumerate(folds(data), start=1):
        train, test = data[data.session_date.isin(train_dates)], data[data.session_date.isin(test_dates)]
        masks = condition_masks(test, thresholds(train))
        for name, mask in masks.items():
            group = test.loc[np.asarray(mask, dtype=bool)]
            row = {"candidate": name, "fold": fold_id, **economic_metrics(group)}
            fold_rows.append(row)
            membership_rows.append(pd.DataFrame({"candidate": name, "fold": fold_id,
                                                  "label": test.label.to_numpy(),
                                                  "selected": np.asarray(mask, dtype=bool)}))
            if len(group):
                selected_rows.append(group.assign(candidate=name, fold=fold_id))
    folds_frame = pd.DataFrame(fold_rows)
    selected = pd.concat(selected_rows, ignore_index=True)
    membership = pd.concat(membership_rows, ignore_index=True)
    rankings = []
    for name, group in selected.groupby("candidate"):
        metrics = economic_metrics(group)
        candidate_folds = folds_frame[folds_frame.candidate.eq(name)]
        discovery = candidate_folds[candidate_folds.fold.le(3)].after_cost_expectancy.dropna()
        pvalue = float(ttest_1samp(discovery, 0, alternative="greater").pvalue) if len(discovery) >= 2 else 1.0
        side_exp = group.groupby("option_type").net_pnl_medium.mean()
        expiry_exp = group.groupby("expiry_day_status").net_pnl_medium.mean()
        regime_exp = group.groupby("trend_regime").net_pnl_medium.mean()
        high_cost = float(group.net_pnl_high.mean())
        membership_group = membership[membership.candidate.eq(name)]
        condition_auc = float(roc_auc_score(membership_group.label,
                                            membership_group.selected.astype(int)))
        complement = membership_group.loc[~membership_group.selected, "label"]
        rankings.append({"candidate": name, **metrics, "discovery_pvalue": pvalue,
                         "positive_folds": int(candidate_folds.after_cost_expectancy.gt(0).sum()),
                         "median_fold_expectancy": float(candidate_folds.after_cost_expectancy.median()),
                         "minimum_fold_expectancy": float(candidate_folds.after_cost_expectancy.min()),
                         "confirmation_fold4_expectancy": float(candidate_folds.loc[
                             candidate_folds.fold.eq(4), "after_cost_expectancy"].iloc[0]),
                         "confirmation_fold5_expectancy": float(candidate_folds.loc[
                             candidate_folds.fold.eq(5), "after_cost_expectancy"].iloc[0]),
                         "ce_expectancy": float(side_exp.get("CE", np.nan)),
                         "pe_expectancy": float(side_exp.get("PE", np.nan)),
                         "expiry_expectancy": float(expiry_exp.get("EXPIRY_DAY", np.nan)),
                         "non_expiry_expectancy": float(expiry_exp.get("NON_EXPIRY_DAY", np.nan)),
                         "minimum_regime_expectancy": float(regime_exp.min()),
                         "high_cost_expectancy": high_cost,
                         "condition_membership_auc": condition_auc,
                         "conditional_win_probability": float(group.label.mean()),
                         "complement_win_probability": float(complement.mean())})
    ranking = pd.DataFrame(rankings)
    ranking["discovery_fdr"] = bh_adjust(ranking.discovery_pvalue)
    ranking["passes"] = (ranking.observations.ge(100) & ranking.after_cost_expectancy.gt(0)
                         & ranking.profit_factor.gt(1) & ranking.median_fold_expectancy.gt(0)
                         & ranking.minimum_fold_expectancy.gt(0) & ranking.positive_folds.ge(3)
                         & ranking.confirmation_fold4_expectancy.gt(0)
                         & ranking.confirmation_fold5_expectancy.gt(0)
                         & ranking.ce_expectancy.gt(0) & ranking.pe_expectancy.gt(0)
                         & ranking.expiry_expectancy.gt(0) & ranking.non_expiry_expectancy.gt(0)
                         & ranking.high_cost_expectancy.gt(0) & ranking.discovery_fdr.le(.10))
    return ranking.sort_values("after_cost_expectancy", ascending=False), folds_frame, selected


def grouped_economics(data: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for value, group in data.groupby(column, observed=True):
        rows.append({column: value, **economic_metrics(group),
                     "low_cost_expectancy": float(group.net_pnl_low.mean()),
                     "high_cost_expectancy": float(group.net_pnl_high.mean())})
    return pd.DataFrame(rows)


def resolution_audit(data: pd.DataFrame) -> dict:
    raw = pd.read_parquet(LABEL_DATA)
    minute_ambiguous = int(raw.first_candle_both_touched.sum())
    returns = raw.return_1m_pct.dropna()
    return {
        "observations": len(raw), "one_minute_ohlc": True,
        "intrabar_order_unrecoverable": minute_ambiguous,
        "intrabar_information_loss_pct": float(minute_ambiguous / len(raw) * 100),
        "current_barrier_ambiguous": int(raw.CURRENT_FIXED_5_3.eq("AMBIGUOUS").sum()),
        "bid_ask_available": False, "spread_available": False, "iv_available": False,
        "greeks_available": False, "option_volume_available_pct": 100.0,
        "option_oi_available_pct": 100.0,
        "expiry_sessions": int(data.loc[data.expiry_day_status.eq("EXPIRY_DAY"), "session_date"].nunique()),
        "non_expiry_sessions": int(data.loc[data.expiry_day_status.eq("NON_EXPIRY_DAY"), "session_date"].nunique()),
        "moneyness": data.moneyness_bucket.value_counts().to_dict(),
        "unique_contract_segments": int(data[["session_date", "expiry", "strike", "option_type"]].drop_duplicates().shape[0]),
        "one_minute_return_percentiles": {str(q): float(returns.quantile(q)) for q in (.01, .05, .5, .95, .99)},
        "absolute_one_minute_jumps_ge_5pct": int(returns.abs().ge(5).sum()),
        "limitation": "OHLC cannot order intrabar events or reconstruct executable bid/ask fills.",
    }


def main() -> None:
    base, leakage = load_data(); data = add_context(load_economic_context(base))
    ranking, fold_frame, selected = discover(data)
    time_rows = []
    for name, start, end in TIME_BUCKETS:
        group = data[data.minute_of_day.between(start, end - 1)]
        time_rows.append({"time_bucket": name, **economic_metrics(group),
                          "opportunity_frequency_pct": float(len(group) / len(data) * 100),
                          "low_cost_expectancy": float(group.net_pnl_low.mean()),
                          "high_cost_expectancy": float(group.net_pnl_high.mean())})
    time_frame = pd.DataFrame(time_rows)
    volatility = pd.qcut(data.nifty_atr_pct, 3, labels=["LOW", "MEDIUM", "HIGH"])
    volatility_frame = grouped_economics(data.assign(volatility_regime=volatility), "volatility_regime")
    cost_rows = []
    for candidate, group in selected.groupby("candidate"):
        for scenario in SLIPPAGE:
            pnl = group[f"net_pnl_{scenario.lower()}"]
            cost_rows.append({"candidate": candidate, "scenario": scenario, "observations": len(group),
                              "expectancy": float(pnl.mean()), "profit_factor": profit_factor(pnl)})
    passing = ranking[ranking.passes]
    strongest = ranking.iloc[0].replace({np.nan: None}).to_dict()
    eligible = ranking.loc[ranking.observations.ge(100)]
    result = {
        "status": "PHASE17_EDGE_DISCOVERY_COMPLETE",
        "verdict": "NO_ROBUST_EDGE_FOUND" if passing.empty else "ROBUST_EDGE_CANDIDATE_FOUND",
        "data_resolution": resolution_audit(data),
        "strongest_observed_condition": strongest,
        "strongest_sample_eligible_condition": eligible.iloc[0].replace({np.nan: None}).to_dict(),
        "strongest_edge_classification": "DESCRIPTIVE_ONLY" if passing.empty else "PREDICTIVE_CANDIDATE",
        "passing_candidates": passing.candidate.tolist(),
        "survival": {"walk_forward": bool(len(passing)), "ce_pe": bool(len(passing)),
                     "expiry_non_expiry": bool(len(passing)), "regimes": bool(len(passing)),
                     "time_of_day": False, "cost_sensitivity": bool(len(passing))},
        "one_minute_dataset_sufficient": False,
        "single_next_direction": "Acquire multi-strike NIFTY option quotes at 5-15 second resolution across 0-3 DTE, then test quote-executable underlying-to-option response asymmetry.",
        "required_data": ["timestamp with seconds", "NIFTY spot", "expiry", "strike", "option_type",
                          "contract identifier", "LTP", "bid", "ask", "volume", "OI", "IV",
                          "delta", "gamma", "theta", "vega"],
        "minimum_scope": "ATM +/- 5 strikes, CE and PE, 0/1/2/3+ DTE, 60-100 complete sessions across regimes",
        "cost_model": {"quantity": QUANTITY, "slippage_bps_each_side": SLIPPAGE,
                       "other_charges": "unchanged PaperTradingConfig brokerage/STT/exchange/SEBI/stamp/GST",
                       "spread_modelled_separately": False},
        "leakage_audit": leakage,
        "production_changed": False, "strategy_selected": False, "catboost_tuned": False,
        "threshold_tuned": False, "paper_trading_changed": False, "broker_changed": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    pd.DataFrame([result["data_resolution"]]).to_json(OUTPUT / "data_resolution.json", orient="records", indent=2)
    ranking.to_csv(OUTPUT / "condition_ranking.csv", index=False)
    fold_frame.to_csv(OUTPUT / "fold_results.csv", index=False)
    time_frame.to_csv(OUTPUT / "time_of_day_analysis.csv", index=False)
    volatility_frame.to_csv(OUTPUT / "volatility_regime_analysis.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(OUTPUT / "cost_sensitivity.csv", index=False)
    data.to_parquet(OUTPUT / "research_observations.parquet", index=False)
    requirement = "# Required next-level dataset\n\n" + "\n".join(
        f"- {field}" for field in result["required_data"])
    (OUTPUT / "data_requirement.md").write_text(requirement, encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
