"""Reproducible Phase-1 ablation engine.

This engine varies only the enabled filter set. Dataset, session, time frame,
risk, exits, holding period and costs are loaded once from common.yaml.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

LAB = Path(__file__).resolve().parents[1]
if str(LAB.parent) not in sys.path:
    sys.path.insert(0, str(LAB.parent))

from institutional_hybrid.backtests import BacktestConfig, ResearchBacktester
from institutional_hybrid.strategies import InstitutionalHybridStrategy
from institutional_hybrid.utils.io import assert_lab_output
from institutional_hybrid.utils.production_v1_adapter import ProductionV1ReadOnlyAdapter
from institutional_hybrid.utils.research_gate import require_approved_data

ROOT = Path(__file__).resolve().parent
METRICS = [
    "trades", "win_rate", "profit_factor", "expectancy", "average_reward",
    "average_risk", "average_premium_expansion", "average_atr_expansion",
    "average_hold_time", "maximum_drawdown", "trade_frequency",
    "average_trade_quality", "rejected_setup_pct", "average_institutional_score",
]
EXPERIMENTS = {
    "experiment_01_ema_pullback": ["ema_pullback"],
    "experiment_02_ema_liquidity": ["ema_pullback", "liquidity"],
    "experiment_03_ema_momentum": ["ema_pullback", "momentum"],
    "experiment_04_ema_momentum_breakout": ["ema_pullback", "momentum", "breakout"],
    "experiment_05_ema_momentum_liquidity_volume": ["ema_pullback", "momentum", "liquidity", "volume"],
    "experiment_06_full_premium_filter": ["ema_pullback", "momentum", "liquidity", "volume", "premium_expansion"],
}


def load_common() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "common.yaml").read_text(encoding="utf-8"))


def rules_hash(common: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(common, sort_keys=True).encode()).hexdigest()


def prepare() -> None:
    common, digest = load_common(), rules_hash(load_common())
    for number, (name, filters) in enumerate(EXPERIMENTS.items(), 1):
        folder = assert_lab_output(ROOT / name)
        (folder / "charts").mkdir(parents=True, exist_ok=True)
        cfg = {
            "experiment_id": name, "phase": 1, "filters": filters,
            "common_rules_file": "../common.yaml", "common_rules_sha256": digest,
        }
        (folder / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        (folder / "comparison.json").write_text(json.dumps({
            "status": "PENDING_DATA", "experiment_id": name,
            "baseline": None if number == 1 else list(EXPERIMENTS)[number - 2],
            "common_rules_sha256": digest, "metrics": None,
        }, indent=2), encoding="utf-8")
        pd.DataFrame(columns=["status", *METRICS]).to_csv(folder / "metrics.csv", index=False)
        (folder / "report.md").write_text(
            f"# {name}\n\n**Status: PENDING_DATA**\n\nFilters: `{', '.join(filters)}`.\n\n"
            "No market dataset is currently present, so no performance result has been inferred. "
            "Run the Phase-1 engine after adding the fixed futures and ATM-options CSVs.\n",
            encoding="utf-8",
        )
        write_notebook(folder, name)
    write_pending_comparison(digest)


def write_notebook(folder: Path, name: str) -> None:
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                f"# {name}\n", "\n", "Reproducible Phase-1 artifact. Configuration is in `config.yaml`."
            ]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "from pathlib import Path\n",
                "import sys, yaml, pandas as pd\n",
                "LAB = Path.cwd().resolve().parents[1]\n",
                "sys.path.insert(0, str(LAB.parent))\n",
                "from institutional_hybrid.experiments.phase1_engine import run_one\n",
                f"result = run_one('{name}')\n",
                "result\n",
            ]},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    (folder / f"{name}.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def load_data(common: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use only the pinned read-only V1 adapter; never silently fall back."""
    source = ProductionV1ReadOnlyAdapter().load_for_phase1()
    source = source.set_index(pd.to_datetime(source["timestamp"])).sort_index()
    futures = source.rename(columns={
        "index_open": "open", "index_high": "high", "index_low": "low",
        "index_close": "close", "futures_volume": "volume",
    })[["open", "high", "low", "close", "volume"]]
    options = source.rename(columns={
        "option_bid": "bid", "option_ask": "ask", "option_open_interest": "open_interest",
        "option_delta": "delta", "option_gamma": "gamma", "option_theta": "theta",
        "option_iv": "iv", "option_close": "premium",
    })[["bid", "ask", "option_volume", "open_interest", "delta", "gamma", "theta", "iv", "premium"]]
    start, end = common["session"]["trading_start"], common["session"]["trading_end"]
    return futures.between_time(start, end), options.between_time(start, end)


def build_signals(features: pd.DataFrame, filters: list[str], common: dict[str, Any]) -> pd.DataFrame:
    out = features.copy()
    conditions = {
        "ema_pullback": out["ema_pullback"] & (out["trend_direction"] != 0),
        "liquidity": out["false_break"] | (out["market_structure_shift"] != 0),
        "momentum": (out["atr_expansion"] >= common["signal"]["momentum_atr_threshold"])
                    & (out["range_expansion"] >= common["signal"]["momentum_range_threshold"]),
        "breakout": (out["breakout_direction"] != 0) | out["compression_breakout"],
        "volume": out["relative_volume"] >= common["signal"]["relative_volume_threshold"],
        "premium_expansion": out["premium_expansion_score"] >= common["signal"]["premium_expansion_threshold"],
    }
    base = conditions["ema_pullback"].fillna(False)
    accepted = pd.Series(True, index=out.index)
    for name in filters:
        accepted &= conditions[name].fillna(False)
    out["signal"] = np.where(accepted, out["trend_direction"], 0).astype(int)
    failed = pd.DataFrame({k: ~conditions[k].fillna(False) for k in filters}, index=out.index)
    out["rejection_reason"] = ""
    rejected = base & ~accepted
    out.loc[rejected, "rejection_reason"] = failed.loc[rejected].idxmax(axis=1)
    passed = pd.DataFrame({k: conditions[k].astype(float) for k in filters})
    out["entry_quality"] = passed.mean(axis=1).mul(100)
    return out


def run_one(name: str, data: tuple[pd.DataFrame, pd.DataFrame] | None = None) -> dict[str, float]:
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}")
    require_approved_data()
    common, digest = load_common(), rules_hash(load_common())
    folder = ROOT / name
    cfg = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    if cfg["common_rules_sha256"] != digest:
        raise RuntimeError("Common rules changed after experiment preparation; rerun --prepare explicitly.")
    futures, options = data or load_data(common)
    strategy = InstitutionalHybridStrategy()
    features = strategy.score(strategy.build_features(futures, options))
    signals = build_signals(features, cfg["filters"], common)
    risk = common["risk"]
    bt = ResearchBacktester(BacktestConfig(
        max_hold_bars=risk["max_hold_bars"], target_pct=risk["target_pct"],
        stop_pct=risk["stop_pct"], round_trip_cost_pct=common["costs"]["round_trip_cost_pct"],
        cooldown_bars=risk["cooldown_bars"],
    ))
    trades, metrics = bt.run(signals)
    base_count = int((signals["ema_pullback"] & (signals["trend_direction"] != 0)).sum())
    rejected_count = int(signals["rejection_reason"].ne("").sum())
    metrics["rejected_setup_pct"] = rejected_count / base_count if base_count else 0.0
    metrics["average_institutional_score"] = (
        float(trades["institutional_score"].mean()) if not trades.empty else 0.0
    )
    pd.DataFrame([{"status": "COMPLETE", **metrics}]).to_csv(folder / "metrics.csv", index=False)
    trades.to_csv(folder / "trades.csv", index=False)
    make_charts(signals, trades, folder / "charts")
    baseline = None if name == list(EXPERIMENTS)[0] else list(EXPERIMENTS)[list(EXPERIMENTS).index(name) - 1]
    comparison = {"status": "COMPLETE", "experiment_id": name, "baseline": baseline,
                  "common_rules_sha256": digest, "metrics": metrics}
    (folder / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    write_report(folder, name, cfg["filters"], metrics)
    return metrics


def make_charts(signals: pd.DataFrame, trades: pd.DataFrame, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    returns = trades.get("net_return", pd.Series(dtype=float))
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1 if not equity.empty else equity
    plots = {
        "equity_curve": (equity, "Equity"),
        "drawdown_curve": (drawdown, "Drawdown"),
        "premium_expansion_distribution": (trades.get("premium_expansion", pd.Series(dtype=float)), "Premium expansion"),
        "atr_expansion_distribution": (trades.get("atr_expansion", pd.Series(dtype=float)), "ATR expansion"),
        "entry_quality_distribution": (trades.get("trade_quality", pd.Series(dtype=float)), "Entry quality"),
    }
    for filename, (series, label) in plots.items():
        fig, ax = plt.subplots(figsize=(9, 4))
        if "distribution" in filename:
            ax.hist(series.dropna(), bins=30)
        else:
            ax.plot(series.reset_index(drop=True))
        ax.set_title(label)
        fig.tight_layout(); fig.savefig(folder / f"{filename}.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 4))
    if not trades.empty:
        ax.scatter(pd.to_datetime(trades["entry_time"]), trades["net_return"],
                   c=np.where(trades["net_return"] > 0, "green", "red"))
    ax.set_title("Trade timeline"); fig.tight_layout(); fig.savefig(folder / "trade_timeline.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 3))
    heat = signals["signal"].resample("30min").apply(lambda s: (s != 0).sum()).to_numpy()[None, :]
    ax.imshow(heat, aspect="auto", cmap="YlOrRd"); ax.set_title("Signal heatmap")
    fig.tight_layout(); fig.savefig(folder / "signal_heatmap.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    signals.loc[signals["rejection_reason"].ne(""), "rejection_reason"].value_counts().plot.bar(ax=ax)
    ax.set_title("Rejected setup analysis"); fig.tight_layout()
    fig.savefig(folder / "rejected_setup_analysis.png", dpi=130); plt.close(fig)


def write_report(folder: Path, name: str, filters: list[str], metrics: dict[str, float]) -> None:
    rows = "\n".join(f"- {k.replace('_', ' ').title()}: {v:.6g}" for k, v in metrics.items())
    (folder / "report.md").write_text(
        f"# {name}\n\n**Status: COMPLETE**\n\nFilters: `{', '.join(filters)}`\n\n## Metrics\n\n{rows}\n",
        encoding="utf-8",
    )


def write_pending_comparison(digest: str) -> None:
    pd.DataFrame(columns=["rank", "experiment", "status", *METRICS]).to_csv(ROOT / "leaderboard.csv", index=False)
    (ROOT / "ranking.md").write_text(
        "# Phase-1 Ranking\n\n**Status: PENDING_DATA**\n\nNo ranking is possible until the fixed dataset is supplied.\n",
        encoding="utf-8",
    )
    (ROOT / "comparison.html").write_text(
        "<!doctype html><html><body><h1>Phase-1 Comparison</h1><p>Status: PENDING_DATA</p></body></html>",
        encoding="utf-8",
    )
    (ROOT / "Phase1_Best_Strategy.md").write_text(
        "# Phase 1 Best Strategy\n\n**Status: PENDING_DATA**\n\n"
        "1. Best experiment: Not determinable without data.\n"
        "2. Why it outperformed: Not determinable without measured trades.\n"
        "3. Filters that added value: Pending ablation results.\n"
        "4. Filters that reduced performance: Pending ablation results.\n"
        "5. Proceed to Phase 2: **No decision until Phase-1 is executed on the fixed dataset.**\n\n"
        f"Common-rules SHA-256: `{digest}`\n",
        encoding="utf-8",
    )


def compare() -> pd.DataFrame:
    records = []
    for name in EXPERIMENTS:
        path = ROOT / name / "metrics.csv"
        frame = pd.read_csv(path)
        if frame.empty or frame.iloc[0].get("status") != "COMPLETE":
            raise RuntimeError(f"{name} is not complete; comparison cannot be generated.")
        records.append({"experiment": name, **frame.iloc[0].to_dict()})
    board = pd.DataFrame(records)
    board["rank"] = board["expectancy"].rank(ascending=False, method="min").astype(int)
    board = board.sort_values(["rank", "profit_factor"], ascending=[True, False])
    board.to_csv(ROOT / "leaderboard.csv", index=False)
    highlights = {
        "Best Win Rate": board.loc[board["win_rate"].idxmax(), "experiment"],
        "Best Profit Factor": board.loc[board["profit_factor"].idxmax(), "experiment"],
        "Lowest Drawdown": board.loc[board["maximum_drawdown"].idxmax(), "experiment"],
        "Highest Expectancy": board.loc[board["expectancy"].idxmax(), "experiment"],
        "Highest Premium Expansion": board.loc[board["average_premium_expansion"].idxmax(), "experiment"],
        "Highest Institutional Score": board.loc[board["average_institutional_score"].idxmax(), "experiment"],
    }
    ranking = "# Phase-1 Ranking\n\n" + board.to_markdown(index=False) + "\n\n## Highlights\n\n"
    ranking += "\n".join(f"- **{k}:** {v}" for k, v in highlights.items())
    (ROOT / "ranking.md").write_text(ranking, encoding="utf-8")
    (ROOT / "comparison.html").write_text(board.to_html(index=False), encoding="utf-8")
    winner = board.iloc[0]["experiment"]
    prior = board.set_index("experiment")
    filter_effects = []
    names = list(EXPERIMENTS)
    for i in range(1, len(names)):
        delta = prior.loc[names[i], "expectancy"] - prior.loc[names[i-1], "expectancy"]
        filter_effects.append(f"- {names[i-1]} → {names[i]}: expectancy change {delta:.6g}")
    proceed = bool(
        prior.loc[winner, "expectancy"] > 0 and prior.loc[winner, "profit_factor"] > 1
        and prior.loc[winner, "trades"] >= 30
    )
    (ROOT / "Phase1_Best_Strategy.md").write_text(
        f"# Phase 1 Best Strategy\n\n1. **Best experiment:** {winner} (ranked by expectancy).\n\n"
        f"2. **Why:** expectancy {prior.loc[winner, 'expectancy']:.6g}, profit factor "
        f"{prior.loc[winner, 'profit_factor']:.6g}, drawdown {prior.loc[winner, 'maximum_drawdown']:.6g}.\n\n"
        "3–4. **Incremental filter effects:**\n\n" + "\n".join(filter_effects) +
        f"\n\n5. **Proceed to Phase 2:** {'Yes' if proceed else 'No'} under the declared gate "
        "(positive expectancy, profit factor > 1, at least 30 trades). Research only; no production integration.\n",
        encoding="utf-8",
    )
    return board


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.run_all:
        for name in EXPERIMENTS:
            run_one(name)
        compare()
    if not args.prepare and not args.run_all:
        parser.error("choose --prepare or --run-all")


if __name__ == "__main__":
    main()
