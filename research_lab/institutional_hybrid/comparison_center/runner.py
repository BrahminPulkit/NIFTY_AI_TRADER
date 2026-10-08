"""Execute and rank all isolated methodologies without optimisation."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import pandas as pd

LAB = Path(__file__).resolve().parents[1]
if str(LAB.parent) not in sys.path: sys.path.insert(0, str(LAB.parent))

from institutional_hybrid.experiments.phase1_engine import load_data
from institutional_hybrid.strategies import InstitutionalHybridStrategy
from institutional_hybrid.traders import TRADERS
from institutional_hybrid.traders.core import frame_hash, guard_output, write_json
from institutional_hybrid.utils.research_gate import require_approved_data

OUTPUT = Path(__file__).resolve().parent / "outputs"
RANK_COLUMNS = [
    "strategy", "total_trades", "win_rate", "profit_factor", "expectancy",
    "sharpe", "sortino", "calmar", "maximum_drawdown",
    "average_premium_expansion", "average_hold_time", "average_r_multiple",
    "maximum_losing_streak", "maximum_winning_streak", "strategy_stability",
    "consistency_score",
]


def execute(data=None) -> pd.DataFrame:
    require_approved_data()
    futures, options = data or load_data({})
    base = InstitutionalHybridStrategy()
    features = base.score(base.build_features(futures, options))
    dataset_digest = frame_hash(features)
    records = []
    for trader in TRADERS:
        package = f"institutional_hybrid.traders.{trader}"
        strategy = importlib.import_module(f"{package}.strategy")
        backtester = importlib.import_module(f"{package}.backtest")
        analytics = importlib.import_module(f"{package}.analytics")
        charts = importlib.import_module(f"{package}.charts")
        signals = strategy.signal(features)
        rejected = signals.loc[signals.rejection_reason.ne("")].copy()
        trades = backtester.run(signals)
        candidates = int((signals.signal.ne(0) | signals.rejection_reason.ne("")).sum())
        rejected_pct = len(rejected) / candidates if candidates else 0.0
        result = analytics.analyse(trades, rejected_pct)
        result["strategy"] = trader
        folder = guard_output(OUTPUT / trader)
        folder.mkdir(parents=True, exist_ok=True)
        trades.to_csv(folder / "trade_journal.csv", index=False)
        rejected.to_csv(folder / "rejected_setups.csv")
        monthly_weekly_tables(trades, folder)
        charts.render(trades, rejected, folder / "charts")
        write_json(folder / "metrics.json", result)
        write_json(folder / "manifest.json", {
            "strategy": trader, "dataset_hash": dataset_digest,
            "rows": len(features), "optimised": False,
        })
        records.append(result)
    board = pd.DataFrame(records)[RANK_COLUMNS]
    board["rank"] = board["expectancy"].rank(ascending=False, method="min").astype(int)
    board = board.sort_values(["rank", "profit_factor"], ascending=[True, False])
    write_leaderboards(board)
    return board


def monthly_weekly_tables(trades: pd.DataFrame, folder: Path) -> None:
    if trades.empty:
        pd.DataFrame(columns=["period", "return"]).to_csv(folder / "monthly_returns.csv", index=False)
        pd.DataFrame(columns=["period", "return"]).to_csv(folder / "weekly_returns.csv", index=False)
        return
    dates = pd.to_datetime(trades.entry_time)
    for name, periods in (("monthly", dates.dt.to_period("M")), ("weekly", dates.dt.to_period("W"))):
        table = trades.assign(period=periods.astype(str)).groupby("period", as_index=False).net_return.sum()
        table.rename(columns={"net_return": "return"}).to_csv(folder / f"{name}_returns.csv", index=False)


def write_leaderboards(board: pd.DataFrame) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    board.to_csv(OUTPUT / "leaderboard.csv", index=False)
    board.to_parquet(OUTPUT / "leaderboard.parquet", index=False)
    (OUTPUT / "leaderboard.html").write_text(board.to_html(index=False), encoding="utf-8")
    (OUTPUT / "leaderboard.md").write_text("# Strategy Leaderboard\n\n" + board.to_markdown(index=False), encoding="utf-8")
    board.to_csv(OUTPUT / "comparison.csv", index=False)
    (OUTPUT / "comparison.html").write_text(board.to_html(index=False), encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--run",action="store_true"); args=parser.parse_args()
    if not args.run: parser.error("choose --run")
    execute()


if __name__ == "__main__": main()
