"""Consume Step-24 predictions and actual option candles into paper trades."""

from __future__ import annotations

from dataclasses import fields
import argparse
import json
from pathlib import Path

import pandas as pd

from src.paper_trading_analytics import (
    empty_trade_frame, equity_and_drawdown, grouped_summary, performance,
    time_summary,
)
from src.paper_trading_engine import (
    OptionCandle, PaperTradingConfig, PaperTradingEngine,
)


def config(path: str) -> PaperTradingConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(PaperTradingConfig)}
    return PaperTradingConfig(**{key: value for key, value in payload.items() if key in allowed})


def predictions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def market(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    mapping = {
        "option_open": "open", "option_high": "high", "option_low": "low",
        "option_close": "close",
    }
    if set(mapping).issubset(frame.columns):
        frame = frame.rename(columns=mapping)
    required = {"timestamp", "open", "high", "low", "close"}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"Option market source missing: {missing}")
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    return frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def write_reports(engine: PaperTradingEngine, reports: Path) -> None:
    reports.mkdir(parents=True, exist_ok=True)
    trades = engine.trades() if engine.csv_path.exists() else empty_trade_frame()
    if trades.empty:
        trades = empty_trade_frame()
        trades.to_csv(engine.csv_path, index=False)
        trades.to_parquet(engine.parquet_path, index=False)
    stats = performance(trades, engine.config.initial_capital)
    daily, monthly = time_summary(trades, "D"), time_summary(trades, "M")
    equity = equity_and_drawdown(trades, engine.config.initial_capital)
    strategy, regime = grouped_summary(trades, "strategy"), grouped_summary(trades, "regime")
    daily.to_csv(reports / "daily_summary.csv", index=False)
    monthly.to_csv(reports / "monthly_summary.csv", index=False)
    equity.to_csv(reports / "equity_curve.csv", index=False)
    pd.DataFrame([stats]).to_csv(reports / "risk_statistics.csv", index=False)
    strategy.to_csv(reports / "strategy_statistics.csv", index=False)
    regime.to_csv(reports / "regime_statistics.csv", index=False)
    metadata = {
        "step": "25", "mode": "PAPER_ONLY", "prediction_source": "Step 24 journal",
        "total_trades": stats["total_trades"], "config": engine.config.__dict__,
        "no_broker_api": True, "order_execution": False,
    }
    (reports / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (reports / "paper_trading_report.md").write_text(
        f"""# Step 25 — Institutional Paper Trading

- Completed paper trades: {stats['total_trades']}
- Net P&L: {stats['net_pnl']:.2f}
- Win rate: {stats['win_rate']:.2%}
- Profit factor: {stats['profit_factor']}
- Maximum drawdown: {stats['maximum_drawdown']:.2%}
- Ending virtual equity: {stats['ending_equity']:.2f}

This engine is paper-only. It contains no broker order interface.
Entries require a Step-24 TRADE plus an exact timestamp-matched option candle.
""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/paper_trading.json")
    parser.add_argument("--predictions", default="logs/live_inference/predictions.jsonl")
    parser.add_argument("--option-market", default="data/aligned/canonical_index_option_aligned.parquet")
    parser.add_argument("--output", default="reports/paper_trading")
    args = parser.parse_args()
    cfg, root = config(args.config), Path(args.output)
    engine = PaperTradingEngine(root, cfg)
    records, candles = predictions(Path(args.predictions)), market(Path(args.option_market))
    cursor_path = root / "prediction_cursor.json"
    cursor = json.loads(cursor_path.read_text())["processed"] if cursor_path.exists() else 0
    pending = records[cursor:]
    by_time: dict[pd.Timestamp, list[dict]] = {}
    for record in pending:
        timestamp = pd.to_datetime(record["timestamp"], utc=True).tz_convert("Asia/Kolkata")
        by_time.setdefault(timestamp, []).append(record)
    if by_time:
        start, end = min(by_time), max(by_time) + pd.Timedelta(minutes=cfg.maximum_holding_minutes + 1)
        window = candles.loc[candles.timestamp.between(start, end)]
        health_path = Path("logs/live_inference/pipeline_health.json")
        health = json.loads(health_path.read_text()) if health_path.exists() else {"status": "SAFE_BLOCK"}
        consumed = set()
        for row in window.itertuples(index=False):
            candle = OptionCandle(row.timestamp.isoformat(), row.open, row.high, row.low, row.close)
            if engine.position is not None:
                engine.on_candle(candle)
            for record in by_time.get(row.timestamp, []):
                engine.consume_prediction(record, candle, health)
                consumed.add((row.timestamp, record["inference_timestamp"]))
        for timestamp, items in by_time.items():
            for record in items:
                if (timestamp, record["inference_timestamp"]) not in consumed:
                    engine._event("SKIP", timestamp, "MISSING_MARKET_TIMESTAMP", record)
    temporary = cursor_path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"processed": len(records)}, indent=2))
    temporary.replace(cursor_path)
    write_reports(engine, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
