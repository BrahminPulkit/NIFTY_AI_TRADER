"""CLI entry point for a reproducible, isolated experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from institutional_hybrid.analytics.charts import plot_research_dashboard
    from institutional_hybrid.analytics.reporting import write_experiment_reports, write_manifest
    from institutional_hybrid.backtests import BacktestConfig, ResearchBacktester
    from institutional_hybrid.strategies import InstitutionalHybridStrategy
    from institutional_hybrid.utils.io import LAB_ROOT, assert_lab_output, read_market_csv
else:
    from .analytics.charts import plot_research_dashboard
    from .analytics.reporting import write_experiment_reports, write_manifest
    from .backtests import BacktestConfig, ResearchBacktester
    from .strategies import InstitutionalHybridStrategy
    from .utils.io import LAB_ROOT, assert_lab_output, read_market_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--futures", type=Path, required=True)
    parser.add_argument("--options", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=LAB_ROOT / "configs" / "baseline.yaml")
    parser.add_argument("--experiment-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    futures, options = read_market_csv(args.futures), read_market_csv(args.options)
    strategy = InstitutionalHybridStrategy(config["strategy"]["threshold"], config["weights"])
    scored = strategy.score(strategy.build_features(futures, options))
    bt = ResearchBacktester(BacktestConfig(**config["backtest"]))
    trades, metrics = bt.run(scored)
    report_dir = assert_lab_output(LAB_ROOT / "reports" / args.experiment_id)
    chart_dir = assert_lab_output(LAB_ROOT / "charts" / args.experiment_id)
    write_experiment_reports(args.experiment_id, trades, metrics, report_dir)
    plot_research_dashboard(scored, chart_dir / "diagnostic.png", args.experiment_id)
    write_manifest(
        report_dir / "manifest.json",
        {"experiment_id": args.experiment_id, "created_utc": datetime.now(timezone.utc),
         "config": config, "futures_source": str(args.futures.resolve()),
         "options_source": str(args.options.resolve()), "rows": len(scored)},
    )
    scored.to_csv(report_dir / "scored_bars.csv")
    print(f"Experiment complete: {report_dir}")


if __name__ == "__main__":
    main()
