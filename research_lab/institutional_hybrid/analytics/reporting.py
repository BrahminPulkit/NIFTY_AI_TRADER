"""Research-only CSV, Markdown and HTML report generation."""
from __future__ import annotations

from pathlib import Path
import html
import json
import pandas as pd


def write_experiment_reports(
    experiment_id: str, trades: pd.DataFrame, metrics: dict[str, float], output_dir: Path
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / experiment_id
    trades_path = stem.with_name(f"{experiment_id}_trades.csv")
    metrics_path = stem.with_name(f"{experiment_id}_metrics.csv")
    md_path = stem.with_suffix(".md")
    html_path = stem.with_suffix(".html")
    trades.to_csv(trades_path, index=False)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    metric_lines = "\n".join(f"- **{k.replace('_', ' ').title()}**: {v:.6g}" for k, v in metrics.items())
    md = f"# Experiment {experiment_id}\n\n## Metrics\n\n{metric_lines}\n\n## Configuration\n\nSee the immutable experiment manifest.\n"
    md_path.write_text(md, encoding="utf-8")
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>"
        + html.escape(experiment_id) + "</title></head><body><pre>"
        + html.escape(md) + "</pre></body></html>", encoding="utf-8",
    )
    return {"trades_csv": trades_path, "metrics_csv": metrics_path, "markdown": md_path, "html": html_path}


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
