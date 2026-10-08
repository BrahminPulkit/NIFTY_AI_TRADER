from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.production_horizon_research import build_research_tables


INPUT = Path("data/setups/stage1_setup_dataset.parquet")
REPORTS = Path("reports/production_label_horizon_research")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    source = pd.read_parquet(INPUT)
    if not source.trade_allowed.isin([0, 1]).all():
        raise RuntimeError("Invalid Stage-1 trade_allowed values")
    tables = build_research_tables(source)
    tables["comparison"].to_csv(REPORTS / "horizon_comparison.csv", index=False)
    tables["directional"].to_csv(REPORTS / "expected_r_by_horizon.csv", index=False)
    tables["yearly"].to_csv(REPORTS / "yearly_horizon_statistics.csv", index=False)
    tables["regime"].to_csv(REPORTS / "regime_horizon_statistics.csv", index=False)
    tables["session"].to_csv(REPORTS / "session_horizon_statistics.csv", index=False)
    tables["excursions"].to_csv(REPORTS / "mfe_mae_statistics.csv", index=False)

    comparison = tables["comparison"]
    best_er = comparison.loc[comparison.expected_r.idxmax()]
    stable = comparison.loc[comparison.yearly_expected_r_std.idxmin()]
    lowest_timeout = comparison.loc[comparison.timeout_rate.idxmin()]
    directional = tables["directional"].pivot(index="horizon", columns="direction", values="expected_r")
    positive_both = directional[(directional.BUY_CALL > 0) & (directional.BUY_PUT > 0)].index.tolist()
    positive_text = ", ".join(map(str, positive_both)) if positive_both else "none"
    (REPORTS / "recommendation.md").write_text(f"""# Production Label Horizon Research Recommendation

No production horizon is selected in this step.

- Highest research Expected R: **{int(best_er.horizon)} candles** ({best_er.expected_r:.4f}R)
- Lowest year-direction dispersion: **{int(stable.horizon)} candles**
- Lowest timeout rate: **{int(lowest_timeout.horizon)} candles** ({lowest_timeout.timeout_rate:.2%})
- Horizons with positive Expected R in both directions: **{positive_text}**

These findings describe only Stage-1 V2 allowed opportunities under the fixed 0.20% stop / 0.30% target contract. They are not a horizon promotion, backtest, or model result.
""", encoding="utf-8")
    rows = []
    for row in comparison.itertuples():
        rows.append(f"| {row.horizon} | {row.win_rate:.2%} | {row.stop_loss_hit_rate:.2%} | "
                    f"{row.timeout_rate:.2%} | {row.expected_r:.4f} | {row.average_mfe_r:.3f} | "
                    f"{row.average_mae_r:.3f} | {row.opportunity_frequency_per_day:.2f} |")
    (REPORTS / "label_horizon_report.md").write_text(f"""# Production Label Horizon Research

## Scope

Only `stage1_v2` rows with `trade_allowed == 1` are evaluated, in their emitted LONG/SHORT direction. Entry is candle close; stop is 0.20%; target is 0.30% (1.5R); gaps fill at the first available open; ambiguous candles are stop-first; and observations never cross sessions. MFE/MAE are measured through label exit. This is overlapping label research, not a backtest.

| Horizon | Win | Stop | Timeout | Expected R | MFE R | MAE R | Opportunities/session |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

No production horizon is selected or changed.
""", encoding="utf-8")
    freeze = {
        "status": "FROZEN", "engine_version": "stage1_v2",
        "setup_engine_source_sha256": digest(Path("src/setup_engine.py")),
        "setup_dataset_sha256": digest(INPUT),
        "setup_metadata_sha256": digest(Path("reports/setup_engine/setup_metadata.json")),
    }
    Path("reports/setup_engine/stage1_v2_freeze_manifest.json").write_text(
        json.dumps(freeze, indent=2), encoding="utf-8")
    metadata = {"stage1_freeze": freeze, "input": INPUT.as_posix(), "allowed_setups": int(source.trade_allowed.sum()),
                "horizons": [5, 10, 15, 20], "research_observations": len(tables["observations"])}
    (REPORTS / "research_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Production horizon research complete: {len(tables['observations']):,} observations")


if __name__ == "__main__":
    main()

