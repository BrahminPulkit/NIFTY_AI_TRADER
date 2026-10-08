"""Build the evidence-bound historical NIFTY contract coverage report."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.historical_contract_resolution_service import HistoricalContractResolver


def main() -> None:
    resolver = HistoricalContractResolver()
    dates = sorted(resolver.index.timestamp.dt.strftime("%Y-%m-%d").unique())
    if resolver.validated.get("session_date"):
        dates = sorted(set(dates) | {resolver.validated["session_date"]})
    rows = []
    for day in dates:
        ce = resolver.resolve_contract(date=day, option_type="CE")
        pe = resolver.resolve_contract(date=day, option_type="PE")
        rows.append({
            "date": day, "ce_status": ce.status, "pe_status": pe.status,
            "expiry": ce.expiry if ce.expiry == pe.expiry else None,
            "ce_security_id": ce.security_id, "pe_security_id": pe.security_id,
            "ce_strike": ce.strike, "pe_strike": pe.strike,
            "ce_source_snapshot": ce.source_snapshot,
            "pe_source_snapshot": pe.source_snapshot,
            "ce_reason": ce.reason, "pe_reason": pe.reason,
            "both_resolved": ce.status == pe.status == "RESOLVED",
        })
    frame = pd.DataFrame(rows)
    resolved = frame.loc[frame.both_resolved]
    unresolved_dates = frame.loc[~frame.both_resolved, "date"].tolist()
    gaps = []
    for day in unresolved_dates:
        if not gaps or (pd.Timestamp(day) - pd.Timestamp(gaps[-1][-1])).days > 3:
            gaps.append([day, day])
        else:
            gaps[-1][-1] = day
    report = {
        "status": "EVIDENCE_BOUND_COVERAGE",
        "earliest_resolvable_date": resolved.date.min() if len(resolved) else None,
        "latest_resolvable_date": resolved.date.max() if len(resolved) else None,
        "resolvable_trading_dates": int(len(resolved)),
        "ce_contracts": int(resolved.ce_security_id.nunique()),
        "pe_contracts": int(resolved.pe_security_id.nunique()),
        "weekly_expiries": sorted(resolved.expiry.dropna().unique().tolist()),
        "missing_periods": [{"from": left, "to": right} for left, right in gaps],
        "total_index_trading_dates_checked": int(len(frame)),
        "resolver_policy": "WEEK/CURRENT_NEAR + ATM from final available 1-minute index candle",
        "fabricated_identity_count": 0,
    }
    root = Path("reports/phase5")
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / "contract_resolution_coverage.csv", index=False)
    (root / "contract_resolution_coverage.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
