"""Acquire strict non-expiry sessions from explicit weekly contract archives."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.expired_option_dataset import acquire_expiry_session, build_session_features  # noqa: E402


RAW = ROOT / "data/external/shoonyatrader/validated20_raw"
OUTPUT = ROOT / "data/normalized/historical_options/non_expiry15"
REPORT = ROOT / "reports/phase10_label_suitability"
TARGET = 15


def main() -> None:
    nifty = pd.read_parquet(ROOT / "data/features/feature_dataset.parquet")
    nifty["date"] = nifty.timestamp.dt.strftime("%Y-%m-%d")
    complete_dates = sorted(nifty.groupby("date").size().loc[lambda x: x.eq(375)].index)
    attempts, accepted, features = [], [], []
    archives = sorted(RAW.glob("*.zip"), reverse=True)
    for archive in archives:
        expiry = pd.to_datetime(archive.stem, format="%Y%m%d").strftime("%Y-%m-%d")
        prior = [date for date in complete_dates if date < expiry]
        if not prior:
            continue
        date = prior[-1]
        try:
            result = acquire_expiry_session(archive, nifty.drop(columns="date"), session_date=date)
            audit = dict(result.audit, source_archive=archive.name, failure_reason=None)
            attempts.append(audit)
            if audit["status"] != "TRAINING_READY":
                continue
            session_dir = OUTPUT / f"{date}__{expiry}"
            session_dir.mkdir(parents=True, exist_ok=True)
            result.nifty.to_parquet(session_dir / "nifty.parquet", index=False)
            result.ce.to_parquet(session_dir / "ce.parquet", index=False)
            result.pe.to_parquet(session_dir / "pe.parquet", index=False)
            (session_dir / "manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
            frame = build_session_features(result)
            frame["expiry"] = expiry
            features.append(frame)
            accepted.append(audit)
        except Exception as exc:
            attempts.append({"date": date, "expiry": expiry, "source_archive": archive.name,
                             "status": "PARTIAL", "failure_reason": str(exc)})
        if len(accepted) >= TARGET:
            break
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    if features:
        pd.concat(features, ignore_index=True).to_parquet(
            OUTPUT / "feature_dataset_49.parquet", index=False)
    summary = {
        "status": "NON_EXPIRY_ACQUISITION_COMPLETE" if len(accepted) >= 10 else "DATA_INSUFFICIENT",
        "target": TARGET, "attempted": len(attempts), "validated": len(accepted),
        "partial": sum(row["status"] != "TRAINING_READY" for row in attempts),
        "sessions": attempts,
    }
    (REPORT / "non_expiry_acquisition.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(attempts).to_csv(REPORT / "non_expiry_coverage.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
