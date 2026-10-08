"""Acquire and validate explicit expired option sessions for research."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.expired_option_dataset import acquire_expiry_session, build_session_features


DEFAULT_PROOF = ["20260407", "20260413", "20260421", "20260428", "20260505"]


def extract_nested(outer: Path, destination: Path, names: list[str]) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    with zipfile.ZipFile(outer) as archive:
        lookup = {Path(name).name: name for name in archive.namelist() if name.endswith(".zip")}
        for stem in names:
            target = destination / f"{stem}.zip"
            if not target.exists():
                member = lookup.get(f"{stem}.zip")
                if member is None:
                    raise FileNotFoundError(f"Archive has no expiry {stem}")
                target.write_bytes(archive.read(member))
            paths.append(target)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", default=str(ROOT / "data/external/shoonyatrader/nifty.zip"))
    parser.add_argument("--expiries", nargs="*", default=DEFAULT_PROOF)
    parser.add_argument("--name", default="proof5")
    args = parser.parse_args()
    raw_dir = ROOT / "data/external/shoonyatrader" / f"{args.name}_raw"
    archives = extract_nested(Path(args.outer), raw_dir, args.expiries)
    nifty = pd.read_parquet(ROOT / "data/features/feature_dataset.parquet")
    output = ROOT / "data/normalized/historical_options" / args.name
    report_dir = ROOT / "reports/phase8_option_data"
    output.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    audits, feature_frames = [], []
    for archive in archives:
        result = acquire_expiry_session(archive, nifty)
        session_dir = output / result.date
        session_dir.mkdir(parents=True, exist_ok=True)
        result.nifty.to_parquet(session_dir / "nifty.parquet", index=False)
        result.ce.to_parquet(session_dir / "ce.parquet", index=False)
        result.pe.to_parquet(session_dir / "pe.parquet", index=False)
        (session_dir / "manifest.json").write_text(
            json.dumps(result.audit, indent=2), encoding="utf-8")
        audits.append(result.audit)
        if result.audit["research_training_eligible"]:
            feature_frames.append(build_session_features(result))
    summary = {
        "source": "Shoonya Trader public expired NIFTY option archive",
        "source_url": "https://shoonyatrader.in/free-historical-expired-options-contract-data/",
        "security_id_policy": "not_available_research_only",
        "sessions_attempted": len(audits),
        "training_ready": sum(row["status"] == "TRAINING_READY" for row in audits),
        "partial": sum(row["status"] == "PARTIAL" for row in audits),
        "sessions": audits,
    }
    (report_dir / f"{args.name}_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(audits).to_csv(report_dir / f"{args.name}_coverage.csv", index=False)
    if feature_frames:
        combined = pd.concat(feature_frames, ignore_index=True)
        combined.to_parquet(output / "feature_dataset_49.parquet", index=False)
        summary["feature_rows"] = len(combined)
        from src.ce_pe_feature_preview import feature_columns
        summary["fully_populated_feature_rows"] = int(
            combined[feature_columns()].dropna().shape[0])
        (report_dir / f"{args.name}_audit.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
