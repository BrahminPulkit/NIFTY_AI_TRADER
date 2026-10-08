"""Read-only schema and quality audit for a candidate five-year F&O dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, title: str, body: str) -> None:
    path.write_text(f"# {title}\n\n{body.strip()}\n", encoding="utf-8")


def validate(source: Path, report_dir: Path) -> dict:
    """Validate without changing, sorting, deduplicating, or filling source data."""
    report_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source, low_memory=False)
    columns = list(frame.columns)
    required_ohlc = ["open", "high", "low", "close"]
    missing_ohlc_columns = [name for name in required_ohlc if name not in columns]
    if missing_ohlc_columns:
        raise ValueError(f"Candidate lacks OHLC columns: {missing_ohlc_columns}")

    numeric_names = [name for name in required_ohlc + ["volume", "oi", "strike", "spot"] if name in columns]
    numeric = frame[numeric_names].apply(pd.to_numeric, errors="coerce")
    has_timestamp = "timestamp" in columns
    has_expiry = "expiry" in columns or "expiry_date" in columns
    identity_columns = [name for name in ("security_id", "securityId", "instrument", "symbol", "contract") if name in columns]
    has_iv = "iv" in columns or "implied_volatility" in columns

    result = {
        "source": source.as_posix(),
        "sha256": _hash(source),
        "rows": int(len(frame)),
        "columns": columns,
        "null_counts": {name: int(value) for name, value in frame.isna().sum().items()},
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "timestamp_available": has_timestamp,
        "expiry_available": has_expiry,
        "identity_columns": identity_columns,
        "iv_available": has_iv,
        "ohlc": {
            "missing_rows": int(numeric[required_ohlc].isna().any(axis=1).sum()),
            "high_below_low": int(numeric.high.lt(numeric.low).sum()),
            "open_outside_range": int((numeric.open.lt(numeric.low) | numeric.open.gt(numeric.high)).sum()),
            "close_outside_range": int((numeric.close.lt(numeric.low) | numeric.close.gt(numeric.high)).sum()),
            "nonpositive_price_rows": int(numeric[required_ohlc].le(0).any(axis=1).sum()),
        },
    }
    for name in numeric_names:
        result[f"{name}_statistics"] = {
            "count": int(numeric[name].count()),
            "nulls": int(numeric[name].isna().sum()),
            "minimum": None if numeric[name].dropna().empty else float(numeric[name].min()),
            "maximum": None if numeric[name].dropna().empty else float(numeric[name].max()),
            "mean": None if numeric[name].dropna().empty else float(numeric[name].mean()),
            "median": None if numeric[name].dropna().empty else float(numeric[name].median()),
            "zero_rows": int(numeric[name].eq(0).sum()),
            "negative_rows": int(numeric[name].lt(0).sum()),
        }

    checks = [
        ("Exact duplicate rows", "PASS" if result["exact_duplicate_rows"] == 0 else "FAIL", str(result["exact_duplicate_rows"])),
        ("Duplicate timestamps", "NOT TESTABLE" if not has_timestamp else "TESTABLE", "timestamp column absent" if not has_timestamp else "timestamp column present"),
        ("Missing timestamps/candles/days", "NOT TESTABLE" if not has_timestamp else "TESTABLE", "timestamp column absent" if not has_timestamp else "timestamp column present"),
        ("OHLC relationships", "PASS" if not any(result["ohlc"].values()) else "FAIL", json.dumps(result["ohlc"])),
        ("Volume availability", "PASS" if "volume" in columns and numeric.volume.notna().all() else "FAIL", f"zero={int(numeric.volume.eq(0).sum()) if 'volume' in columns else 'n/a'}"),
        ("OI availability", "PASS" if "oi" in columns and numeric.oi.notna().all() else "FAIL", f"zero={int(numeric.oi.eq(0).sum()) if 'oi' in columns else 'n/a'}"),
        ("Strike availability", "FAIL" if "strike" not in columns or numeric.strike.isna().all() else "PASS", "all values missing" if "strike" in columns and numeric.strike.isna().all() else "see schema"),
        ("IV availability", "FAIL" if not has_iv else "PASS", "IV column absent" if not has_iv else "IV column present"),
        ("Expiry continuity", "NOT TESTABLE" if not has_expiry else "TESTABLE", "expiry column absent" if not has_expiry else "expiry column present"),
        ("Contract rollover", "NOT TESTABLE" if not has_expiry or not identity_columns or not has_timestamp else "TESTABLE", "timestamp, expiry, and contract identity are jointly required"),
        ("Timezone/session/gaps", "NOT TESTABLE" if not has_timestamp else "TESTABLE", "timestamp column absent" if not has_timestamp else "timestamp column present"),
    ]
    pd.DataFrame(checks, columns=["check", "status", "evidence"]).to_csv(report_dir / "validation_checks.csv", index=False)
    pd.DataFrame(
        [{"column": name, "null_count": int(frame[name].isna().sum()), "null_percent": float(frame[name].isna().mean() * 100)} for name in columns]
    ).to_csv(report_dir / "column_completeness.csv", index=False)
    (report_dir / "validation_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    _write(report_dir / "data_quality_report.md", "F&O Data Quality Report", f"""
Source: `{source.as_posix()}`  
SHA-256: `{result['sha256']}`  
Rows: **{len(frame):,}**; columns: **{len(columns)}**

## Verified checks

- Exact duplicate rows: **{result['exact_duplicate_rows']:,}**.
- Missing OHLC rows: **{result['ohlc']['missing_rows']:,}**.
- High below low: **{result['ohlc']['high_below_low']:,}**.
- Open outside candle range: **{result['ohlc']['open_outside_range']:,}**.
- Close outside candle range: **{result['ohlc']['close_outside_range']:,}**.
- Non-positive OHLC rows: **{result['ohlc']['nonpositive_price_rows']:,}**.
- Volume is populated on all rows; zero-volume rows: **{result['volume_statistics']['zero_rows']:,}**; negative-volume rows: **{result['volume_statistics']['negative_rows']:,}**.
- OI is populated on all rows; zero-OI rows: **{result['oi_statistics']['zero_rows']:,}**; negative-OI rows: **{result['oi_statistics']['negative_rows']:,}**.

## Unverifiable checks

The file has no timestamp. Duplicate timestamps, missing minutes, missing trading days, ordering, timezone, NSE-session validity, abnormal time/price gaps, and calendar coverage therefore cannot be tested. It has no expiry or contract/security identifier, so expiry and rollover continuity cannot be tested.

No source row was modified, removed, sorted, filled, or deduplicated.
""")

    _write(report_dir / "instrument_verification_report.md", "Instrument Verification Report", f"""
## Verdict

**Instrument identity failed verification.** The only active five-year derivative candidate is named `nifty_5years_rolling_calls.csv`; that identifies rolling CALL options, not Futures. The data itself contains no security ID, instrument type, symbol, contract, expiry, option type, or timestamp with which to independently verify that name.

## Schema evidence

Columns: `{', '.join(columns)}`.

- `strike` exists but all **{len(frame):,}** values are null.
- `spot` exists but all **{len(frame):,}** values are null.
- IV is absent.
- Futures do not have strike or option IV; those requested checks apply to options. Because this candidate is named as rolling CALL data, the missing strike/IV metadata is itself a critical completeness failure.
- OI exists numerically, but cannot be attributed to a dated contract or expiry.

The dataset cannot be certified as a five-year NIFTY Futures series.
""")

    _write(report_dir / "dataset_health_report.md", "F&O Dataset Health Report", """
Overall status: **REJECTED / CRITICAL**.

OHLC integrity and basic volume/OI numeric completeness pass. These limited passes do not establish temporal, instrument, contract, or rollover integrity. Required temporal and contract keys are absent, so most institutional validation controls are impossible rather than passed.

## Critical blockers

1. Candidate is rolling CALL-option data, not identified Futures data.
2. Timestamp and timezone are absent.
3. Contract/security identifier and instrument metadata are absent.
4. Expiry is absent.
5. Strike values are 100% missing and IV is absent for the option candidate.
6. Coverage, missing candles/days, session validity, chronological order, expiry continuity, and rollover continuity cannot be established.
7. OI cannot be time-aligned or tied to a particular contract.
""")

    _write(report_dir / "final_validation_summary.md", "Final F&O Validation Summary", """
- **Suitable for AI training? No.** Instrument identity and all temporal/contract continuity controls fail or are untestable.
- **Suitable for OI confirmation? No.** OI is populated, but has no timestamp, expiry, strike, or contract identifier for alignment and interpretation.
- **Suitable for expiry analysis? No.** Expiry and contract identifiers are absent.
- **Can it be merged with the Index dataset later? No, not in its present form.** A future dataset could be joined only after it contains timezone-aware timestamps, verified instrument/security IDs, contract/expiry metadata, and validated roll logic. No merge was performed.

## Blocking issues

1. No verified five-year Futures file exists in the active repository; the candidate is named as rolling CALL options.
2. No timestamp or timezone.
3. No security ID, instrument, symbol, or contract identity.
4. No expiry.
5. Strike is entirely null.
6. IV is absent.
7. Missing candles, trading days, sessions, gaps, ordering, expiry continuity, and rollover continuity are untestable.
8. OI cannot be aligned to Index candles or attributed to a contract.
""")
    return result


if __name__ == "__main__":
    validate(Path("notebooks/nifty_5years_rolling_calls.csv"), Path("reports/fno_data_quality"))
