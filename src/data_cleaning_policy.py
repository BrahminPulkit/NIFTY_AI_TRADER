"""Step 14B policy evidence generator.

This module classifies proposed actions only. It never writes a cleaned dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MUHURAT_DATES = {"2021-11-04", "2022-10-24", "2023-11-12", "2024-11-01", "2025-10-21"}
SPECIAL_DATES = {"2024-01-20", "2024-03-02", "2024-05-18", "2025-02-01", "2026-02-01"}


def _table(rows: list[dict]) -> str:
    if not rows:
        return "None."
    keys = list(rows[0])
    head = "| " + " | ".join(keys) + " |"
    rule = "| " + " | ".join("---" for _ in keys) + " |"
    body = ["| " + " | ".join(str(row[k]) for k in keys) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def design_policy(source: Path, report_dir: Path) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(source)
    ts = pd.to_datetime(raw.timestamp, unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    frame = raw.copy()
    frame.insert(0, "source_row", np.arange(2, len(frame) + 2))
    frame["timestamp_ist"] = ts
    frame["date"] = ts.dt.strftime("%Y-%m-%d")
    frame["minute_bucket"] = ts.dt.floor("min")
    frame["second"] = ts.dt.second
    frame["day_of_week"] = ts.dt.dayofweek
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    frame["regular_time"] = minute_of_day.between(555, 929)
    frame["session_class"] = "regular_weekday"
    frame.loc[frame.date.isin(MUHURAT_DATES), "session_class"] = "muhurat"
    frame.loc[frame.date.isin(SPECIAL_DATES), "session_class"] = "special_trading"
    weekend_other = frame.day_of_week.ge(5) & ~frame.date.isin(MUHURAT_DATES | SPECIAL_DATES)
    frame.loc[weekend_other, "session_class"] = "weekend_provider_anomaly"
    ordinary_out = ~frame.regular_time & frame.session_class.eq("regular_weekday")
    frame.loc[ordinary_out, "session_class"] = "ordinary_out_of_session"

    canonical_scope = frame[frame.session_class.eq("regular_weekday") & frame.regular_time].copy()
    collisions = canonical_scope[canonical_scope.minute_bucket.duplicated(keep=False)].copy()
    offset_collision = collisions[collisions.second.ne(0)]
    normalized = canonical_scope.sort_values(["minute_bucket", "second", "source_row"]).drop_duplicates(
        "minute_bucket", keep="first")

    incomplete_dates: list[str] = []
    missing_rows: list[dict] = []
    for date, day in normalized.groupby("date"):
        expected = pd.date_range(f"{date} 09:15", periods=375, freq="min", tz="Asia/Kolkata")
        missing = expected.difference(pd.DatetimeIndex(day.minute_bucket))
        if len(missing):
            incomplete_dates.append(date)
            missing_rows.extend({"date": date, "missing_timestamp": value.isoformat(),
                                 "category": "provider_gap_or_download_failure",
                                 "required_action": "retry_download_then_cross-check_provider"}
                                for value in missing)
    incomplete = normalized[normalized.date.isin(incomplete_dates)]
    strict = normalized[~normalized.date.isin(incomplete_dates)].copy()

    open_bad = (strict.open < strict.low) | (strict.open > strict.high)
    close_bad = (strict.close < strict.low) | (strict.close > strict.high)
    close_delta = pd.concat([(strict.close - strict.high).clip(lower=0),
                             (strict.low - strict.close).clip(lower=0)], axis=1).max(axis=1)
    precision_bad = close_bad & close_delta.le(0.001)
    substantive_ohlc = open_bad | (close_bad & ~precision_bad)
    strict["range_pct"] = (strict.high - strict.low) / strict.open * 100
    gap_review = strict[strict.range_pct.gt(1.0)].copy()

    safe_parts: list[pd.DataFrame] = []
    excluded_session = frame[~(frame.session_class.eq("regular_weekday") & frame.regular_time)].copy()
    excluded_session["policy_action"] = "segregate_from_regular_production_dataset"
    excluded_session["reason"] = excluded_session.session_class
    safe_parts.append(excluded_session)
    safe_offset = offset_collision.copy()
    safe_offset["policy_action"] = "quarantine_offset_duplicate_keep_exact_minute_row"
    safe_offset["reason"] = "minute_bucket_collision"
    safe_parts.append(safe_offset)
    safe_precision = strict[precision_bad].copy()
    safe_precision["policy_action"] = "round_ohlc_to_two_decimals_then_revalidate"
    safe_precision["reason"] = "floating_precision_boundary_violation"
    safe_parts.append(safe_precision)
    safe = pd.concat(safe_parts, ignore_index=True, sort=False)
    safe[["source_row", "timestamp_ist", "reason", "policy_action"]].to_csv(
        report_dir / "rows_safe_for_automatic_cleaning.csv", index=False)

    manual_parts: list[pd.DataFrame] = []
    manual_ohlc = strict[substantive_ohlc].copy()
    manual_ohlc["reason"] = "substantive_ohlc_violation"
    manual_ohlc["required_action"] = "redownload_and_cross-check; quarantine_session_if_unresolved"
    manual_parts.append(manual_ohlc)
    manual_gap = gap_review.copy()
    manual_gap["reason"] = "candle_range_above_1pct"
    manual_gap["required_action"] = "cross-check_independent_market_source; retain_if_confirmed"
    manual_parts.append(manual_gap)
    manual = pd.concat(manual_parts, ignore_index=True, sort=False)
    manual[["source_row", "timestamp_ist", "open", "high", "low", "close", "volume",
            "reason", "required_action"]].drop_duplicates(["source_row", "reason"]).to_csv(
        report_dir / "rows_requiring_manual_review.csv", index=False)
    pd.DataFrame(missing_rows).to_csv(report_dir / "missing_candles_policy_review.csv", index=False)

    decisions = [
        {"issue": "Negative volume", "severity": "Major", "rule": "Never alter OHLC or remove row solely for volume. Preserve volume_raw; set analytical volume to null for the entire index history.", "automatic": "Yes", "affected": int((frame.volume < 0).sum())},
        {"issue": "Zero/inconsistent volume", "severity": "Critical", "rule": "Preserve volume_raw for audit; analytical volume is null for all rows. No index volume feature until a consistent source exists.", "automatic": "Yes after approval", "affected": int(len(frame))},
        {"issue": "Ordinary out-of-session", "severity": "Major", "rule": "Segregate from canonical production data; never merge with 09:15-15:29 bars.", "automatic": "Yes", "affected": int(frame.session_class.eq("ordinary_out_of_session").sum())},
        {"issue": "Muhurat session", "severity": "Minor", "rule": "Retain in a separate special_sessions archive; exclude from ordinary intraday training baseline.", "automatic": "Yes", "affected": int(frame.session_class.eq("muhurat").sum())},
        {"issue": "Known special trading", "severity": "Minor", "rule": "Retain separately with session metadata; exclude from ordinary baseline unless a later regime-aware policy opts in.", "automatic": "Yes", "affected": int(frame.session_class.eq("special_trading").sum())},
        {"issue": "Weekend/provider anomaly", "severity": "Critical", "rule": "Quarantine. Admit only after an NSE notice and independent price validation establish a real live session.", "automatic": "Quarantine only", "affected": int(frame.session_class.eq("weekend_provider_anomaly").sum())},
        {"issue": "Seconds-offset timestamp", "severity": "Major", "rule": "Floor only after validating one observation per bucket. If an exact-minute row exists, quarantine the offset duplicate; never average OHLC.", "automatic": "Conditional", "affected": int((ts.dt.second != 0).sum())},
        {"issue": "Minute-bucket collision", "severity": "Critical", "rule": "Prefer the exact-minute record only when unique; quarantine competing offset record. Otherwise manual provider comparison.", "automatic": "Conditional", "affected": int(frame.minute_bucket.duplicated(keep=False).sum())},
        {"issue": "Verified holiday/closure", "severity": "Minor", "rule": "No candle expected; do not synthesize a row and do not classify as missing.", "automatic": "Yes", "affected": 71},
        {"issue": "Provider gap/download failure", "severity": "Major", "rule": "Retry source request. Never forward-fill OHLCV. Until recovered, quarantine the entire affected session from strict baseline.", "automatic": "Retry; quarantine", "affected": len(missing_rows)},
        {"issue": "Floating OHLC boundary", "severity": "Minor", "rule": "Round all OHLC to two decimal places and revalidate; allowed only when original violation <=0.001 point.", "automatic": "Yes", "affected": int(precision_bad.sum())},
        {"issue": "Substantive OHLC violation", "severity": "Critical", "rule": "Redownload/cross-check. Never clamp or invent values. If unresolved, quarantine the complete session.", "automatic": "No", "affected": int(substantive_ohlc.sum())},
        {"issue": "Abnormal price/range gap", "severity": "Major", "rule": "Do not remove by threshold alone. Validate timestamp continuity, neighbouring prices and independent source; retain genuine market gaps.", "automatic": "No", "affected": int(len(gap_review))},
    ]
    pd.DataFrame(decisions).to_csv(report_dir / "cleaning_decision_matrix.csv", index=False)
    pd.DataFrame(decisions)[["issue", "severity", "affected"]].to_csv(
        report_dir / "severity_classification.csv", index=False)

    unresolved_ohlc_days = strict.loc[substantive_ohlc, "date"].nunique()
    estimates = {
        "raw_rows": int(len(frame)),
        "automatic_session_scope_rows": int(len(canonical_scope)),
        "session_rows_segregated": int(len(frame) - len(canonical_scope)),
        "offset_collision_rows_quarantined": int(len(offset_collision)),
        "rows_after_session_and_collision_policy": int(len(normalized)),
        "missing_minute_buckets_in_ordinary_sessions": len(missing_rows),
        "incomplete_sessions": len(incomplete_dates),
        "rows_quarantined_if_incomplete_sessions_unrecovered": int(len(incomplete)),
        "strict_complete_session_rows": int(len(strict)),
        "target_rows_if_all_missing_bars_recovered": int(len(normalized) + len(missing_rows)),
        "target_rows_if_missing_bars_unrecovered": int(len(strict)),
        "conservative_rows_if_substantive_ohlc_sessions_also_unresolved": int(len(strict) - unresolved_ohlc_days * 375),
        "substantive_ohlc_sessions": int(unresolved_ohlc_days),
        "records_physically_deleted": 0,
    }
    (report_dir / "cleaning_policy_estimates.json").write_text(json.dumps(estimates, indent=2), encoding="utf-8")

    policy = f"""# Step 14B — Institutional Data Cleaning Policy

## Status and governing principles

**Policy version:** 1.0-draft  
**Status:** Awaiting explicit approval; not applied.  
**Source:** `{source.as_posix()}`

The raw file is immutable. Cleaning must produce a new versioned dataset plus a row-level manifest. No OHLC candle is synthesized, forward-filled, averaged, or silently removed. Every excluded row remains recoverable by `source_row`.

## Production session policy

- Canonical baseline: ordinary weekday candles timestamped 09:15 through 15:29 Asia/Kolkata.
- Muhurat and known special sessions are preserved in a separate special-session dataset, not mixed into the baseline.
- Other weekend rows are quarantined until matched to an NSE live-session notice and independently validated.
- Pre-open, closing-session and other out-of-session rows are segregated from the baseline.

## Volume policy

The index volume field changes semantics across the sample and is unusable as one continuous measure. Preserve it as `volume_raw`. In a future cleaned dataset, create nullable `volume` containing null for **every** row, not only zero/negative rows. This avoids teaching a model that provider availability is a market signal. Rows are not removed solely because volume is zero or negative. Volume-derived features remain prohibited until a separately verified, consistent source is approved.

## Timestamp policy

Convert Unix epoch seconds as UTC and display/store canonical timestamps in Asia/Kolkata. For seconds-offset observations, assign a minute bucket only after collision validation. Where one exact-minute row and one offset row collide, retain the exact-minute row and quarantine the offset record. Do not average candles. Any collision without exactly one canonical record requires manual review.

## Missing-candle policy

- Weekend or verified exchange closure: no bar expected.
- Muhurat/special session: validate against that session's own timetable; never use the 375-bar regular template.
- Ordinary-session missing bucket: retry the same Dhan request and compare a second authoritative source.
- Never interpolate or forward-fill OHLC.
- If still unresolved, quarantine the complete session from the strict training baseline while retaining it in an incomplete-session archive.

## OHLC policy

First round raw numeric representation to two decimal places and revalidate. A boundary discrepancy no larger than 0.001 point is classified as serialization precision and may be corrected automatically with before/after values logged. Larger violations must be redownloaded and independently checked. Never clamp substantive violations. If unresolved, quarantine the entire session.

## Gap policy

A threshold breach is not proof of corruption. Validate session continuity, adjacent close/open, duplicated buckets, known market events and an independent source. Confirmed market gaps remain unchanged. Unconfirmed gaps are manually quarantined; they are never winsorized or capped.

## Decision matrix

{_table(decisions)}

## Size estimate

- Raw: **{estimates['raw_rows']:,}** rows.
- Ordinary regular-session scope: **{estimates['automatic_session_scope_rows']:,}**.
- After deterministic collision quarantine: **{estimates['rows_after_session_and_collision_policy']:,}**.
- If all 28 missing bars are recovered: **{estimates['target_rows_if_all_missing_bars_recovered']:,}**.
- If incomplete sessions remain unrecovered and are quarantined: **{estimates['target_rows_if_missing_bars_unrecovered']:,}**.
- Conservative lower estimate if the two substantive-OHLC sessions also remain unresolved: **{estimates['conservative_rows_if_substantive_ohlc_sessions_also_unresolved']:,}**.

The approved final size is therefore expected between **{estimates['conservative_rows_if_substantive_ohlc_sessions_also_unresolved']:,} and {estimates['target_rows_if_all_missing_bars_recovered']:,}**, depending only on documented recovery outcomes.

## Approval gate

No policy action may run until explicit approval. Implementation must produce a new dataset, source hash, policy version, action counts, before/after manifest, and output hash. Step 14B does not authorize feature generation or labels.
"""
    (report_dir / "data_cleaning_policy.md").write_text(policy, encoding="utf-8")
    return estimates

