import hashlib
import json
from pathlib import Path

import pandas as pd


CSV = Path("data/cleaned/nifty_5year_cleaned.csv")
PARQUET = Path("data/cleaned/nifty_5year_cleaned.parquet")
REPORT = Path("reports/five_year_data_quality/step14c/dataset_hash_report.json")


def test_cleaned_outputs_and_hash():
    assert CSV.exists() and PARQUET.exists() and REPORT.exists()
    expected = json.loads(REPORT.read_text(encoding="utf-8"))["cleaned_csv"]["sha256"]
    assert hashlib.sha256(CSV.read_bytes()).hexdigest() == expected


def test_cleaned_dataset_integrity():
    frame = pd.read_parquet(PARQUET)
    ts = pd.to_datetime(frame.timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    assert ts.is_monotonic_increasing
    assert not ts.duplicated().any()
    assert (ts.dt.second == 0).all()
    minute = ts.dt.hour * 60 + ts.dt.minute
    assert minute.between(555, 929).all()
    assert (frame.high >= frame.low).all()
    assert frame.open.between(frame.low, frame.high).all()
    assert frame.close.between(frame.low, frame.high).all()
    assert frame.volume.isna().all()
    assert frame.volume_raw.notna().all()
