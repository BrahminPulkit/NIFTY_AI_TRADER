from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from institutional_hybrid.utils.production_v1_adapter import (
    DatasetNotSuitableError, ProductionV1ReadOnlyAdapter,
)


def test_adapter_is_pinned_and_fails_closed_on_v1():
    adapter = ProductionV1ReadOnlyAdapter()
    before = adapter.source.stat()
    audit = adapter.audit()
    after = adapter.source.stat()
    assert audit.rows == 7643
    assert not audit.suitable
    assert "option_close" in audit.missing_continuous_fields
    assert "futures_volume" in audit.missing_continuous_fields
    assert "expected_premium_expansion" in audit.forbidden_proxy_fields_present
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    try:
        adapter.load_for_phase1()
    except DatasetNotSuitableError:
        pass
    else:
        raise AssertionError("unsuitable production data was accepted")


def test_adapter_rejects_unapproved_source():
    try:
        ProductionV1ReadOnlyAdapter(Path("different.parquet"))
    except ValueError:
        pass
    else:
        raise AssertionError("adapter accepted a non-V1 source")
