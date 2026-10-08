from pathlib import Path


def test_paper_engine_contains_no_broker_order_interfaces():
    source = "\n".join(
        Path(path).read_text(encoding="utf-8").lower()
        for path in (
            "src/paper_trading_engine.py",
            "src/paper_trading_manager.py",
            "apps/api/routers/paper.py",
        )
    )
    forbidden = ["place_order", "modify_order", "cancel_order", "dhanhq", "access-token"]
    assert not [token for token in forbidden if token in source]
