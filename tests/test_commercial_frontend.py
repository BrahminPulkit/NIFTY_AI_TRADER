from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = ROOT / "apps/research_ui"


def test_primary_commercial_pages_render_without_exceptions():
    pages = (
        "app.py", "pages/3_Live_Market.py",
        "pages/4_Options_Analytics.py", "pages/12_Paper_Trading.py",
    )
    for page in pages:
        app = AppTest.from_file(str(UI_ROOT / page))
        app.run(timeout=45)
        assert not app.exception


def test_trading_desk_uses_safe_commercial_language():
    source = (UI_ROOT / "app.py").read_text(encoding="utf-8")
    assert "Trading Desk" in source
    assert "WHY NO TRADE" in source
    assert "safe_action(snapshot)" in source
    assert '"Buy Call"' not in source
    assert "No entry, stop or target is estimated" in source


def test_commercial_shell_has_responsive_and_chain_contracts():
    css = (UI_ROOT / "assets/institutional.css").read_text(encoding="utf-8")
    for selector in (
        ".trade-command", ".decision-panel", ".workflow-rail",
        ".chart-workspace", ".chain-scroll", ".paper-flow",
    ):
        assert selector in css
    assert "@media(max-width:980px)" in css
    assert "@media(max-width:720px)" in css


def test_sidebar_uses_session_preserving_streamlit_navigation():
    source = (UI_ROOT / "components/theme.py").read_text(encoding="utf-8")
    assert "st.switch_page(path)" in source
    assert "f'<a class=\"nav-link" not in source


def test_frontend_does_not_claim_unavailable_chain_values():
    source = (UI_ROOT / "pages/4_Options_Analytics.py").read_text(encoding="utf-8")
    assert "NO VALUES ARE ESTIMATED OR FABRICATED" in source
    assert "OI feed unavailable" in source
    assert "IV feed unavailable" in source
