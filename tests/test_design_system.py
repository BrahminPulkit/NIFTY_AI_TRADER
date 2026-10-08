from pathlib import Path

from streamlit.testing.v1 import AppTest

from apps.research_ui.components.theme import DEFAULT_PREFERENCES, THEMES, chart_palette


ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = ROOT / "apps/research_ui"


def test_required_theme_variants_and_exact_palette_tokens():
    assert set(THEMES) == {
        "Institutional Blue", "Emerald", "Royal Purple", "Graphite", "Crimson"}
    css = (UI_ROOT / "assets/institutional.css").read_text(encoding="utf-8")
    for token in (
        "#101827", "#141F33", "#1C2940", "#243552",
        "#4F8CFF", "#2ECC71", "#FF5A5F", "#F5B041",
        "#9B7BFF", "#F8FAFC", "#AAB7C8", "#6B7280",
    ):
        assert token in css
    assert "#000000" not in css


def test_design_preferences_have_safe_defaults():
    assert DEFAULT_PREFERENCES == {
        "theme": "Institutional Blue",
        "font_size": "Comfortable",
        "compact_mode": False,
        "animations": True,
        "chart_density": "Balanced",
        "sidebar_width": "Standard",
        "card_spacing": "Comfortable",
    }


def test_lucide_and_responsive_system_present():
    theme = (UI_ROOT / "components/theme.py").read_text(encoding="utf-8")
    css = (UI_ROOT / "assets/institutional.css").read_text(encoding="utf-8")
    assert 'class="lucide' in theme
    assert "@media(max-width:980px)" in css
    assert "@media(max-width:720px)" in css
    assert "[data-testid=stDataFrame] [role=row]:nth-child(even)" in css
    assert "ui_experience" in theme
    assert ".novice-strategy" in css


def test_chart_contract_uses_commercial_dark_template_and_series_colours():
    chart_source = (UI_ROOT / "components/charts.py").read_text(encoding="utf-8")
    assert 'template="plotly_dark"' in chart_source
    colors = chart_palette()
    assert colors["ema20"] == "#F5B041"
    assert colors["ema50"] == "#4F8CFF"
    assert colors["vwap"] == "#9B7BFF"
    assert colors["probability"] == "#9B7BFF"


def test_dashboard_and_settings_render():
    for page in ("app.py", "pages/9_Settings.py"):
        app = AppTest.from_file(str(UI_ROOT / page))
        app.run(timeout=45)
        assert not app.exception
