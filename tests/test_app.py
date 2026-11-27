"""Dashboard tests with Streamlit's AppTest: every view renders, official figures are dated,
personal affordability works, missing data never becomes zero."""
import pytest

import config

pytest.importorskip("streamlit")
pytestmark = pytest.mark.skipif(not (config.MODELS_DIR / "rent_model_meta.json").exists(),
                                reason="model not built")

APP = str(config.ROOT / "app" / "streamlit_app.py")


def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(APP, default_timeout=120).run()


def _search(at, area, measure, income=None):
    at.sidebar.selectbox[0].set_value(area)
    at.sidebar.selectbox[1].set_value(measure)
    if income is not None:
        at.sidebar.number_input[0].set_value(income)
    return at.run()


def test_rent_check_shows_official_rent_with_date():
    at = _search(_app(), "Westminster", "Flat")
    assert not at.exception
    m = {x.label: x.value for x in at.metric}
    assert m["Official average rent"].startswith("£")
    assert any("Official ONS figure" in c.value for c in at.caption)


def test_bedroom_rent_is_the_official_figure():
    from src.models.area_data import AreaData
    d = AreaData()
    at = _search(_app(), "Leeds", "2 bed")
    assert not at.exception
    want = d.official_rent(d.code_for("Leeds"), "2 bed")
    assert {x.label: x.value for x in at.metric}["Official average rent"] == f"£{want:,}"


def test_personal_affordability_uses_my_income():
    at = _search(_app(), "Manchester", "2 bed", income=42000)
    assert not at.exception
    m = {x.label: x.value for x in at.metric}
    assert "Share of your income" in m
    assert any("of your gross income" in w.value for w in list(at.warning) + list(at.success))


def test_area_with_missing_crime_still_answers():
    at = _search(_app(), "Manchester", "Flat")
    assert not at.exception and at.metric
    assert any("no police data" in x.value for x in at.markdown)


def test_sources_panel_lists_no_data_areas():
    at = _app()
    assert any("Rents:" in m.value for m in at.sidebar.markdown)
    assert any("No police data" in c.value for c in at.sidebar.caption)


@pytest.mark.parametrize("view", ["Compare areas", "Area map", "Methodology & data", "System results"])
def test_other_views_render(view):
    at = _app()
    at.sidebar.radio[0].set_value(view).run()
    assert not at.exception


def test_compare_offers_csv_downloads():
    at = _app()
    at.sidebar.radio[0].set_value("Compare areas").run()
    assert not at.exception
    labels = [b.label for b in at.get("download_button")]
    assert "Download comparison (CSV)" in labels and "Download all areas (CSV)" in labels
