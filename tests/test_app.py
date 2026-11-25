"""Dashboard smoke tests with Streamlit's AppTest: every view renders without errors."""
import pytest

import config

pytest.importorskip("streamlit")
pytestmark = pytest.mark.skipif(not (config.MODELS_DIR / "rent_model_meta.json").exists(),
                                reason="model not built")

APP = str(config.ROOT / "app" / "streamlit_app.py")


def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(APP, default_timeout=120).run()


def test_estimator_shows_official_rent_with_date():
    at = _app()
    assert not at.exception
    at.sidebar.selectbox[0].set_value("Westminster")
    at.sidebar.selectbox[1].set_value("Flat")
    at.sidebar.button[0].click().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Official median rent" in labels
    assert any("Official ONS figure" in c.value for c in at.caption)


def test_area_with_missing_crime_still_answers():
    at = _app()
    at.sidebar.selectbox[0].set_value("Manchester")
    at.sidebar.selectbox[1].set_value("Flat")
    at.sidebar.button[0].click().run()
    assert not at.exception and at.metric


def test_sources_panel_lists_no_data_areas():
    at = _app()
    assert any("Rents:" in m.value for m in at.sidebar.markdown)
    assert any("No police data" in c.value for c in at.sidebar.caption)


@pytest.mark.parametrize("view", ["Area map", "System results"])
def test_other_views_render(view):
    at = _app()
    at.sidebar.radio[0].set_value(view).run()
    assert not at.exception
