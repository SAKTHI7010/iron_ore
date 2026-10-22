import sys
from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

APP = Path(__file__).resolve().parents[2] / "frontend" / "app.py"
PAGES = ["Overview & EDA", "Single Sample Prediction", "Model Performance", "Batch Upload", "Digital Twin Map"]


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_exceptions(page):
    at = streamlit_testing.AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    at.sidebar.radio(key="nav").set_value(page).run()
    assert not at.exception, f"{page}: {at.exception}"


def test_single_prediction_flow():
    at = streamlit_testing.AppTest.from_file(str(APP), default_timeout=120).run()
    at.sidebar.radio(key="nav").set_value("Single Sample Prediction").run()
    at.button(key="btn_predict").click().run()
    assert not at.exception, at.exception
    assert any("Predicted Fe grade" in m.value for m in at.markdown)
