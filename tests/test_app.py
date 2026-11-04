"""Smoke tests of the Streamlit app itself, run headless with AppTest.

These catch widgets built against an API that a newer Streamlit removed, and
anything in the page code that only fails when the page is actually drawn.
AppTest cannot upload files, so st.file_uploader is replaced by a stub that
hands back a synthetic figure.

Every test here also fails if Streamlit reports a deprecation while the app
runs. A deprecated argument works for months and then disappears in a
release, breaking the app; this turns the warning into a failing test as soon
as CI installs the Streamlit release that deprecates it.
"""
import io
import logging

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from conftest import ROOT

APP = str(ROOT / "grating_streamlit.py")
PAGES = ["Output Angle", "Sampling Sweep", "Spectrometer Analysis", "Spectral Units"]


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages: list = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture(autouse=True)
def no_streamlit_deprecations():
    """Fail on any Streamlit deprecation notice.

    Streamlit sends all of them through deprecation_util.show_deprecation_warning,
    which always logs them to this logger (and, depending on config, may also
    show them in the page). Some are shown once per process, so that memory is
    cleared first, or a notice seen by an earlier test would be missed here.
    """
    from streamlit import deprecation_util

    getattr(deprecation_util, "_shown_warnings", set()).clear()
    logger = logging.getLogger("streamlit.deprecation_util")
    handler = _Collect()
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
    assert not handler.messages, (
        "Streamlit deprecation(s) reported while running the app:\n- "
        + "\n- ".join(dict.fromkeys(handler.messages))
    )


class _Upload(io.BytesIO):
    """Stands in for Streamlit's UploadedFile."""

    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name
        self.size = len(data)


@pytest.fixture
def upload(monkeypatch, sensor_pickle):
    """Make every st.file_uploader return the synthetic figure while
    upload['on'] is True, and nothing (an empty uploader) otherwise."""
    state = {"on": True}
    data = sensor_pickle.read_bytes()
    monkeypatch.setattr(
        st, "file_uploader",
        lambda *a, **k: _Upload(data, sensor_pickle.name) if state["on"] else None,
    )
    return state


@pytest.fixture
def app(monkeypatch):
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _go(at, page):
    at.sidebar.radio(key="page").set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(app, page):
    _go(app, page)


def test_spectrometer_analysis_runs(upload, app):
    _go(app, "Spectrometer Analysis")
    run = next(b for b in app.sidebar.button if b.label == "Run Analysis")
    assert not run.disabled
    run.click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert "fa_error" not in app.session_state
    assert set(app.session_state["fa_figs_png"]) == set("ABCDEF")
    assert len(app.session_state["fa_data"]["inputs"]["detected_pixels"]) >= 2


def test_settings_survive_switching_pages(upload, app):
    def ni(key):  # look the widget up in the latest run, never a stale tree
        return app.sidebar.number_input(key=key)

    ni("lines_per_mm_oa").set_value(1200)
    ni("order_oa").set_value(2)
    app.toggle(key="mirror_diagram_oa").set_value(True)
    app.run()

    _go(app, "Spectrometer Analysis")
    ni("tolerance_px_fa").set_value(3.5)
    app.sidebar.text_input(key="ref_pixels_str").set_value("16, 34")
    app.run()

    _go(app, "Output Angle")
    app.run()  # a full run on page 1, so page 3's widgets are really dropped
    assert ni("lines_per_mm_oa").value == 1200
    assert ni("order_oa").value == 2
    assert app.toggle(key="mirror_diagram_oa").value is True

    upload["on"] = False  # the uploader comes back empty, as in a browser
    _go(app, "Spectrometer Analysis")
    app.run()
    assert ni("tolerance_px_fa").value == 3.5
    assert app.sidebar.text_input(key="ref_pixels_str").value == "16, 34"
    assert any("kept from earlier" in c.value for c in app.sidebar.caption)
    run = next(b for b in app.sidebar.button if b.label == "Run Analysis")
    assert not run.disabled
