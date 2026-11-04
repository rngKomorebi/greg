"""Every module imports: catches syntax a newer Python rejects, removed stdlib
modules, and dependency APIs removed at import time."""
import importlib

import pytest


@pytest.mark.parametrize(
    "module", ["analysis_core", "physics_core", "spectral_units", "export_core"]
)
def test_module_imports(module):
    importlib.import_module(module)


def test_neon_reference_loads():
    import analysis_core as ac
    import spectral_units as su

    assert len(ac.NEON_REFERENCE_SPECTRUM) > 50
    assert all(400.0 <= wl <= 900.0 for wl in ac.NEON_REFERENCE_SPECTRUM)
    assert su.neon_lines()
