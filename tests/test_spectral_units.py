"""Unit conversions: round trips and fixed points."""
import numpy as np
import pytest

import spectral_units as su


@pytest.mark.parametrize("wl", np.linspace(250.0, 1000.0, 16))
def test_air_vacuum_round_trip(wl):
    assert su.air_to_vacuum(su.vacuum_to_air(wl)) == pytest.approx(wl, abs=1e-5)


def test_air_is_shorter_than_vacuum():
    # Visible light: n_air ≈ 1.00028, so λ_air is about 0.18 nm shorter at 650 nm
    assert 0.15 < 650.0 - su.vacuum_to_air(650.0) < 0.21


@pytest.mark.parametrize("wl", [400.0, 632.8, 1550.0])
def test_frequency_and_wavenumber_invert(wl):
    assert su.wl_from_freq(su.freq_thz(wl)) == pytest.approx(wl, rel=1e-14)
    assert su.wl_from_wnum(su.wnum_cm(wl)) == pytest.approx(wl, rel=1e-14)


def test_photon_energy_fixed_point():
    assert su.photon_energy_ev(su.HC_EV_NM) == pytest.approx(1.0)


@pytest.mark.parametrize("unit, value, key, expected", [
    ("nm", 0.05, "dl_nm", 0.05),
    ("pm", 50.0, "dl_nm", 0.05),
    ("GHz", 35.0, "dnu_ghz", 35.0),
    ("MHz", 500.0, "dnu_ghz", 0.5),
    ("cm⁻¹", 1.2, "dwnum_cm", 1.2),
])
def test_width_conversions_return_their_input(unit, value, key, expected):
    out = su.width_in(value, unit, 650.0)
    assert out[key] == pytest.approx(expected, rel=1e-12)
    # Every unit describes the same width
    assert out["dnu_ghz"] * 1e-3 == pytest.approx(
        su.C_NM_THZ * out["dl_nm"] / 650.0 ** 2, rel=1e-12
    )


def test_unknown_width_unit_raises():
    with pytest.raises(ValueError):
        su.width_in(1.0, "furlong", 650.0)
