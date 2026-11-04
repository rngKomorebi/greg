"""Spectrometer analysis on synthetic sensor figures with known answers."""
import time

import numpy as np
import pytest

import analysis_core as ac
from conftest import A_NM_PER_PX, B_NM, SIGMA_PX, px_of


# ── Legend parsing ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("label, pixel", [
    ("Peak at 19, Rate: 29.94 kHz", 19.0),   # rate larger than the pixel
    ("Peak at 371, Rate: 1.85 kHz", 371.0),
    ("peak: 42.5", 42.5),
    ("Ne line 640 nm, px 33", 640.0),        # no "peak": largest number
    ("no numbers here", None),
])
def test_legend_pixel(label, pixel):
    assert ac._legend_pixel(label) == pixel


def test_peaks_read_from_pickle(sensor_pickle, lines):
    rows = ac.extract_peaks_from_legend(sensor_pickle)
    assert [r["pixel"] for r in rows] == sorted(round(px_of(wl)) for wl in lines)
    assert all(r["signal"] > 1.0 for r in rows)  # on the peaks, not background
    assert ac.extract_pixels_from_legend(sensor_pickle) == [r["pixel"] for r in rows]


# ── Duplicate peaks ──────────────────────────────────────────────────────────
def test_close_peaks_keep_the_strongest():
    pixels = [19, 63, 81, 371, 373]
    signal = [30.0, 54.0, 67.0, 1.85, 7.7]
    assert ac.flag_close_peaks(pixels, signal, 2.0) == [None, None, None, 373.0, None]
    assert ac.flag_close_peaks(pixels, signal, 20.0) == [None, 81.0, None, 373.0, None]
    assert ac.flag_close_peaks(pixels, signal, 0.0) == [None] * 5


def test_close_peaks_chain_into_one_group():
    # 10-11-12 are one group even though 10 and 12 are 2 px apart
    assert ac.flag_close_peaks([10, 11, 12], [1.0, 5.0, 2.0], 1.0) == [11.0, None, 11.0]


def test_close_peaks_order_does_not_matter():
    a = ac.flag_close_peaks([373, 19, 371], [7.7, 30.0, 1.85], 2.0)
    assert a == [None, None, 373.0]


# ── Gaussian fit ─────────────────────────────────────────────────────────────
def _trace(peaks, n=512, noise=0.0):
    x = np.arange(n, dtype=float)
    y = np.full_like(x, 0.2)
    for amp, mu, sig in peaks:
        y += amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)
    if noise:
        y += np.random.default_rng(0).normal(0.0, noise, n)
    return x, y


@pytest.mark.parametrize("sigma", [0.7, 1.2, 4.0, 8.0])
def test_fit_recovers_sigma(sigma):
    peaks = [(50.0, 100.3, sigma), (30.0, 250.0, sigma), (80.0, 400.6, sigma)]
    x, y = _trace(peaks, noise=0.05)
    sigmas, popt, perr = ac.fit_gaussians(x, y, [round(p[1]) for p in peaks])
    assert sigmas == pytest.approx([sigma] * 3, rel=0.03)
    assert popt[1::3][:3] == pytest.approx([p[1] for p in peaks], abs=0.05)
    assert np.all(np.isfinite(perr))


def test_fit_of_many_narrow_lines_is_fast():
    """Regression: the fit used to start every σ at (trace length)/50, ~10 px,
    and needed >2 minutes (then failed) for nine sub-pixel-wide lines."""
    rng = np.random.default_rng(1)
    centres = [19, 63, 81, 177, 201, 263, 337, 373, 450]
    peaks = [(rng.uniform(5, 70), c + rng.uniform(-0.3, 0.3), rng.uniform(0.5, 0.9))
             for c in centres]
    x, y = _trace(peaks)
    t0 = time.perf_counter()
    sigmas, _, _ = ac.fit_gaussians(x, y, centres)
    assert time.perf_counter() - t0 < 10.0
    assert sigmas == pytest.approx([p[2] for p in peaks], rel=0.1)


# ── Full pipeline ────────────────────────────────────────────────────────────
def test_run_neon_analysis_end_to_end(sensor_pickle, lines):
    wls = list(lines)
    ref_wl = [wls[0], wls[-1]]
    ref_px = [px_of(w) for w in ref_wl]
    detected = ac.extract_pixels_from_legend(sensor_pickle)

    r = ac.run_neon_analysis(
        sensor_pickle, detected, ref_wl, ref_px,
        reference_spectrum=lines, tolerance_px=3.0,
    )

    assert sorted(v["wavelength_nm"] for v in r["matched"].values()) == wls
    assert r["unmatched_pixels"] == []
    assert r["calibration"]["a_nm_per_pixel"] == pytest.approx(A_NM_PER_PX, rel=0.01)
    # Legend pixels are rounded, so the intercept is good to about half a pixel
    assert r["calibration"]["b_nm"] == pytest.approx(B_NM, abs=0.5 * A_NM_PER_PX)
    assert r["gauss_fit"]["sigmas_px"] == pytest.approx([SIGMA_PX] * len(wls), rel=0.02)
    assert r["plot_errors"] == {}
    assert r["export_errors"] == {}
    assert set(r["figs"]) == set("ABCDEF")
    assert all(fig is not None for fig in r["figs"].values())
    assert r["tables"]


def test_run_with_full_reference_spectrum(sensor_pickle):
    """With the full NIST list (what the app uses) the run must still finish and
    draw every plot, whatever it matches."""
    detected = ac.extract_pixels_from_legend(sensor_pickle)
    r = ac.run_neon_analysis(
        sensor_pickle, detected, [638.299, 640.225], [16.0, px_of(640.225)],
    )
    assert len(r["matched"]) >= 2
    assert r["plot_errors"] == {}
