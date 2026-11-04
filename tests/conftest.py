"""Shared fixtures: synthetic sensor figures, pickled the way daplis saves them.

No real measurement is needed. A figure is built from a known calibration and
known Neon lines, so every test knows exactly what the analysis should find.
"""
import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # runners have no display

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent

# Synthetic spectrometer: wavelength = A_NM_PER_PX * pixel + B_NM.
# Anchored so the two default reference lines of the app (638.299 and
# 640.225 nm) fall inside the 256-px sensor.
A_NM_PER_PX = 0.11
B_NM = 638.299 - A_NM_PER_PX * 16.0
SIGMA_PX = 1.2
N_PIXELS = 256


def px_of(wl_nm: float) -> float:
    return (wl_nm - B_NM) / A_NM_PER_PX


def planted_lines(min_gap_px: float = 8.0, n_max: int = 7) -> dict:
    """Strong Neon lines on the sensor, spaced so none can be mistaken for
    another: {wavelength_nm: intensity}."""
    import analysis_core as ac

    on_sensor = {
        wl: i
        for wl, i in ac.NEON_REFERENCE_SPECTRUM.items()
        if 3.0 <= px_of(wl) <= N_PIXELS - 4.0
    }
    chosen: dict = {}
    for wl in sorted(on_sensor, key=on_sensor.get, reverse=True):
        if all(abs(px_of(wl) - px_of(c)) >= min_gap_px for c in chosen):
            chosen[wl] = on_sensor[wl]
        if len(chosen) == n_max:
            break
    return dict(sorted(chosen.items()))


def make_sensor_figure(lines: dict, sigma_px: float = SIGMA_PX,
                       n_pixels: int = N_PIXELS) -> Figure:
    """One trace of Gaussians plus a legend 'Peak at N, Rate: X kHz' per line,
    with proxy handles, as in the figures daplis saves."""
    x = np.arange(n_pixels, dtype=float)
    peak_max = max(lines.values())
    y = np.full_like(x, 0.15)
    for wl, inten in lines.items():
        y += 60.0 * inten / peak_max * np.exp(-0.5 * ((x - px_of(wl)) / sigma_px) ** 2)

    fig = Figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    ax.set_xlabel("Pixel number (-)")
    ax.set_ylabel("Photon rate (kHz)")
    labels = []
    for wl in lines:
        px = int(round(px_of(wl)))
        labels.append(f"Peak at {px}, Rate: {y[px]:.2f} kHz")
    ax.legend(handles=[Line2D([], []) for _ in labels], labels=labels)
    return fig


@pytest.fixture
def lines():
    return planted_lines()


@pytest.fixture
def sensor_pickle(tmp_path, lines) -> Path:
    path = tmp_path / "synthetic_rates.pickle"
    path.write_bytes(pickle.dumps(make_sensor_figure(lines)))
    return path
