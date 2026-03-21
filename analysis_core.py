"""
analysis_core.py — Neon calibration and dispersion analysis for GREG.

Authors
-------
Sarlote Simsone — original implementation:
    peak matching, dispersion analysis, Gaussian fitting, all six plot functions.
Sergei Kulkov (rngKomorebi) — integration, UI wiring, dark-theme styling.

Public API
----------
Calibration
    check_linearity(ref_wavelengths_nm, ref_pixels) → (a, b)
    match_peaks_to_neon(detected_pixels, ref_wavelengths_nm, ref_pixels, ...) → dict

Dispersion
    dispersion_neighbours(wavelengths_nm, peak_pixels) → (table, xmid, scales, mean)
    dispersion_all_pairs(wavelengths_nm, peak_pixels)  → (table, i_idx, j_idx,
                                                           xmid, dist_px, scales, mean)

Six plot functions — each returns a matplotlib Figure:
    plot_neighbour_scales(pkl_file, wavelengths_nm, peak_pixels, ...)   # A
    plot_allpair_scales(wavelengths_nm, peak_pixels, ...)               # B
    plot_scale_vs_separation(wavelengths_nm, peak_pixels, ...)          # C
    plot_ideal_spectrum(pkl_file, ideal_pixels, intensities, ...)       # D
    plot_linearity(pixels, wavelengths_nm, ...)                       # E
    plot_gaussian_fit(x, y, popt, ...)                                  # F

Entry point
    run_neon_analysis(pkl_file, detected_pixels, ...) → dict

Figures are never shown (no plt.show()) and global rcParams are never mutated.
"""

from __future__ import annotations

import pickle
import re
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.optimize import curve_fit

# ── Plot colour palette (matches GREG dark theme) ─────────────────────────────
_C_FIG_BG = "#0e1117"
_C_AX_BG = "#0e1117"
_C_TEXT = "#e5e2e1"
_C_MUTED = "#bbcac1"
_C_ACCENT = "#42e5b0"
_C_YELLOW = "#FFD10F"
_C_ORANGE = "#FF8C42"
_C_GRID = "#3c4a43"

# ─────────────────────────────────────────────────────────────────────────────
# Neon reference spectrum  {wavelength_nm: relative_intensity}
# ─────────────────────────────────────────────────────────────────────────────
NEON_REFERENCE_SPECTRUM: Dict[float, float] = {
    600.0963: 1000,
    603.000: 10000,
    607.434: 10000,
    614.306: 10000,
    616.359: 10000,
    618.215: 1500,
    620.578: 1000,
    626.650: 10000,
    630.479: 1000,
    633.443: 10000,
    638.299: 10000,
    640.225: 20000,
    650.653: 15000,
    653.288: 1000,
    659.895: 10000,
    665.209: 1500,
    667.828: 5000,
    671.704: 700,
    692.947: 100000,
    702.405: 34000,
    703.241: 85000,
}

# Hardcoded reference wavelengths that anchor the two-point pixel→wavelength
# display conversion used when plot functions receive ref_pixels.
_REF_WL1, _REF_WL2 = 638.299, 640.2248


# ─────────────────────────────────────────────────────────────────────────────
# Signal processing
# ─────────────────────────────────────────────────────────────────────────────
def _gaussian_model(x, *p):
    """N Gaussians + constant background.  p = [A1, mu1, s1, …, c]."""
    c = p[-1]
    y = np.full_like(x, c, dtype=float)
    for i in range(0, len(p) - 1, 3):
        A, mu, s = p[i], p[i + 1], p[i + 2]
        y += A * np.exp(-0.5 * ((x - mu) / s) ** 2)
    return y


def fit_gaussians(
    x,
    y,
    peak_pixels: List[float],
    max_sigma_px: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit N Gaussians + constant background to (x, y), anchored at peak_pixels.

    Returns
    -------
    sigmas : 1-D array of per-peak σ values in pixels
    popt   : full parameter vector [A1, mu1, s1, …, c]
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    min_A = 20.0
    c0 = float(np.median(y))

    p0, lower, upper = [], [], []
    for peak in peak_pixels:
        A0 = max(float(y[np.argmin(np.abs(x - peak))] - c0), min_A)
        s0 = float((x.max() - x.min()) / 50)
        p0 += [A0, float(peak), s0]
        lower += [0.0, x.min(), 0.1]
        upper += [np.inf, x.max(), max_sigma_px]
    p0 += [c0]
    lower += [-np.inf]
    upper += [np.inf]

    popt, _ = curve_fit(
        _gaussian_model, x, y, p0=p0, bounds=(lower, upper), maxfev=20000
    )
    sigmas = np.array([popt[i + 2] for i in range(0, len(popt) - 1, 3)])
    return sigmas, popt


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────
def check_linearity(ref_wavelengths_nm, ref_pixels) -> Tuple[float, float]:
    """Fit wavelength = a·pixel + b from ≥2 reference pairs.

    Returns
    -------
    (a, b) : slope (nm/pixel) and intercept (nm)
    """
    wl = np.asarray(ref_wavelengths_nm, dtype=float)
    px = np.asarray(ref_pixels, dtype=float)
    if len(wl) < 2:
        raise ValueError("At least two reference points are required.")
    if wl.size != px.size:
        raise ValueError(
            "ref_wavelengths_nm and ref_pixels must be the same length."
        )
    a, b = np.polyfit(px, wl, 1)
    return float(a), float(b)


def match_peaks_to_neon(
    detected_pixels,
    ref_wavelengths_nm,
    ref_pixels,
    reference_spectrum: Optional[Dict[float, float]] = None,
    tolerance_px: float = 5.0,
    n_iterations: int = 3,
) -> Tuple[dict, list, dict, float, float]:
    """Match detected pixel positions to known Neon emission lines.

    Starts from a two-point seed calibration, then iteratively refines it
    using the growing set of matched peaks.

    Parameters
    ----------
    detected_pixels    : pixel positions of all visible peaks
    ref_wavelengths_nm : two wavelengths whose pixel positions are known
    ref_pixels         : pixel positions matching ref_wavelengths_nm (same order)
    reference_spectrum : {wavelength_nm: intensity}; defaults to NEON_REFERENCE_SPECTRUM
    tolerance_px       : maximum pixel residual to accept a peak–line match
    n_iterations       : calibration refinement iterations

    Returns
    -------
    matched          : {pixel: {wavelength_nm, intensity, pred_pixel, delta_px}}
    unmatched_pixels : detected pixels with no Neon match
    unmatched_lines  : Neon lines not matched to any detected peak
    a, b             : final calibration coefficients (wavelength = a·pixel + b)
    """
    if reference_spectrum is None:
        reference_spectrum = NEON_REFERENCE_SPECTRUM

    a, b = check_linearity(ref_wavelengths_nm, ref_pixels)
    matched: dict = {}
    used_wavelengths: set = set()

    for _ in range(n_iterations):
        predicted = {wl: (wl - b) / a for wl in reference_spectrum}
        detected = sorted(detected_pixels)
        matched = {}
        used_wavelengths = set()

        for px in detected:
            best_wl, best_dist, best_pred = None, float("inf"), None
            for wl, pred_px in predicted.items():
                if wl in used_wavelengths:
                    continue
                dist = abs(px - pred_px)
                if dist < best_dist and dist <= tolerance_px:
                    best_dist, best_wl, best_pred = dist, wl, pred_px
            if best_wl is not None:
                matched[float(px)] = {
                    "wavelength_nm": best_wl,
                    "intensity": reference_spectrum[best_wl],
                    "pred_pixel": best_pred,
                    "delta_px": float(px) - best_pred,
                }
                used_wavelengths.add(best_wl)

        if len(matched) >= 2:
            a, b = check_linearity(
                [v["wavelength_nm"] for v in matched.values()],
                list(matched.keys()),
            )

    unmatched_pixels = [
        px for px in detected_pixels if float(px) not in matched
    ]
    unmatched_lines = {
        wl: reference_spectrum[wl]
        for wl in reference_spectrum
        if wl not in used_wavelengths
    }
    return matched, unmatched_pixels, unmatched_lines, a, b


# ─────────────────────────────────────────────────────────────────────────────
# Dispersion analysis
# ─────────────────────────────────────────────────────────────────────────────
def _sort_by_wavelength(
    wavelengths_nm, peak_pixels
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (wavelengths, pixels) both sorted by ascending wavelength."""
    wl = np.asarray(wavelengths_nm, dtype=float)
    px = np.asarray(peak_pixels, dtype=float)
    if wl.size != px.size:
        raise ValueError(
            "wavelengths_nm and peak_pixels must be the same length."
        )
    order = np.argsort(wl)
    return wl[order], px[order]


def dispersion_neighbours(
    wavelengths_nm, peak_pixels
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute nm/pixel scale between every consecutive (neighbouring) peak pair.

    Returns
    -------
    table      : array (N-1, 8) — [i, j, px_i, px_j, wl_i, wl_j, scale, xmid]
    xmid       : midpoint pixel of each pair
    scales     : nm/pixel for each pair
    mean_scale : mean over all pairs
    """
    wl, px = _sort_by_wavelength(wavelengths_nm, peak_pixels)
    rows = []
    for i in range(len(px) - 1):
        scale = abs((wl[i + 1] - wl[i]) / (px[i + 1] - px[i]))
        xmid = 0.5 * (px[i] + px[i + 1])
        rows.append(
            (i + 1, i + 2, px[i], px[i + 1], wl[i], wl[i + 1], scale, xmid)
        )
    table = np.array(rows, dtype=float)
    scales = table[:, 6]
    xmid = table[:, 7]
    mean_scale = float(np.mean(scales)) if len(scales) else float("nan")
    return table, xmid, scales, mean_scale


def dispersion_all_pairs(wavelengths_nm, peak_pixels) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """Compute nm/pixel scale for every combination of peak pairs.

    Returns
    -------
    table      : array (M, 9) — [i, j, px_i, px_j, wl_i, wl_j, scale, xmid, dist_px]
    i_idx      : first-peak index for each row
    j_idx      : second-peak index for each row
    xmid       : midpoint pixel of each pair
    dist_px    : inter-peak distance in pixels
    scales     : nm/pixel for each pair
    mean_scale : mean over all pairs
    """
    wl, px = _sort_by_wavelength(wavelengths_nm, peak_pixels)
    rows = []
    for i, j in combinations(range(len(px)), 2):
        scale = abs((wl[j] - wl[i]) / (px[j] - px[i]))
        xmid = 0.5 * (px[i] + px[j])
        rows.append(
            (i, j, px[i], px[j], wl[i], wl[j], scale, xmid, abs(px[j] - px[i]))
        )
    table = np.array(rows, dtype=float)
    i_idx = table[:, 0].astype(int)
    j_idx = table[:, 1].astype(int)
    scales = table[:, 6]
    xmid = table[:, 7]
    dist_px = table[:, 8]
    mean_scale = float(np.mean(scales)) if len(scales) else float("nan")
    return table, i_idx, j_idx, xmid, dist_px, scales, mean_scale


def _ideal_pixels(
    wavelengths_nm,
    peak_pixels,
    mean_scale: Optional[float] = None,
    round_to_int: bool = True,
) -> List[float]:
    """Predict where peaks would fall under perfectly uniform dispersion.

    Anchors at the lowest-wavelength observed pixel and steps forward using
    mean_scale. If mean_scale is None it is computed from neighbours.
    """
    if mean_scale is None:
        mean_scale = dispersion_neighbours(wavelengths_nm, peak_pixels)[-1]
    wl, px = _sort_by_wavelength(wavelengths_nm, peak_pixels)
    sign = -1.0 if px[-1] < px[0] else 1.0
    signed_scale = sign * abs(mean_scale)

    ideal = np.empty_like(wl, dtype=float)
    ideal[0] = float(px[0])
    for i in range(1, len(wl)):
        ideal[i] = ideal[i - 1] + (wl[i] - wl[i - 1]) / signed_scale
    if round_to_int:
        ideal = np.round(ideal, 0)

    # Return in the original (unsorted) input order
    wl_in = np.asarray(wavelengths_nm, dtype=float)
    out = np.empty_like(wl_in, dtype=float)
    out[np.argsort(wl_in)] = ideal
    return out.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Figure utilities (private)
# ─────────────────────────────────────────────────────────────────────────────
def _load_pickle_fig(pkl_file) -> plt.Figure:
    """Return a matplotlib Figure from a file path or a BytesIO object."""
    if hasattr(pkl_file, "read"):
        pkl_file.seek(0)
        fig = pickle.load(pkl_file)
    else:
        with open(pkl_file, "rb") as fh:
            fig = pickle.load(fh)
    return fig


def extract_pixels_from_legend(pkl_file) -> List[float]:
    """Extract peak pixel positions from the legend labels of a pickled figure.

    Scans every legend across all axes and collects numeric tokens found in
    the label text. Only values in [0, 4096] are returned (covers any real
    detector width). Values are de-duplicated and returned in ascending order.
    """
    fig = _load_pickle_fig(pkl_file)
    pixels: list = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        for text_obj in leg.get_texts():
            candidates = [
                float(t)
                for t in re.findall(r"\b\d+(?:\.\d+)?\b", text_obj.get_text())
                if 10.0 <= float(t) <= 4096.0
            ]
            if candidates:
                pixels.append(max(candidates))
    plt.close(fig)
    return sorted(set(pixels))


def _isolate_sensor_axis(fig: plt.Figure) -> plt.Axes:
    """Keep only the data-richest axis in fig; remove all others.

    "Richest" is defined as the axis whose lines contain the most data points.
    Returns the kept axis.
    """
    best_ax = max(
        fig.axes,
        key=lambda ax: sum(
            len(ln.get_xdata())
            for ln in ax.lines
            if ln.get_xdata() is not None
        ),
    )
    for ax in list(fig.axes):
        if ax is not best_ax:
            fig.delaxes(ax)
    return best_ax


def _remap_to_wavelength(ax: plt.Axes, ref_pixels) -> None:
    """Convert every line's x-data from pixels to wavelengths (in-place).

    Uses the two-point linear calibration anchored at _REF_WL1 / _REF_WL2,
    with the pixel positions given by ref_pixels[0] and ref_pixels[1].
    """
    a = (_REF_WL2 - _REF_WL1) / (ref_pixels[1] - ref_pixels[0])
    b = _REF_WL1 - a * ref_pixels[0]
    for ln in ax.lines:
        px = np.asarray(ln.get_xdata(), float)
        ln.set_xdata(a * px + b)
    ax.relim()
    ax.autoscale_view()


def _px_to_wl(pixels, ref_pixels) -> np.ndarray:
    """Map pixel array to wavelengths using the two-point display calibration."""
    a = (_REF_WL2 - _REF_WL1) / (ref_pixels[1] - ref_pixels[0])
    b = _REF_WL1 - a * ref_pixels[0]
    return a * np.asarray(pixels, float) + b


def _normalize_lines(ax: plt.Axes) -> float:
    """Subtract median background and normalise all lines to their joint max."""
    all_y = [
        v
        for ln in ax.lines
        for v in np.asarray(ln.get_ydata(), float).tolist()
    ]
    background = float(np.nanmedian(all_y)) if all_y else 0.0
    y_max = max(
        (
            float(np.nanmax(np.asarray(ln.get_ydata(), float) - background))
            for ln in ax.lines
            if np.asarray(ln.get_ydata()).size
        ),
        default=0.0,
    )
    if y_max > 0:
        for ln in ax.lines:
            y = np.asarray(ln.get_ydata(), float)
            ln.set_ydata((y - background) / y_max)
    return y_max


def _apply_dark_style(fig: plt.Figure, *axes: plt.Axes) -> None:
    """Apply GREG dark-theme colours to a figure and its axes in-place."""
    fig.set_facecolor(_C_FIG_BG)
    for ax in axes:
        ax.set_facecolor(_C_AX_BG)
        ax.tick_params(colors=_C_TEXT, which="both")
        for spine in ax.spines.values():
            spine.set_color(_C_TEXT)
        ax.xaxis.label.set_color(_C_TEXT)
        ax.yaxis.label.set_color(_C_TEXT)
        ax.title.set_color(_C_TEXT)
        for txt in ax.texts:
            txt.set_color(_C_TEXT)
            bp = txt.get_bbox_patch()
            if bp is not None:
                bp.set_facecolor(_C_FIG_BG)
                bp.set_edgecolor(_C_MUTED)
        leg = ax.get_legend()
        if leg:
            leg.get_frame().set_facecolor(_C_FIG_BG)
            leg.get_frame().set_edgecolor(_C_MUTED)
            for text in leg.get_texts():
                text.set_color(_C_TEXT)


def _annotate_sigma_labels(
    ax: plt.Axes,
    popt,
    scale_nm_per_px: Optional[float] = None,
    max_rows: Optional[int] = None,
) -> None:
    """Place collision-avoiding σ annotation boxes above each Gaussian peak."""
    fig = ax.figure
    if not hasattr(fig.canvas, "get_renderer"):
        FigureCanvasAgg(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bb = ax.get_window_extent(renderer=renderer)

    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    xr = (xhi - xlo) or 1.0
    yr = (yhi - ylo) or 1.0
    n = (len(popt) - 1) // 3
    c = float(popt[-1])  # background offset
    if max_rows is None:
        max_rows = max(8, n * 2)

    peaks = sorted(
        (float(popt[3 * i + 1]), float(popt[3 * i + 2]), float(popt[3 * i]))
        for i in range(n)
    )
    fs = 0.55 * plt.rcParams["font.size"]
    dy, offset = 0.09, 0.04 * xr
    placed: list = []
    box_style = dict(
        boxstyle="round,pad=0.25", fc=_C_FIG_BG, ec=_C_MUTED, alpha=0.88
    )

    for mu, sigma, A in peaks:
        if not np.isfinite(mu):
            continue
        sigma_px = abs(sigma) if np.isfinite(sigma) else np.nan
        if scale_nm_per_px is not None and np.isfinite(sigma_px):
            label = rf"$\sigma={sigma_px * abs(scale_nm_per_px):.3f}\,\mathrm{{nm}}$"
        else:
            label = (
                rf"$\sigma={sigma_px:.3f}\,\mathrm{{px}}$"
                if np.isfinite(sigma_px)
                else r"$\sigma=?$"
            )
        x_frac = float(np.clip((mu + offset - xlo) / xr, 0.03, 0.97))
        # Start search just above the Gaussian peak; try higher rows on collision
        y0_peak = float(np.clip((A + c - ylo) / yr + 0.04, 0.05, 0.95))

        kept = False
        for row in range(max_rows):
            txt = ax.text(
                x_frac,
                y0_peak + row * dy,
                label,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=fs,
                bbox=box_style,
                zorder=20,
                clip_on=True,
                rotation=90,
            )
            fig.canvas.draw()
            bb = txt.get_window_extent(renderer=renderer).expanded(1.02, 1.15)
            inside = (
                bb.x0 >= axes_bb.x0 + 4
                and bb.x1 <= axes_bb.x1 - 4
                and bb.y0 >= axes_bb.y0 + 4
                and bb.y1 <= axes_bb.y1 - 4
            )
            if inside and not any(bb.overlaps(p) for p in placed):
                placed.append(bb)
                kept = True
                break
            txt.remove()

        if not kept:
            txt = ax.text(
                x_frac,
                y0_peak + (max_rows - 1) * dy,
                label,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=fs,
                bbox=box_style,
                zorder=20,
                clip_on=True,
            )
            fig.canvas.draw()
            placed.append(
                txt.get_window_extent(renderer=renderer).expanded(1.02, 1.15)
            )


# ─────────────────────────────────────────────────────────────────────────────
# Plot A — neighbour nm/px scale overlaid on sensor population figure
# ─────────────────────────────────────────────────────────────────────────────
def plot_neighbour_scales(
    pkl_file,
    wavelengths_nm,
    peak_pixels,
    scale_ylim=None,
    ref_pixels=None,
) -> Tuple[plt.Figure, dict]:
    """Load the sensor-population figure and overlay nm/px scale between neighbours.

    The original pickled figure is preserved as-is (axes labels, x-axis units).
    A second y-axis on the right shows the nm/px scale scatter and mean line.

    Returns
    -------
    fig  : matplotlib Figure
    data : {"neighbour_table": ndarray, "mean_scale_neighbours": float}
    """
    fig = _load_pickle_fig(pkl_file)
    fig.set_size_inches(16, 10)
    with plt.rc_context({"font.size": 30}):
        ax_ph = _isolate_sensor_axis(fig)
        leg = ax_ph.get_legend()
        if leg:
            leg.remove()

        # Add scale axis on the right — leave the original axis completely untouched
        ax_scale = ax_ph.twinx()
        ax_scale.set_ylabel("Scale (nm/pixel)")

        table, xmid, scales, mean_scale = dispersion_neighbours(
            wavelengths_nm, peak_pixels
        )
        ax_scale.scatter(
            xmid,
            scales,
            marker="X",
            s=60,
            color=_C_ACCENT,
            alpha=0.85,
            zorder=5,
        )
        ax_scale.axhline(
            mean_scale,
            color=_C_YELLOW,
            linestyle="--",
            linewidth=2,
            label=f"Mean {mean_scale:.3f} nm/px",
        )
        if scale_ylim is not None:
            ax_scale.set_ylim(*scale_ylim)
        ax_scale.set_title("Neighbour nm/px scale (midpoint between peaks)")
        ax_scale.legend(loc="upper right", frameon=True)
        _apply_dark_style(fig, ax_ph, ax_scale)
        fig.tight_layout()
    return fig, {"neighbour_table": table, "mean_scale_neighbours": mean_scale}


# ─────────────────────────────────────────────────────────────────────────────
# Plot B — nm/px scale for all peak pairs, coloured by first-peak index
# ─────────────────────────────────────────────────────────────────────────────
def plot_allpair_scales(
    wavelengths_nm,
    peak_pixels,
    scale_ylim=None,
) -> Tuple[plt.Figure, dict]:
    """Scatter nm/px for every peak combination, grouped by first peak (colour-coded).

    Returns
    -------
    fig  : matplotlib Figure
    data : {"all_pairs_table": ndarray, "mean_scale_all_pairs": float}
    """
    table, i_idx, _, xmid, _, scales, mean_scale = dispersion_all_pairs(
        wavelengths_nm, peak_pixels
    )
    n_groups = len(np.asarray(peak_pixels)) - 1
    colors = [
        plt.get_cmap("cool")(k)
        for k in np.linspace(0.1, 0.9, max(n_groups, 1))
    ]

    with plt.rc_context({"font.size": 30}):
        fig, ax = plt.subplots(figsize=(16, 10), constrained_layout=False)
        for i in range(n_groups):
            mask = i_idx == i
            ax.scatter(
                xmid[mask],
                scales[mask],
                marker="X",
                s=60,
                color=colors[i],
                alpha=0.85,
                label=f"Peak {i + 1}",
            )
        ax.axhline(
            mean_scale,
            color=_C_YELLOW,
            linestyle="--",
            linewidth=2,
            label=f"Mean {mean_scale:.3f} nm/px",
        )
        ax.set_xlabel("Pixel number (-)  [pair midpoint]")
        ax.set_ylabel("Scale (nm/pixel)")
        ax.set_title("nm/px scale — coloured by first peak in pair")
        if scale_ylim is not None:
            ax.set_ylim(*scale_ylim)
        ax.legend(loc="upper right", frameon=True)
        _apply_dark_style(fig, ax)
        fig.tight_layout()
    return fig, {"all_pairs_table": table, "mean_scale_all_pairs": mean_scale}


# ─────────────────────────────────────────────────────────────────────────────
# Plot C — nm/px scale vs inter-peak pixel separation
# ─────────────────────────────────────────────────────────────────────────────
def plot_scale_vs_separation(
    wavelengths_nm,
    peak_pixels,
    scale_ylim=None,
) -> Tuple[plt.Figure, dict]:
    """Scatter nm/px vs pixel distance between peaks (uniformity check).

    Returns
    -------
    fig  : matplotlib Figure
    data : {"all_pairs_table": ndarray, "mean_scale_all_pairs": float}
    """
    table, _, _, _, dist_px, scales, mean_scale = dispersion_all_pairs(
        wavelengths_nm, peak_pixels
    )
    with plt.rc_context({"font.size": 30}):
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.scatter(
            dist_px, scales, marker="X", s=60, color=_C_ACCENT, alpha=0.85
        )
        ax.axhline(
            mean_scale,
            color=_C_YELLOW,
            linestyle="--",
            linewidth=2,
            label=f"Mean {mean_scale:.3f} nm/px",
        )
        ax.set_xlabel("Distance between peaks (pixels)")
        ax.set_ylabel("Scale (nm/pixel)")
        ax.set_title("nm/px scale as a function of distance between peaks")
        if scale_ylim is not None:
            ax.set_ylim(*scale_ylim)
        ax.legend(loc="upper right", frameon=True)
        _apply_dark_style(fig, ax)
        fig.tight_layout()
    return fig, {"all_pairs_table": table, "mean_scale_all_pairs": mean_scale}


# ─────────────────────────────────────────────────────────────────────────────
# Plot D — idealised Neon spectrum overlaid on sensor population figure
# ─────────────────────────────────────────────────────────────────────────────
def plot_ideal_spectrum(
    pkl_file,
    ideal_pixels,
    intensities,
    sigma_px: float = 2.5,
    window: int = 12,
    ideal_color: str = _C_ORANGE,
    sensor_range: Tuple[int, int] = (0, 255),
    ref_pixels=None,
) -> plt.Figure:
    """Overlay idealised Neon Gaussians on the normalised sensor-population figure.

    The measured data is median-subtracted and normalised to its joint peak.
    Ideal Gaussian amplitudes are proportional to Neon line intensities.

    Returns
    -------
    fig : matplotlib Figure
    """
    fig = _load_pickle_fig(pkl_file)
    fig.set_size_inches(16, 10)
    with plt.rc_context({"font.size": 30}):
        ax = _isolate_sensor_axis(fig)
        leg = ax.get_legend()
        if leg:
            leg.remove()

        # Re-label the original lines so they appear in the new combined legend
        for i, ln in enumerate(ax.lines):
            ln.set_label("Measured" if i == 0 else "_nolegend_")

        use_wl = ref_pixels is not None and len(ref_pixels) >= 2
        if use_wl:
            _remap_to_wavelength(ax, ref_pixels)
        _normalize_lines(ax)

        # Filter ideal peaks to sensor range and normalise intensities
        ideal_px = np.asarray(ideal_pixels, dtype=float)
        amps = np.asarray(intensities, dtype=float)
        in_range = (ideal_px >= sensor_range[0]) & (
            ideal_px <= sensor_range[1]
        )
        ideal_px, amps = ideal_px[in_range], amps[in_range]
        amps = amps / float(np.max(amps))
        order = np.argsort(ideal_px)
        ideal_px, amps = ideal_px[order], amps[order]

        for i, (x0, amp) in enumerate(zip(ideal_px, amps)):
            xs_px = np.arange(
                int(np.floor(x0 - window)),
                int(np.ceil(x0 + window)) + 1,
                dtype=float,
            )
            ys = amp * np.exp(-0.5 * ((xs_px - x0) / sigma_px) ** 2)
            xs_plot = _px_to_wl(xs_px, ref_pixels) if use_wl else xs_px
            ax.plot(
                xs_plot,
                ys,
                linewidth=3,
                color=ideal_color,
                label="NIST" if i == 0 else "_nolegend_",
            )

        ax.set_xlabel("Wavelength (nm)" if use_wl else "Pixel number (-)")
        ax.set_ylabel("Normalised intensity (-)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("NIST values vs measured data")
        ax.legend(loc="upper right", frameon=True)
        _apply_dark_style(fig, ax)
        fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plot E — linear wavelength calibration fit + residuals
# ─────────────────────────────────────────────────────────────────────────────
def plot_linearity(
    pixels,
    wavelengths_nm,
) -> Tuple[plt.Figure, dict]:
    """Linear fit of wavelength vs pixel with a residuals sub-panel.

    Returns
    -------
    fig  : matplotlib Figure
    data : {"a_nm_per_pixel": float, "b_nm": float, "delta_lambda_nm": ndarray}
    """
    px = np.asarray(pixels, dtype=float)
    wl = np.asarray(wavelengths_nm, dtype=float)
    if px.size != wl.size:
        raise ValueError("pixels and wavelengths_nm must be the same length.")
    order = np.argsort(px)
    px, wl = px[order], wl[order]

    a, b = np.polyfit(px, wl, 1)
    residuals = wl - (a * px + b)
    px_dense = np.linspace(px.min(), px.max(), 500)

    with plt.rc_context({"font.size": 30}):
        fig, (ax_top, ax_bot) = plt.subplots(
            2,
            1,
            figsize=(16, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        ax_top.scatter(px, wl, s=60, marker="X", color=_C_ACCENT, label="Data")
        ax_top.plot(
            px_dense,
            a * px_dense + b,
            color=_C_YELLOW,
            linewidth=2,
            label=f"Fit  a={a:.3f} nm/px",
        )
        ax_top.set_ylabel("Wavelength (nm)")
        ax_top.legend(loc="best", frameon=True)
        ax_top.set_title("Linearity", pad=28)

        ax_bot.axhline(0.0, color=_C_MUTED, linewidth=1)
        ax_bot.scatter(px, residuals, s=60, marker="X", color=_C_ACCENT)
        ax_bot.set_xlabel("Pixel number (-)")
        ax_bot.set_ylabel(r"$\Delta\lambda$ (nm)")
        ax_bot.set_title("Residuals from linear fit")
        _apply_dark_style(fig, ax_top, ax_bot)
        fig.tight_layout()
    return fig, {
        "a_nm_per_pixel": float(a),
        "b_nm": float(b),
        "delta_lambda_nm": residuals,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot F — multi-Gaussian fit with σ annotations
# ─────────────────────────────────────────────────────────────────────────────
def plot_gaussian_fit(
    x,
    y,
    popt,
    scale_nm_per_px: Optional[float] = None,
    ref_pixels=None,
) -> plt.Figure:
    """Plot raw sensor data against the multi-Gaussian fit, annotating each peak's σ.

    Returns
    -------
    fig : matplotlib Figure
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    use_wl = ref_pixels is not None and len(ref_pixels) >= 2

    with plt.rc_context({"font.size": 30}):
        fig, ax = plt.subplots(figsize=(16, 10), constrained_layout=False)
        x_plot = _px_to_wl(x, ref_pixels) if use_wl else x
        xlabel = "Wavelength (nm)" if use_wl else "Pixel number (-)"

        y_fit = _gaussian_model(x, *popt)
        ax.plot(
            x_plot,
            y,
            marker="X",
            linewidth=1.5,
            markersize=3,
            color=_C_ACCENT,
            label="data",
        )
        ax.plot(x_plot, y_fit, color=_C_YELLOW, linewidth=2.5, label="fit")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counts (-)")
        ax.set_title("Gaussian fit of measured peaks")
        ax.legend(loc="upper right", frameon=True)

        # Remap popt peak centres to wavelength space before annotating
        popt_ann = list(popt)
        if use_wl:
            for i in range(0, len(popt) - 1, 3):
                popt_ann[i + 1] = float(
                    _px_to_wl([popt[i + 1]], ref_pixels)[0]
                )
        _annotate_sigma_labels(ax, popt_ann, scale_nm_per_px=scale_nm_per_px)

        ax.set_ylim(0, max(float(np.max(y)), float(np.max(y_fit))) * 1.1)
        _apply_dark_style(fig, ax)
        fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def run_neon_analysis(
    pkl_file,
    detected_pixels: List[float],
    ref_wavelengths_nm: List[float],
    ref_pixels: List[float],
    reference_spectrum: Optional[Dict[float, float]] = None,
    tolerance_px: float = 5.0,
    ideal_pixels: Optional[List[float]] = None,
    scale_ylim=None,
    window: int = 12,
    ideal_color: str = _C_ORANGE,
    full_sensor: bool = False,
) -> dict:
    """Run the full Neon calibration pipeline and produce all six analysis figures.

    Parameters
    ----------
    pkl_file           : file path (str) or BytesIO — pickled sensor-population figure
    detected_pixels    : pixel positions of every visible peak
    ref_wavelengths_nm : two known Neon wavelengths (seed calibration)
    ref_pixels         : pixel positions of those two wavelengths (same order)
    reference_spectrum : {wavelength_nm: intensity}; defaults to NEON_REFERENCE_SPECTRUM
    tolerance_px       : max pixel residual to match a peak to a Neon line
    ideal_pixels       : override auto-computed ideal pixel positions
    scale_ylim         : (ymin, ymax) for the scale axis in plots A / B / C
    window             : half-width in pixels of each ideal Gaussian patch (plot D)
    ideal_color        : hex colour for ideal spectrum lines (plot D)
    full_sensor        : True → sensor range 0–511 px, False → 0–255 px

    Returns
    -------
    dict with keys
        matched          : {pixel: {wavelength_nm, intensity, pred_pixel, delta_px}}
        unmatched_pixels : list of unmatched detected pixels
        unmatched_lines  : {wavelength_nm: intensity} for unmatched Neon lines
        calibration      : {a_nm_per_pixel, b_nm}
        ideal_pixels_used: list
        gauss_fit        : {sigmas_px, popt}
        figs             : {A, B, C, D, E, F}  — matplotlib Figure objects
        stats            : {mean_scale_neighbours, mean_scale_all_pairs,
                            a_nm_per_pixel, b_nm}
    """
    if reference_spectrum is None:
        reference_spectrum = NEON_REFERENCE_SPECTRUM
    sensor_range = (0, 511) if full_sensor else (0, 255)

    # ── 1. Match peaks to Neon lines ─────────────────────────────────────────
    matched, unmatched_pixels, unmatched_lines, a, b = match_peaks_to_neon(
        detected_pixels=detected_pixels,
        ref_wavelengths_nm=ref_wavelengths_nm,
        ref_pixels=ref_pixels,
        reference_spectrum=reference_spectrum,
        tolerance_px=tolerance_px,
    )

    wavelengths_nm = [v["wavelength_nm"] for v in matched.values()]
    peak_pixels = list(matched.keys())
    intensities = [v["intensity"] for v in matched.values()]

    if len(peak_pixels) == 0:
        raise ValueError(
            "No peaks matched any Neon line. "
            "Check detected_pixels / ref_pixels or increase tolerance_px."
        )
    if len(peak_pixels) < 2:
        raise ValueError(
            f"Only {len(peak_pixels)} peak matched — at least 2 are needed "
            "for scale analysis."
        )

    # ── 2. Compute ideal pixel positions ─────────────────────────────────────
    if ideal_pixels is None:
        ideal_pixels = _ideal_pixels(
            wavelengths_nm,
            peak_pixels,
            mean_scale=dispersion_neighbours(wavelengths_nm, peak_pixels)[-1],
        )

    # ── 3. Fit Gaussians to the raw sensor data ───────────────────────────────
    raw_fig = _load_pickle_fig(pkl_file)
    raw_ax = max(
        raw_fig.axes,
        key=lambda ax: sum(
            len(ln.get_xdata())
            for ln in ax.lines
            if ln.get_xdata() is not None
        ),
    )
    ln = max(
        raw_ax.lines,
        key=lambda l: 0 if l.get_xdata() is None else len(l.get_xdata()),
    )
    x_raw = np.asarray(ln.get_xdata(), float)
    y_raw = np.asarray(ln.get_ydata(), float)
    plt.close(raw_fig)

    sigmas_px, popt = fit_gaussians(x_raw, y_raw, peak_pixels)

    # ── 4. Six plots ──────────────────────────────────────────────────────────
    _, _, _, _mean_scale = dispersion_neighbours(wavelengths_nm, peak_pixels)
    sigma_px = _mean_scale / 3.0
    fig_A, data_A = plot_neighbour_scales(
        pkl_file,
        wavelengths_nm,
        peak_pixels,
        scale_ylim=scale_ylim,
        ref_pixels=ref_pixels,
    )
    fig_B, data_B = plot_allpair_scales(
        wavelengths_nm,
        peak_pixels,
        scale_ylim=scale_ylim,
    )
    fig_C, data_C = plot_scale_vs_separation(
        wavelengths_nm,
        peak_pixels,
        scale_ylim=scale_ylim,
    )
    fig_D = plot_ideal_spectrum(
        pkl_file,
        ideal_pixels=ideal_pixels,
        intensities=intensities,
        sigma_px=sigma_px,
        window=window,
        ideal_color=ideal_color,
        sensor_range=sensor_range,
        ref_pixels=ref_pixels,
    )
    fig_E, data_E = plot_linearity(
        pixels=peak_pixels,
        wavelengths_nm=wavelengths_nm,
    )
    fig_F = plot_gaussian_fit(
        x_raw,
        y_raw,
        popt,
        scale_nm_per_px=a,
        ref_pixels=ref_pixels,
    )

    return {
        "matched": matched,
        "unmatched_pixels": unmatched_pixels,
        "unmatched_lines": unmatched_lines,
        "calibration": {"a_nm_per_pixel": a, "b_nm": b},
        "ideal_pixels_used": ideal_pixels,
        "gauss_fit": {
            "sigmas_px": sigmas_px.tolist(),
            "popt": popt.tolist(),
        },
        "figs": {
            "A": fig_A,
            "B": fig_B,
            "C": fig_C,
            "D": fig_D,
            "E": fig_E,
            "F": fig_F,
        },
        "stats": {
            "mean_scale_neighbours": data_A["mean_scale_neighbours"],
            "mean_scale_all_pairs": data_B["mean_scale_all_pairs"],
            "a_nm_per_pixel": data_E["a_nm_per_pixel"],
            "b_nm": data_E["b_nm"],
        },
    }
