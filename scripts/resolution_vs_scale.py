# Line width vs spectral scale: GREG's model, measured Ne lines, and a fit of α.
#
# Per line on the detector:  σ_px² = σ_optics,px(α)² + σ_pixel²
# σ_optics is the far field of the single-mode beam after the grating. The
# grating widens the beam by cos β / cos α, so the measured widths fix α when
# nothing else blurs the line. Compare the fitted α with the lab value: if the
# lab α is steeper, the difference is extra angular blur (also printed).
# physics_core is imported from the greg folder: run with greg as the root.

import math

import komorebi_mpl as kmpl
import numpy as np
import pandas as pd
import physics_core as pc
from matplotlib import pyplot as plt
from scipy.optimize import brentq

kmpl.use("sci_pure")

# Setup: Thorlabs PF2F63 collimator + SM600 fibre (P1-630A), 25 mm grating,
# ACT508 achromats, LinoSPAD2 with microlenses (pixel width = pitch)
pixel_pitch_um = 26.2
pixel_width_frac = 1.0
wavelength_nm = 640.0
order = 1
lens_axis = pc.LENS_AXIS_BEAM  # camera lens along the diffracted beam
beam_diameter_mm = pc.collimated_beam_diameter_mm(
    collimator_f_mm=10.99,
    mfd_um=4.3,  # SM600 spec 3.6–5.3 µm at 633 nm
    lambda_nm=wavelength_nm,
)
grating_width_mm = 25.0
# Lab α from the normal: grating turned 65° from normal incidence (2026-10-30;
# 25° to the surface). The 2400/200 scale (0.043 nm/px) gives 66.7°.
alpha_nominal_deg = 65.0


def budget(lines_per_mm, f_mm, alpha_deg, extra_sigma_urad=0.0):
    return pc.line_width_budget(
        p=pixel_pitch_um * 1e-6,
        lines_per_mm=lines_per_mm,
        alpha_deg=alpha_deg,
        lambda_nm=wavelength_nm,
        m=order,
        f=f_mm * 1e-3,
        beam_diameter_mm=beam_diameter_mm,
        pixel_width_frac=pixel_width_frac,
        extra_sigma_urad=extra_sigma_urad,
        grating_width_mm=grating_width_mm,
        lens_axis=lens_axis,
    )


# Measured line widths: sigma_px (+ err_px) or sigma_nm (+ err_frac), and the
# calibrated nm_per_px, or None to use the model scale at the lab α (hollow).
measurements = [
    # 2026-09-07, 2400 gr/mm, 400 mm, ~640 nm, full sensor: Gaussian fits to
    # the four peaks of 0000210400-0000210404_senpop.npy (same for 1–100
    # files). Scale from the 638.299 / 640.225 nm pair, 101 px apart.
    {
        "lines_per_mm": 2400,
        "f_mm": 400,
        "sigma_px": 0.86,
        "err_px": 0.02,
        "nm_per_px": 0.0191,
        "label": "2400/400 px 45",
    },
    {
        "lines_per_mm": 2400,
        "f_mm": 400,
        "sigma_px": 0.87,
        "err_px": 0.02,
        "nm_per_px": 0.0191,
        "label": "2400/400 px 146",
    },
    {
        "lines_per_mm": 2400,
        "f_mm": 400,
        "sigma_px": 0.60,
        "err_px": 0.06,
        "nm_per_px": 0.0191,
        "label": "2400/400 px 284",
    },
    {
        "lines_per_mm": 2400,
        "f_mm": 400,
        "sigma_px": 0.73,
        "err_px": 0.05,
        "nm_per_px": 0.0191,
        "label": "2400/400 px 386",
    },
    # 2026-10-30, α = 65°: σ = 17 pm (rms) at 0.043 nm/px, scale from the
    # 638.299 / 640.225 nm pair. σ ≈ 0.4 px, where a Gaussian fit to pixel
    # values is unreliable, hence the larger error.
    {
        "lines_per_mm": 2400,
        "f_mm": 200,
        "sigma_nm": 0.017,
        "err_frac": 0.20,
        "nm_per_px": 0.043,
        "label": "2400/200",
    },
    # σ = 48 pm (rms) at 0.11 nm/px as reported
    {
        "lines_per_mm": 1200,
        "f_mm": 200,
        "sigma_nm": 0.048,
        "err_frac": 0.10,
        "nm_per_px": 0.11,
        "label": "1200/200",
    },
]

rows = []
for m in measurements:
    scale = m["nm_per_px"] or budget(
        m["lines_per_mm"], m["f_mm"], alpha_nominal_deg
    )["nm_per_pixel"]
    if "sigma_px" in m:
        s_px, e_px = m["sigma_px"], m["err_px"]
    else:
        s_px = m["sigma_nm"] / scale
        e_px = s_px * m["err_frac"]
    rows.append(
        {
            "label": m["label"],
            "lines_per_mm": m["lines_per_mm"],
            "f_mm": m["f_mm"],
            "scale": scale,
            "scale_measured": m["nm_per_px"] is not None,
            "sigma_px": s_px,
            "err_px": e_px,
            "sigma_nm": s_px * scale,
            "err_nm": e_px * scale,
        }
    )
points = pd.DataFrame(rows)
setups = sorted({(r.lines_per_mm, r.f_mm) for r in points.itertuples()})


# Fit α: model σ_px vs α for each setup, χ² over all lines. The 2400 lines
# scatter more than their errors, so the interval is scaled by the reduced χ².
alpha_grid = np.arange(40.0, 89.0, 0.05)
sigma_curves = {
    s: np.array([budget(*s, a)["sigma_total_px"] for a in alpha_grid])
    for s in setups
}
model = np.array(
    [sigma_curves[(r.lines_per_mm, r.f_mm)] for r in points.itertuples()]
)
chi2 = (
    ((model - points["sigma_px"].to_numpy()[:, None])
     / points["err_px"].to_numpy()[:, None]) ** 2
).sum(axis=0)
alpha_fit = alpha_grid[np.argmin(chi2)]
chi2_red = chi2.min() / max(len(points) - 1, 1)
inside = alpha_grid[chi2 <= chi2.min() + max(chi2_red, 1.0)]
alpha_lo, alpha_hi = inside.min(), inside.max()


def alpha_from_width(sigma_px, lines_per_mm, f_mm, lo=40.0, hi=89.0):
    """α at which the ideal model gives this σ, or NaN outside [lo, hi]."""

    def gap(a):
        return budget(lines_per_mm, f_mm, a)["sigma_total_px"] - sigma_px

    if gap(lo) * gap(hi) > 0:
        return np.nan
    return brentq(gap, lo, hi, xtol=0.01)


points["alpha_deg"] = [
    alpha_from_width(r.sigma_px, r.lines_per_mm, r.f_mm)
    for r in points.itertuples()
]

# The other reading: α is the lab value and an angular blur makes up the rest
base_px = np.array(
    [budget(r.lines_per_mm, r.f_mm, alpha_nominal_deg)["sigma_total_px"]
     for r in points.itertuples()]
)
points["extra_urad"] = (
    np.sqrt(np.clip(points["sigma_px"] ** 2 - base_px**2, 0, None))
    * pixel_pitch_um / points["f_mm"] * 1e3
)
extra_urad_fit = math.sqrt(np.mean(points["extra_urad"] ** 2))

print(
    f"D = {beam_diameter_mm:.2f} mm; fitted α = {alpha_fit:.1f}° "
    f"({alpha_lo:.1f}–{alpha_hi:.1f}°, χ²_red = {chi2_red:.1f}), "
    f"per line {points['alpha_deg'].min():.1f}–{points['alpha_deg'].max():.1f}°"
)
print(
    f"At α = {alpha_nominal_deg:.0f}° instead: extra angular blur "
    f"{extra_urad_fit:.0f} µrad (per line "
    f"{points['extra_urad'].min():.0f}–{points['extra_urad'].max():.0f} µrad)"
)


def predict(configs, alpha_deg, extra_sigma_urad=0.0):
    """σ, FWHM and resolution (nm) for each (lines/mm, f mm)."""
    rows = []
    for g, f in configs:
        b = budget(g, f, alpha_deg, extra_sigma_urad)
        rows.append(
            {
                "config": f"{g}/{f}",
                "lines_per_mm": g,
                "f_mm": f,
                "nm_per_px": b["nm_per_pixel"],
                "sigma_nm": b["sigma_total_nm"],
                "fwhm_nm": b["fwhm_nm"],
                "resolution_nm": b["resolution_nm"],
                "R": b["resolving_power"],
            }
        )
    return pd.DataFrame(rows)


configs = [
    (1200, 100),
    (1200, 200),
    (1200, 300),
    (1200, 400),
    (2400, 200),
    (2400, 400),
]
predictions = predict(configs, alpha_fit)
predictions_nominal = predict(configs, alpha_nominal_deg, extra_urad_fit)


colors = {1200: "tab:blue", 2400: "tab:red"}

# Line σ vs α: where each setup's ideal curve meets its measured widths
plt.figure()
for s in setups:
    c = colors.get(s[0], "k")
    plt.plot(
        alpha_grid, sigma_curves[s], color=c, ls="-", marker="",
        label=f"{s[0]}/{s[1]}, model",
    )
    for r in points[points["lines_per_mm"] == s[0]].itertuples():
        plt.axhline(r.sigma_px, color=c, ls=":", lw=1)
plt.axvspan(alpha_lo, alpha_hi, color="0.5", alpha=0.2, label="Fitted α")
plt.axvline(alpha_nominal_deg, color="0.3", ls="--", label="Lab estimate")
plt.plot([], [], color="0.3", ls=":", label="Measured σ")
plt.xlabel("Incident angle α (°)")
plt.ylabel("Line σ (px)")
plt.legend(loc="upper right")


# Line σ vs scale. The optics floor in nm is fixed per grating (any f).
s = np.logspace(np.log10(0.005), np.log10(0.4), 200)
sigma_pix = pixel_width_frac / math.sqrt(12)
plt.figure()
for g in sorted({g for g, _ in configs} | set(points["lines_per_mm"])):
    c = colors.get(g, "k")
    floor = budget(g, 400, alpha_fit)["sigma_optics_nm"]
    floor_nominal = budget(g, 400, alpha_nominal_deg)["sigma_optics_nm"]
    plt.plot(
        s,
        np.sqrt(floor**2 + (sigma_pix * s) ** 2),
        color=c,
        ls="-",
        marker="",
        label=f"{g} gr/mm, α = {alpha_fit:.1f}°",
    )
    plt.plot(
        s,
        np.sqrt(floor_nominal**2 + (sigma_pix * s) ** 2),
        color=c,
        ls="--",
        marker="",
        label=f"{g} gr/mm, α = {alpha_nominal_deg:.0f}°, no extra blur",
    )
for r in points.itertuples():
    c = colors.get(r.lines_per_mm, "k")
    plt.errorbar(
        r.scale,
        r.sigma_nm,
        yerr=r.err_nm,
        fmt="o",
        mfc=c if r.scale_measured else "white",
        mec=c,
        ecolor=c,
        capsize=3,
        zorder=5,
    )
for p in predictions.itertuples():
    plt.plot(
        p.nm_per_px, p.sigma_nm, "D", color=colors.get(p.lines_per_mm, "k"),
        zorder=6,
    )
    plt.annotate(
        p.config, (p.nm_per_px, p.sigma_nm), xytext=(6, -12),
        textcoords="offset points",
    )
plt.plot([], [], "o", color="0.3", ls="", label="Measured")
plt.plot([], [], "D", color="0.3", ls="", label="Predicted (fitted α)")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Spectral scale (nm/px)")
plt.ylabel("Line σ (nm)")
plt.legend(loc="upper left", fontsize=8)


# FWHM against the 2-pixel sampling limit; diamonds = resolution.
plt.figure()
fw = pc.FWHM_PER_SIGMA
for g in sorted({g for g, _ in configs}):
    c = colors.get(g, "k")
    floor = budget(g, 400, alpha_fit)["sigma_optics_nm"]
    plt.plot(
        s,
        fw * np.sqrt(floor**2 + (sigma_pix * s) ** 2),
        color=c,
        ls="-",
        marker="",
        label=f"{g} gr/mm FWHM, α = {alpha_fit:.1f}°",
    )
plt.plot(
    s, 2 * s, color="0.2", ls=":", marker="", label="Sampling limit (2 px)"
)
for p in predictions.itertuples():
    plt.plot(
        p.nm_per_px, p.resolution_nm, "D", color=colors.get(p.lines_per_mm, "k"),
        zorder=6,
    )
    plt.annotate(
        p.config, (p.nm_per_px, p.resolution_nm), xytext=(6, -12),
        textcoords="offset points",
    )
plt.plot([], [], "D", color="0.3", ls="", label="Resolution = max(FWHM, 2 px)")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Spectral scale (nm/px)")
plt.ylabel("Δλ (nm)")
plt.legend(loc="upper left", fontsize=8)
