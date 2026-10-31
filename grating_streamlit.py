"""
GREG - Grating Equation Generator (Streamlit Web Application)

An interactive web application for diffraction grating analysis and spectrometer design.
Built with Streamlit, this tool provides real-time calculations and visualizations for
optical engineers and researchers working with diffraction gratings.

Features:
    - Output Angle Analysis: Calculate and visualize output angles vs incident angles,
      identify Littrow configurations, and compute spectral sampling (nm/pixel)
    - Sampling Sweep: Analyze how spectral sampling varies with different system
      parameters (focal length, groove density, wavelength, incident angle)
    - File Analysis: Upload and analyze data files with multi-plot visualization
      and statistical analysis
    - Spectral Units: wavelength / frequency / wavenumber converter with air vs
      vacuum, spectral widths and coherence time (ported from Prismatic)

Usage:
    Run with: streamlit run grating_streamlit.py
    Access at: http://localhost:8501

Dependencies:
    - streamlit: Web application framework
    - numpy: Numerical computations
    - matplotlib: Plotting and visualization
    - physics_core: Core grating physics calculations

Author: rngKomorebi
License: MIT
"""

import io

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.figure import Figure
from PIL import Image

import analysis_core as ac
import export_core as ec
import physics_core as pc
import spectral_units as su

# Page configuration
st.set_page_config(
    page_title="GREG - Grating Equation Generator",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _active_theme() -> str:
    """Return "light" or "dark" for the viewer's current Streamlit theme.

    Streamlit infers this from the app background colour, and the value lags by
    one rerun right after the theme is flipped (upstream issue #11920). Only the
    matplotlib figures depend on it: the CSS carries both palettes and is
    switched in the browser, and the theme-sync script requests the one rerun
    that brings the figures back in step.
    """
    try:
        theme_type = st.context.theme.type
    except Exception:  # noqa: BLE001 - no session context (bare script run)
        return "dark"
    return theme_type if theme_type in ("light", "dark") else "dark"


THEME = _active_theme()

# The hidden button the theme-sync script clicks to request a rerun.
THEME_SYNC_KEY = "greg_theme_sync"
THEME_SYNC_LABEL = "sync theme"

# Design tokens for both themes. The stylesheet and the inline HTML reference
# these as CSS variables (--greg-*), so a theme flip repaints the whole UI in
# the browser with no rerun. The dark values are the original palette.
PALETTES = {
    "dark": {
        "bg": "#131313",
        "sidebar": "#1c1b1b",
        "nav": "#201f1f",
        "surface": "#2a2a2a",
        "accent": "#42e5b0",
        "text": "#e5e2e1",
        "muted": "#bbcac1",
        "border": "#3c4a43",
        "font_ui": "'Inter', sans-serif",
        "focus": "rgba(163, 166, 255, 0.15)",
        # Matplotlib side: baked into the PNGs, so these cannot be CSS variables.
        "plot_bg": "#0e1117",
        "plot_surface": "#0e1117",
        "plot_axis": "white",
        "plot_grid": "gray",
        "grid_rgb": (14, 17, 23),
    },
    # "Precision Technical Interface" — light, lab-report aesthetic.
    "light": {
        "bg": "#ffffff",
        "sidebar": "#f2f4f6",
        "nav": "#eceef0",
        "surface": "#f7f9fb",
        "accent": "#10b981",
        "text": "#191c1e",
        "muted": "#45464d",
        "border": "#e2e8f0",
        "font_ui": "'Hanken Grotesk', sans-serif",
        "focus": "rgba(16, 185, 129, 0.18)",
        "plot_bg": "#ffffff",
        "plot_surface": "#ffffff",
        "plot_axis": "#191c1e",
        "plot_grid": "#c6c6cd",
        "grid_rgb": (255, 255, 255),
    },
}

# Python-side colours, used only where a value has to be baked in server-side:
# the matplotlib figures and the downloaded plot grid.
_P = PALETTES[THEME]
ACCENT = _P["accent"]
TEXT = _P["text"]
TEXT_MUTED = _P["muted"]
PLOT_BG = _P["plot_bg"]
PLOT_SURFACE = _P["plot_surface"]
PLOT_AXIS = _P["plot_axis"]
PLOT_GRID = _P["plot_grid"]
GRID_RGB = _P["grid_rgb"]

# The six analysis figures are built inside analysis_core, which keeps its own
# plot palette; point it at the active theme before anything is drawn.
ac.set_theme(THEME)


def _token_block(selector: str, palette: dict) -> str:
    """Emit the --greg-* custom properties for one palette under one selector."""
    return selector + " {\n" + "".join(
        f"        --greg-{name}: {palette[key]};\n"
        for name, key in (
            ("bg", "bg"),
            ("sidebar", "sidebar"),
            ("nav", "nav"),
            ("surface", "surface"),
            ("accent", "accent"),
            ("text", "text"),
            ("muted", "muted"),
            ("border", "border"),
            ("font-ui", "font_ui"),
            ("focus", "focus"),
        )
    ) + "    }"

# Camera presets: display name → {pitch_um, pixels, fill_factor, microlenses,
# extra_sigma_um}. Pitch and pixel count seed editable sidebar fields, so
# other cameras can be projected. fill_factor is the light-sensitive fraction
# of the pixel area; with microlenses the effective pixel width is taken as
# the full pitch. extra_sigma_um is the default detector blur of the
# resolution model.
CAMERAS = {
    # Measured Ne line σ: 0.79 px at 2400 gr/mm / 400 mm (0.015 nm,
    # 0.0191 nm/px) and 0.44 px at 1200 gr/mm / 200 mm (0.048 nm,
    # 0.11 nm/px), 640 nm. The beam's own diffraction (2.1 mm beam) explains
    # both at α ≈ 65°; at α ≈ 80° both need ≈ 40 µrad of extra angular blur,
    # i.e. a blur that scales with f, not one on the detector. Until α is
    # measured the detector blur stays at zero.
    "LinoSPAD2 (26.2 µm)": {"pitch_um": 26.2, "pixels": 256,
                            "fill_factor": 0.58, "microlenses": True,
                            "extra_sigma_um": 0.0},
    "Timepix3 (55 µm)": {"pitch_um": 55.0, "pixels": 256,
                         "fill_factor": 1.0, "microlenses": False,
                         "extra_sigma_um": 0.0},
    "Moravian CMOS (3.45 µm)": {"pitch_um": 3.45, "pixels": 8336,
                                "fill_factor": 1.0, "microlenses": True,
                                "extra_sigma_um": 0.0},
}

PAGES = ["Output Angle", "Sampling Sweep", "Spectrometer Analysis", "Spectral Units"]

# Wavelength inputs on pages 1 and 2. Their defaults are seeded through session
# state rather than value=, so the Spectral Units page can set them too
# (Streamlit warns when a widget gets both a value= and a session-state value).
WAVELENGTH_KEYS = {"Output Angle": "wavelength_oa", "Sampling Sweep": "wavelength_sw"}
WAVELENGTH_MIN_NM, WAVELENGTH_MAX_NM = 300.0, 1000.0
for _k in WAVELENGTH_KEYS.values():
    st.session_state.setdefault(_k, 650.0)

LENS_AXIS_OPTIONS = {
    "Diffracted beam (cosβ)": pc.LENS_AXIS_BEAM,
    "Grating normal (cos³β)": pc.LENS_AXIS_NORMAL,
}


def _optics_inputs(suffix: str, lambda_nm: float) -> tuple[dict, dict]:
    """Sidebar inputs for the resolution model, shared by pages 1 and 2.

    Returns the keyword arguments for physics_core.line_width_budget and, for
    the data export, the fibre and collimator the beam diameter came from.
    """
    st.subheader("Resolution Model (experimental)")
    lens_axis = LENS_AXIS_OPTIONS[
        st.selectbox(
            "Camera lens axis",
            list(LENS_AXIS_OPTIONS),
            index=0,
            key=f"lens_axis_{suffix}",
            help="Direction the camera lens points. Along the diffracted beam "
            "(the usual layout) the dispersion goes as cosβ; along the grating "
            "normal the detector position is x = f·tanβ, so it goes as cos³β. "
            "Check against a measured nm/px.",
        )
    ]
    collimator_f_mm = st.number_input(
        "Collimator focal length (mm)",
        min_value=0.1,
        max_value=1000.0,
        value=10.99,
        step=0.1,
        format="%.2f",
        key=f"collimator_f_{suffix}",
        help="Fibre collimator that makes the beam on the grating. Default: "
        "Thorlabs PF2F63 (10.99 mm).",
    )
    mfd_um = st.number_input(
        "Fibre mode-field diameter (µm)",
        min_value=0.5,
        max_value=100.0,
        value=4.3,
        step=0.1,
        format="%.2f",
        key=f"mfd_{suffix}",
        help="Mode-field diameter of the single-mode fibre near the working "
        "wavelength. Default: SM600 (P1-630A), 4.3 µm at 633 nm, specified "
        "3.6–5.3 µm.",
    )
    beam_diameter_mm = pc.collimated_beam_diameter_mm(
        collimator_f_mm, mfd_um, lambda_nm
    )
    st.caption(
        f"Beam on the grating: D = 4λf/(π·MFD) = **{beam_diameter_mm:.2f} mm** "
        "(1/e²), held at this value in a λ sweep."
    )
    grating_width_mm = st.number_input(
        "Grating width (mm)",
        min_value=1.0,
        max_value=1000.0,
        value=25.0,
        step=1.0,
        key=f"grating_width_{suffix}",
        help="Ruled width across the grooves. When the beam footprint D/cosα "
        "is wider, the grating edges cut the beam and the resolution stops "
        "improving with α.",
    )
    microlenses = st.checkbox(
        "Microlenses (pixel width = pitch)",
        value=_cam["microlenses"],
        key=f"microlenses_{suffix}_{selected_camera}",
        help=f"Fill factor {_cam['fill_factor']:.0%}. Without microlenses the "
        "light-sensitive width is taken as √(fill factor) × pitch.",
    )
    extra_sigma_urad = st.number_input(
        "Extra angular blur σ (µrad)",
        min_value=0.0,
        max_value=10000.0,
        value=0.0,
        step=5.0,
        key=f"extra_sigma_urad_{suffix}",
        help="Angular spread the ideal beam does not have (collimation, "
        "mirror and grating flatness). On the detector it scales with f. "
        "Zero gives the ideal instrument.",
    )
    extra_sigma_um = st.number_input(
        "Extra detector blur σ (µm)",
        min_value=0.0,
        max_value=1000.0,
        value=_cam["extra_sigma_um"],
        step=1.0,
        key=f"extra_sigma_{suffix}_{selected_camera}",
        help="Blur on the detector itself (crosstalk, timing); it does not "
        "scale with f. Zero gives the ideal instrument.",
    )
    optics = {
        "lens_axis": lens_axis,
        "beam_diameter_mm": beam_diameter_mm,
        "grating_width_mm": grating_width_mm,
        "pixel_width_frac": 1.0 if microlenses else _cam["fill_factor"] ** 0.5,
        "extra_sigma_urad": extra_sigma_urad,
        "extra_sigma_um": extra_sigma_um,
    }
    beam_source = {"collimator_f_mm": collimator_f_mm, "mfd_um": mfd_um}
    return optics, beam_source


def _fmt_nm(v: float) -> str:
    return f"{v:.3g} nm"


def _show_resolution(budget: dict, source: str) -> None:
    """Resolution cards for pages 1 and 2, from physics_core.line_width_budget.

    Also remembers the result so the Spectral Units page can offer it as a
    spectral width.
    """
    b = budget
    st.session_state["_last_resolution"] = (b["resolution_nm"], source)
    st.caption(
        "**Experimental:** the resolution projection is still being checked "
        "against measured Ne line widths. Treat it as an estimate."
    )
    r1, r2, r3 = st.columns(3)
    with r1:
        st.info(
            f"Line σ: **{_fmt_nm(b['sigma_total_nm'])}**  ·  FWHM "
            f"**{_fmt_nm(b['fwhm_nm'])}**"
        )
    with r2:
        st.info(
            f"Resolution: **{_fmt_nm(b['resolution_nm'])}**  ·  "
            f"R = λ/Δλ ≈ **{b['resolving_power']:,.0f}**"
        )
    with r3:
        _limit_text = {
            "sampling": "Bottleneck: **pixel sampling** (FWHM < 2 px)",
            "optics": "Bottleneck: **optics** (beam diffraction)",
            "angular": "Bottleneck: **extra angular blur**",
            "pixel": "Bottleneck: **pixel width**",
            "extra": "Bottleneck: **extra detector blur**",
        }
        st.warning(_limit_text[b["limiting"]])
    st.caption(
        f"σ contributions: optics {_fmt_nm(b['sigma_optics_nm'])} (beam "
        f"×{b['anamorphic_factor']:.2f} after the grating) · angular "
        f"{_fmt_nm(b['sigma_angular_nm'])} · pixel "
        f"{_fmt_nm(b['sigma_pixel_nm'])} · detector "
        f"{_fmt_nm(b['sigma_extra_nm'])}, added in quadrature. Resolution is "
        f"the larger of the FWHM and the 2-pixel sampling limit "
        f"({_fmt_nm(b['sampling_limit_nm'])})."
    )
    if b["clipped"]:
        st.caption(
            f"The beam footprint on the grating ({b['footprint_mm']:.0f} mm) "
            "is wider than the grating, so its edges cut the beam: a steeper "
            "α no longer narrows the line."
        )


# ── Spectral Units page state ─────────────────────────────────────────────────
# The canonical value is the vacuum wavelength in nm (su_vac_nm, not a widget),
# so the three fields, their units and the air/vacuum choice only change how it
# is displayed. Streamlit drops a widget's value when its page is not shown, so
# _su_restore() puts the last values back on return; _su_save() stores them at
# the end of every Spectral Units run.
SU_MEDIA = ["Air (standard)", "Vacuum"]
_SU_DEFAULTS = {
    "su_medium": SU_MEDIA[0],
    "su_wl_unit": "nm",
    "su_freq_unit": "THz",
    "su_wnum_unit": "cm⁻¹",
    "su_width_unit": "pm",
    "su_width": 50.0,
}
_SU_INIT_VAC_NM = su.air_to_vacuum(632.8)  # HeNe


def _su_air() -> bool:
    return st.session_state.get("su_medium", SU_MEDIA[0]) == SU_MEDIA[0]


def _su_refresh_fields() -> None:
    """Rewrite the three value fields from the canonical vacuum wavelength."""
    ss = st.session_state
    vac = ss["su_vac_nm"]
    wl = su.vacuum_to_air(vac) if _su_air() else vac
    ss["su_wl"] = wl / su.WL_UNITS[ss["su_wl_unit"]]
    ss["su_freq"] = su.freq_thz(vac) / su.FREQ_UNITS[ss["su_freq_unit"]]
    ss["su_wnum"] = su.wnum_cm(vac) / su.WNUM_UNITS[ss["su_wnum_unit"]]


def _su_restore() -> None:
    ss = st.session_state
    ss.setdefault("su_vac_nm", _SU_INIT_VAC_NM)
    for key, default in _SU_DEFAULTS.items():
        if key not in ss:
            ss[key] = ss.get(f"_saved_{key}", default)
    if any(k not in ss for k in ("su_wl", "su_freq", "su_wnum")):
        _su_refresh_fields()


def _su_save() -> None:
    for key in _SU_DEFAULTS:
        st.session_state[f"_saved_{key}"] = st.session_state[key]


def _su_set_vacuum(vac_nm: float) -> None:
    st.session_state["su_vac_nm"] = vac_nm
    _su_refresh_fields()


def _su_on_wl() -> None:
    v = st.session_state["su_wl"]
    if v and v > 0:
        wl = v * su.WL_UNITS[st.session_state["su_wl_unit"]]
        _su_set_vacuum(su.air_to_vacuum(wl) if _su_air() else wl)


def _su_on_freq() -> None:
    v = st.session_state["su_freq"]
    if v and v > 0:
        _su_set_vacuum(
            su.wl_from_freq(v * su.FREQ_UNITS[st.session_state["su_freq_unit"]])
        )


def _su_on_wnum() -> None:
    v = st.session_state["su_wnum"]
    if v and v > 0:
        _su_set_vacuum(
            su.wl_from_wnum(v * su.WNUM_UNITS[st.session_state["su_wnum_unit"]])
        )


def _su_jump(wl_nm: float, in_air: bool) -> None:
    _su_set_vacuum(su.air_to_vacuum(wl_nm) if in_air else wl_nm)


def _su_send(target_page: str) -> None:
    """Put the current air wavelength into page 1 or 2 and open that page."""
    wl_air = su.vacuum_to_air(st.session_state["su_vac_nm"])
    st.session_state[WAVELENGTH_KEYS[target_page]] = round(wl_air, 4)
    st.session_state["page"] = target_page


def _su_use_width(width_nm: float) -> None:
    st.session_state["su_width_unit"] = "pm"
    st.session_state["su_width"] = width_nm * 1e3


def _frequency_caption(wl_air_nm: float) -> None:
    """Small ν / ν̃ readout under a wavelength input (λ taken as air)."""
    wl_vac = su.air_to_vacuum(wl_air_nm)
    st.caption(
        f"ν = {su.freq_thz(wl_vac):.3f} THz · ν̃ = {su.wnum_cm(wl_vac):,.2f} cm⁻¹ "
        f"(λ in air; vacuum {wl_vac:.3f} nm)"
    )


def _data_download_button(tables: dict, meta: dict, stem: str, key: str) -> None:
    """Offer the data behind a page's plots as a ZIP of CSVs plus run.json.

    on_click="ignore" stops the click from rerunning the app, so downloading
    never recomputes the page.
    """
    st.download_button(
        "Download data (CSV + JSON)",
        data=ec.build_zip(tables, meta),
        file_name=f"{stem}_greg_data.zip",
        mime="application/zip",
        key=key,
        on_click="ignore",
        help="A ZIP with one CSV per plot, run.json with the inputs and "
        "results, and a README on loading them.",
    )


def _make_grating_diagram(
    alpha_deg: float,
    lines_per_mm: float,
    wavelength_nm: float,
    order_m: int,
) -> Figure:
    """Return a Figure illustrating the grating geometry for the current params."""
    d_nm = 1e6 / lines_per_mm
    alpha_rad = np.radians(alpha_deg)
    L = 3.2

    # Determine which orders exist
    valid_orders, evanescent_m = [], []
    for m in range(-3, 5):
        s = m * wavelength_nm / d_nm - np.sin(alpha_rad)
        if -1.0 <= s <= 1.0:
            valid_orders.append((m, float(np.degrees(np.arcsin(s)))))
        else:
            evanescent_m.append(m)
    valid_orders.sort(key=lambda x: x[0])

    # Colour scheme: selected order = ACCENT, m=0 = MUTED, rest cycle
    _cycle = ["#FFD10F", "#c084fc", "#FF8C42", "#f87171", "#60a5fa"]
    order_colors: dict = {}
    ci = 0
    for m, _ in valid_orders:
        if m == order_m:
            order_colors[m] = ACCENT
        elif m == 0:
            order_colors[m] = TEXT_MUTED
        else:
            order_colors[m] = _cycle[ci % len(_cycle)]
            ci += 1

    fig = Figure(figsize=(12, 6.5), facecolor=PLOT_BG)
    ax = fig.add_subplot(111, facecolor=PLOT_BG)
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-1.4, 5.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Grating surface
    ax.plot([-5, 5], [0, 0], color=ACCENT, lw=3, solid_capstyle="round", zorder=5)
    for x in np.linspace(-4.8, 4.8, 38):
        ax.plot([x, x - 0.055], [0, -0.2], color=ACCENT, lw=0.7, alpha=0.4)
    ax.text(4.8, -0.72, "grating", color=ACCENT, fontsize=10, ha="right")

    # Normal
    ax.plot([0, 0], [-0.7, 4.7], color=TEXT_MUTED, lw=1.1, linestyle="--", alpha=0.55)
    ax.annotate(
        "", xy=(0, 4.7), xytext=(0, 4.35),
        arrowprops=dict(arrowstyle="->", color=TEXT_MUTED, lw=1.1),
    )
    ax.text(0.12, 4.58, "normal", color=TEXT_MUTED, fontsize=10)

    # Incident beam: source at (−sin α · L, cos α · L)
    sx, sy = -np.sin(alpha_rad) * L, np.cos(alpha_rad) * L
    ax.annotate(
        "", xy=(0, 0), xytext=(sx, sy),
        arrowprops=dict(arrowstyle="->", color=PLOT_AXIS, lw=2.2, mutation_scale=15),
    )
    ax.text(sx - 0.15, sy + 0.12, "incident", color=PLOT_AXIS, fontsize=11,
            ha="right", fontweight="bold")

    # α arc (from normal CCW to incident beam direction)
    r_a = 0.85
    ax.add_patch(mpatches.Arc(
        (0, 0), r_a * 2, r_a * 2, angle=0,
        theta1=90, theta2=90 + alpha_deg, color=PLOT_AXIS, lw=1.6,
    ))
    mid_a = np.radians(90 + alpha_deg / 2)
    ax.text(
        np.cos(mid_a) * (r_a + 0.3), np.sin(mid_a) * (r_a + 0.3),
        f"α = {alpha_deg:.0f}°", color=PLOT_AXIS, fontsize=11,
        fontweight="bold", ha="center", va="center",
    )

    # Diffracted orders: beam endpoint = (−sin β · L, cos β · L)
    beta_selected = None
    for m, beta_deg in valid_orders:
        color = order_colors.get(m, TEXT_MUTED)
        lw = 2.4 if m == order_m else 1.7
        beta_rad = np.radians(beta_deg)
        ex, ey = -np.sin(beta_rad) * L, np.cos(beta_rad) * L
        ax.annotate(
            "", xy=(ex, ey), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=color, lw=lw, mutation_scale=14),
        )
        side = "right" if ex > 0 else "left"
        dx, ha = (0.18, "left") if side == "right" else (-0.18, "right")
        fw = "bold" if m == order_m else "normal"
        m_label = f"m = {m:+d}" if m != 0 else "m = 0"
        ax.text(ex + dx, ey + 0.12, m_label, color=color, fontsize=11, ha=ha, fontweight=fw)
        if m == 0:
            ax.text(ex + dx, ey - 0.28, "specular  (β = −α)",
                    color=color, fontsize=8, ha=ha, alpha=0.8)
        if m == order_m:
            beta_selected = beta_deg
            ax.text(ex + dx, ey - 0.28, f"β = {beta_deg:+.1f}°",
                    color=color, fontsize=8, ha=ha)

    # β arc for the selected order
    if beta_selected is not None:
        r_b = 1.5
        col_sel = order_colors.get(order_m, ACCENT)
        t1 = 90 if beta_selected >= 0 else 90 + beta_selected
        t2 = 90 + beta_selected if beta_selected >= 0 else 90
        ax.add_patch(mpatches.Arc(
            (0, 0), r_b * 2, r_b * 2, angle=0, theta1=t1, theta2=t2,
            color=col_sel, lw=1.5,
        ))
        mid_b = np.radians(90 + beta_selected / 2)
        ax.text(
            np.cos(mid_b) * (r_b + 0.32), np.sin(mid_b) * (r_b + 0.32),
            "β", color=col_sel, fontsize=15, fontweight="bold", ha="center", va="center",
        )

    # Side legend
    ax.text(-4.8, 4.72, "β > 0  (same side as incident)", color=TEXT_MUTED, fontsize=9)
    ax.text(4.8, 4.72, "β < 0  (opposite side)", color=TEXT_MUTED, fontsize=9, ha="right")

    # Evanescent note
    ev_shown = sorted(m for m in evanescent_m if -2 <= m <= 3)
    if ev_shown:
        ax.text(
            0, -0.98,
            "Evanescent (not shown): " + ", ".join(f"m={m:+d}" for m in ev_shown),
            color=TEXT_MUTED, fontsize=8, ha="center", alpha=0.65,
        )

    # Equation footer
    ax.text(
        0, -1.2,
        (rf"$m\lambda = d\,(\sin\alpha + \sin\beta)$   ·   "
         rf"{int(lines_per_mm)} gr/mm · λ = {wavelength_nm:.0f} nm · α = {alpha_deg:.0f}°"),
        color=TEXT_MUTED, fontsize=10, ha="center",
    )

    fig.tight_layout(pad=0.3)
    return fig


# Light-mode-only refinements. Empty in dark mode, so the dark stylesheet is
# byte-for-byte what it has always been. Streamlit's own light theme already
# gives st.info/success/warning the pale blue/green/gold semantic tints the
# design calls for, so alerts are deliberately left alone.
LIGHT_CSS = """
    /* Numbers and technical values in a monospaced face so decimal points
       line up column-wise (DESIGN.md: JetBrains Mono for data/math). */
    html[data-greg-theme="light"] [data-testid="stMetricValue"],
    html[data-greg-theme="light"] [data-testid="stMetricDelta"],
    html[data-greg-theme="light"] .stNumberInput input,
    html[data-greg-theme="light"] .stTextInput input,
    html[data-greg-theme="light"] [data-testid="stSelectbox"] > div > div,
    html[data-greg-theme="light"] [data-testid="stAlertContainer"] p,
    html[data-greg-theme="light"] [data-testid="stCaptionContainer"] p,
    html[data-greg-theme="light"] code,
    html[data-greg-theme="light"] pre,
    html[data-greg-theme="light"] .katex {
        font-family: 'JetBrains Mono', monospace !important;
    }
    /* Flat, outlined "lab report" surfaces instead of shadows. */
    html[data-greg-theme="light"] [data-testid="stMetric"] {
        box-shadow: none !important;
    }
    html[data-greg-theme="light"] [data-testid="stMetric"]:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
    }
"""

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100;300;400;500;600;700;800;900&family=Hanken+Grotesk:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Both palettes ship on every page load. The theme-sync script below stamps
       data-greg-theme on <html>, so flipping the theme repaints instantly in the
       browser instead of waiting for a rerun. The :root block is the server's
       best guess, used for the very first paint before the script runs. */
{_token_block("    :root", PALETTES[THEME])}
{_token_block('    html[data-greg-theme="dark"]', PALETTES["dark"])}
{_token_block('    html[data-greg-theme="light"]', PALETTES["light"])}

    html, body, .stApp {{
        background-color: var(--greg-bg) !important;
        font-family: var(--greg-font-ui) !important;
        color: var(--greg-text) !important;
    }}
    .main .block-container {{
        background-color: var(--greg-bg) !important;
        padding: 2rem 2.5rem !important;
        max-width: 100% !important;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: var(--greg-text) !important;
        font-family: var(--greg-font-ui) !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
    }}
    p, span, label, div, .stMarkdown {{
        font-family: var(--greg-font-ui) !important;
    }}
    /* The rule above would otherwise beat Streamlit's icon font and make the
       Material ligature names ("light_mode", "contrast", ...) render as text. */
    [data-testid="stIconMaterial"],
    [data-testid="stAlertDynamicIcon"],
    [data-testid="stExpanderIcon"],
    [data-testid="stExpanderIconCheck"],
    [data-testid="stExpanderIconError"],
    [data-testid="stExpanderIconSpinner"],
    [data-testid="stSpinnerIcon"],
    [data-testid="stElementToolbarButtonIcon"],
    [data-testid="stChatInputFileIconError"],
    [data-testid="stChatInputFileIconSpinner"],
    span[translate="no"] {{
        font-family: 'Material Symbols Rounded' !important;
    }}
    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"],
    [data-testid="stSidebar"] .block-container {{
        background-color: var(--greg-sidebar) !important;
        border-right: 1px solid var(--greg-border) !important;
    }}
    [data-testid="stSidebar"] .block-container {{
        padding: 0 !important;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] {{
        margin: 0 !important;
        padding: 0 !important;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label {{
        display: flex !important;
        align-items: center !important;
        padding: 0.65rem 1.5rem !important;
        cursor: pointer !important;
        border-left: 2px solid transparent !important;
        color: var(--greg-muted) !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        margin: 0 !important;
        width: 100% !important;
        transition: all 0.15s ease !important;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
        background-color: var(--greg-nav) !important;
        color: var(--greg-text) !important;
        border-left-color: var(--greg-accent) !important;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
        background-color: var(--greg-nav) !important;
        color: var(--greg-accent) !important;
        border-left: 2px solid var(--greg-accent) !important;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {{
        display: none !important;
    }}
    [data-testid="stSidebar"] h3 {{
        color: var(--greg-muted) !important;
        font-size: 0.6rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.18em !important;
        text-transform: uppercase !important;
        padding: 0 1.5rem !important;
        margin-top: 1.25rem !important;
        margin-bottom: 0.25rem !important;
    }}
    .stNumberInput input, .stTextInput input {{
        background-color: var(--greg-surface) !important;
        border: 1px solid var(--greg-border) !important;
        color: var(--greg-text) !important;
        border-radius: 0.25rem !important;
    }}
    .stNumberInput input:focus, .stTextInput input:focus {{
        border-color: var(--greg-accent) !important;
        box-shadow: 0 0 0 2px var(--greg-focus) !important;
    }}
    [data-testid="stSelectbox"] > div > div {{
        background-color: var(--greg-surface) !important;
        border-color: var(--greg-border) !important;
        color: var(--greg-text) !important;
    }}
    [data-testid="stMetric"] {{
        background-color: var(--greg-surface) !important;
        border: 1px solid var(--greg-border) !important;
        border-left: 2px solid var(--greg-accent) !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem 1rem !important;
    }}
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
        color: var(--greg-text) !important;
    }}
    [data-testid="stFileUploader"] {{
        background-color: var(--greg-surface) !important;
        border: 1px dashed var(--greg-border) !important;
        border-radius: 0.25rem !important;
    }}
    hr {{ border-color: var(--greg-border) !important; opacity: 0.4 !important; }}
    [data-testid="collapsedControl"],
    button[kind="header"][aria-label="Close sidebar"],
    [data-testid="stSidebarCollapseButton"] {{
        display: none !important;
    }}
    /* Off-screen rather than display:none, so a scripted click still reaches
       React's handler. This is the hook the theme-sync script uses to request
       one rerun when the theme changes, so the matplotlib figures follow. */
    .st-key-{THEME_SYNC_KEY} {{
        position: fixed !important;
        left: -9999px !important;
        top: 0 !important;
        width: 1px !important;
        height: 1px !important;
        overflow: hidden !important;
    }}
{LIGHT_CSS}
    </style>
    """,
    unsafe_allow_html=True,
)

# One-shot rerun hook: the script only clicks this when it observes the theme
# actually change, and a rerun does not change the theme, so it cannot loop.
st.button(THEME_SYNC_LABEL, key=THEME_SYNC_KEY)

components.html(
    f"""
    <script>
    (function () {{
        var pWin = window.parent, pDoc = pWin.document;
        // Streamlit persists the user's choice here (frontend LocalStore.ACTIVE_THEME).
        var KEY = "stActiveTheme-" + pWin.location.pathname + "-v2";
        var media = pWin.matchMedia("(prefers-color-scheme: dark)");
        var seen = null;

        function liveTheme() {{
            var choice = null;
            try {{ choice = JSON.parse(pWin.localStorage.getItem(KEY)); }} catch (e) {{}}
            if (choice === "Light") return "light";
            if (choice === "Dark") return "dark";
            return media.matches ? "dark" : "light";  // "System" or never set
        }}

        function sync() {{
            var theme = liveTheme();
            pDoc.documentElement.setAttribute("data-greg-theme", theme);
            if (seen === null) {{ seen = theme; return; }}
            if (theme === seen) return;
            seen = theme;
            // CSS has already repainted; ask for a rerun so the plots catch up.
            var btn = pDoc.querySelector(".st-key-{THEME_SYNC_KEY} button");
            if (!btn) {{  // fallback if the key-derived class ever changes
                pDoc.querySelectorAll("button").forEach(function (b) {{
                    if (!btn && b.innerText.trim() === "{THEME_SYNC_LABEL}") btn = b;
                }});
            }}
            if (btn) btn.click();
        }}

        sync();
        pWin.addEventListener("storage", sync);
        media.addEventListener("change", sync);
        setInterval(sync, 300);
    }})();
    </script>
    """,
    height=0,
    scrolling=False,
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 1.5rem 1.5rem 0.75rem; font-family: var(--greg-font-ui);">
            <div style="font-size: 1.4rem; font-weight: 900; color: var(--greg-text);
                        letter-spacing: -0.04em;">GREG</div>
            <div style="font-size: 0.6rem; color: var(--greg-muted); letter-spacing: 0.2em;
                        text-transform: uppercase; margin-top: 0.2rem;">
                Grating Equation Generator
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_camera = st.selectbox(
        "Camera",
        list(CAMERAS.keys()),
        index=0,
        key="camera_select",
    )
    _cam = CAMERAS[selected_camera]
    # Seeded from the preset, keyed per camera so switching camera restores it
    _pitch_col, _pixels_col = st.columns(2)
    with _pitch_col:
        pixel_pitch_um = st.number_input(
            "Pixel pitch (µm)",
            min_value=0.1,
            max_value=1000.0,
            value=float(_cam["pitch_um"]),
            step=0.1,
            format="%.2f",
            key=f"pitch_{selected_camera}",
            help="Centre-to-centre pixel spacing along the dispersion "
            "direction. Change it to project to another camera.",
        )
    with _pixels_col:
        camera_pixels = int(st.number_input(
            "Sensor size (px)",
            min_value=1,
            max_value=100000,
            value=int(_cam["pixels"]),
            step=1,
            key=f"pixels_{selected_camera}",
            help="Number of pixels along the dispersion direction; sets the "
            "spectral range.",
        ))

    page = st.radio(
        "Page",
        PAGES,
        key="page",
        label_visibility="collapsed",
    )

    st.divider()

    # Output Angle params
    if page == "Output Angle":
        st.subheader("Instrument Parameters")
        st.caption(f"Pixel pitch p = **{pixel_pitch_um:.3f} µm**  ·  **{camera_pixels}** px")
        lines_per_mm = st.number_input(
            "Groove density (lines/mm)",
            min_value=50,
            max_value=3600,
            value=2400,
            step=50,
            key="lines_per_mm_oa",
        )
        alpha_deg = st.number_input(
            "Incident angle α (°)",
            min_value=0.0,
            max_value=90.0,
            value=45.0,
            step=1.0,
            key="alpha_deg_oa",
        )
        wavelength_nm = st.number_input(
            "Wavelength λ (nm)",
            min_value=WAVELENGTH_MIN_NM,
            max_value=WAVELENGTH_MAX_NM,
            step=1.0,
            key="wavelength_oa",
        )
        _frequency_caption(wavelength_nm)
        order_m = st.number_input(
            "Order m",
            min_value=-5,
            max_value=5,
            value=1,
            step=1,
            key="order_oa",
        )
        focal_length_mm = st.number_input(
            "Focal length f (mm)",
            min_value=50.0,
            max_value=1000.0,
            value=400.0,
            step=50.0,
            key="focal_length_oa",
        )
        optics_oa, beam_source_oa = _optics_inputs("oa", wavelength_nm)
        st.subheader("Plot Options")
        alpha_min = st.number_input(
            "α min (°)",
            min_value=0.0,
            max_value=89.9,
            value=0.0,
            step=1.0,
            key="alpha_min_oa",
        )
        alpha_max = st.number_input(
            "α max (°)",
            min_value=0.1,
            max_value=89.9,
            value=89.9,
            step=1.0,
            key="alpha_max_oa",
        )
        alpha_step = st.number_input(
            "Step (°)",
            min_value=0.1,
            max_value=20.0,
            value=1.0,
            step=0.5,
            key="alpha_step_oa",
        )
        colormap = st.selectbox(
            "Colormap",
            ["cool", "viridis", "plasma", "inferno", "magma"],
            index=0,
            key="colormap_oa",
        )
        marker_size = st.slider(
            "Marker size",
            min_value=1.0,
            max_value=100.0,
            value=20.0,
            step=1.0,
            key="marker_size_oa",
        )
        show_littrow = st.checkbox(
            "Show Littrow configuration", value=True, key="show_littrow_oa"
        )

    # Sampling Sweep params
    elif page == "Sampling Sweep":
        st.subheader("Instrument Parameters")
        st.caption(f"Pixel pitch p = **{pixel_pitch_um:.3f} µm**  ·  **{camera_pixels}** px")
        lines_per_mm_sw = st.number_input(
            "Groove density (lines/mm)",
            min_value=50,
            max_value=3600,
            value=2400,
            step=50,
            key="lines_per_mm_sw",
        )
        alpha_deg_sw = st.number_input(
            "Incident angle α (°)",
            min_value=0.0,
            max_value=90.0,
            value=45.0,
            step=1.0,
            key="alpha_deg_sw",
        )
        wavelength_nm_sw = st.number_input(
            "Wavelength λ (nm)",
            min_value=WAVELENGTH_MIN_NM,
            max_value=WAVELENGTH_MAX_NM,
            step=1.0,
            key="wavelength_sw",
        )
        _frequency_caption(wavelength_nm_sw)
        order_m_sw = st.number_input(
            "Order m",
            min_value=-5,
            max_value=5,
            value=1,
            step=1,
            key="order_sw",
        )
        focal_length_mm_sw = st.number_input(
            "Focal length f (mm)",
            min_value=50.0,
            max_value=1000.0,
            value=400.0,
            step=50.0,
            key="focal_length_sw",
        )
        optics_sw, beam_source_sw = _optics_inputs("sw", wavelength_nm_sw)
        st.subheader("Sweep Options")
        sweep_param = st.selectbox(
            "Sweep parameter",
            ["f (mm)", "lines/mm", "λ (nm)", "α (deg)"],
            index=0,
            key="sweep_param",
        )
        if sweep_param == "f (mm)":
            default_start, default_stop = 50.0, 1000.0
            min_val, max_val, step_val = 1.0, 10000.0, 10.0
        elif sweep_param == "lines/mm":
            default_start, default_stop = 100.0, 3600.0
            min_val, max_val, step_val = 50.0, 10000.0, 50.0
        elif sweep_param == "λ (nm)":
            default_start, default_stop = 300.0, 1000.0
            min_val, max_val, step_val = 100.0, 5000.0, 10.0
        else:
            default_start, default_stop = 0.0, 89.0
            min_val, max_val, step_val = 0.0, 89.9, 1.0
        sweep_start = st.number_input(
            "Sweep start",
            min_value=min_val,
            max_value=max_val,
            value=default_start,
            step=step_val,
            key=f"sweep_start_{sweep_param}",
        )
        sweep_stop = st.number_input(
            "Sweep stop",
            min_value=min_val,
            max_value=max_val,
            value=default_stop,
            step=step_val,
            key=f"sweep_stop_{sweep_param}",
        )
        sweep_points = st.number_input(
            "Points",
            min_value=2,
            max_value=5000,
            value=50,
            step=10,
            key="sweep_points",
        )

    # Spectral Units params
    elif page == "Spectral Units":
        st.subheader("Wavelength Medium")
        _su_restore()
        st.radio(
            "Wavelengths are given in",
            SU_MEDIA,
            key="su_medium",
            on_change=_su_refresh_fields,
            help="Frequency and wavenumber always refer to vacuum. Air means "
            "dry standard air (15 °C, 101 325 Pa), the convention of NIST "
            "line lists; below 200 nm only vacuum applies.",
        )
        st.caption(
            "Pages 1–2 and the Neon reference use air wavelengths; laser "
            "and astronomical lines are listed in air too."
        )

    # Spectrometer Analysis params
    elif page == "Spectrometer Analysis":
        st.subheader("Data File")
        uploaded_file = st.file_uploader(
            "Choose a data file",
            type=["pkl", "pickle"],
            help="Upload a pickled matplotlib sensor-population figure (.pkl)",
        )
        if uploaded_file is not None:
            st.success(f"Loaded: **{uploaded_file.name}**")
            st.caption(f"{uploaded_file.size:,} bytes")
            # Auto-extract peak pixels from the pickled figure's legend. The
            # theme is part of the key so the rendered PNG is rebuilt when the
            # palette changes, instead of keeping the previous theme's colours.
            _pkl_id = f"{uploaded_file.name}_{uploaded_file.size}_{THEME}"
            if st.session_state.get("_fa_pkl_id") != _pkl_id:
                try:
                    _px_from_legend = ac.extract_pixels_from_legend(
                        uploaded_file
                    )
                    st.session_state["_fa_extracted_pixels"] = _px_from_legend
                except Exception:
                    st.session_state["_fa_extracted_pixels"] = []
                # Render raw pickle to PNG for reference display
                try:
                    _raw_fig = ac._load_pickle_fig(uploaded_file)
                    _raw_fig.set_size_inches(16, 10)
                    for _ax in _raw_fig.axes:
                        _ax.set_facecolor(PLOT_BG)
                        _ax.tick_params(colors=PLOT_AXIS, which="both")
                        for _sp in _ax.spines.values():
                            _sp.set_color(PLOT_AXIS)
                        _ax.xaxis.label.set_color(PLOT_AXIS)
                        _ax.yaxis.label.set_color(PLOT_AXIS)
                        _ax.title.set_color(PLOT_AXIS)
                        _leg = _ax.get_legend()
                        if _leg:
                            _leg.get_frame().set_facecolor(PLOT_BG)
                            for _lt in _leg.get_texts():
                                _lt.set_color(PLOT_AXIS)
                    _raw_fig.set_facecolor(PLOT_BG)
                    _raw_fig.tight_layout()
                    _raw_buf = io.BytesIO()
                    _raw_fig.savefig(
                        _raw_buf, format="png", dpi=110, bbox_inches="tight"
                    )
                    _raw_buf.seek(0)
                    st.session_state["fa_raw_pkl_png"] = _raw_buf.getvalue()
                    plt.close(_raw_fig)
                except Exception:
                    st.session_state.pop("fa_raw_pkl_png", None)
                st.session_state["_fa_pkl_id"] = _pkl_id
        else:
            st.info("No file uploaded yet")

        _auto_pixels = st.session_state.get("_fa_extracted_pixels", [])
        if _auto_pixels:
            st.caption(
                "Detected peaks (from legend): "
                + ", ".join(
                    str(int(p)) if p == int(p) else str(p)
                    for p in _auto_pixels
                )
            )

        st.markdown(
            """
            <style>
            div[data-testid="stSidebarContent"] div[data-testid="stButton"] > button {
                display: block;
                margin: 0.4rem auto 0.2rem auto;
                font-size: 1.15rem;
                font-weight: 700;
                border: 2px solid var(--greg-accent);
                padding: 0.45rem 2rem;
                width: 80%;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        run_analysis_btn = st.button(
            "Run Analysis",
            key="run_analysis_fa",
            disabled=uploaded_file is None,
            use_container_width=True,
        )

        st.subheader("Analysis Parameters")
        ref_wavelengths_str = st.text_input(
            "Reference wavelengths (nm)",
            value="638.299, 640.225",
            help="Two Neon wavelengths whose pixel positions you know with certainty",
            key="ref_wavelengths_str",
        )
        ref_pixels_str = st.text_input(
            "Reference pixels",
            value="16, 33",
            help="Pixel positions matching the reference wavelengths above (same order)",
            key="ref_pixels_str",
        )
        tolerance_px_fa = st.number_input(
            "Matching tolerance (px)",
            min_value=0.5,
            max_value=50.0,
            value=5.0,
            step=0.5,
            key="tolerance_px_fa",
            help=(
                "Maximum allowed distance (in pixels) between a detected peak "
                "and its predicted NIST position for a match to be accepted. "
                "Too large risks false matches; too small rejects real peaks "
                "if the seed calibration is slightly off."
            ),
        )
        full_sensor_fa = st.checkbox(
            "Full sensor (512 px, else 256 px)",
            value=False,
            key="full_sensor_fa",
        )

        # Parse inputs and validate
        _dp_valid = _rw_valid = _rp_valid = True
        _detected_pixels = _auto_pixels
        if len(_detected_pixels) < 2:
            _dp_valid = False
        try:
            _ref_wavelengths = [
                float(x.strip())
                for x in ref_wavelengths_str.split(",")
                if x.strip()
            ]
            if len(_ref_wavelengths) != 2:
                _rw_valid = False
        except ValueError:
            _rw_valid = False
        try:
            _ref_pixels = [
                float(x.strip())
                for x in ref_pixels_str.split(",")
                if x.strip()
            ]
            if len(_ref_pixels) != 2:
                _rp_valid = False
        except ValueError:
            _rp_valid = False

        if not _dp_valid:
            st.caption(
                "\u26a0 Could not read \u22652 peak pixels from the figure legend"
            )
        if not _rw_valid:
            st.caption(
                "\u26a0 Reference wavelengths: exactly 2 values required"
            )
        if not _rp_valid:
            st.caption("\u26a0 Reference pixels: exactly 2 values required")


# ============================================================================
# PAGE 1: OUTPUT ANGLE
# ============================================================================
if page == "Output Angle":
    st.markdown(
        """
        <div style="margin-bottom: 1.75rem;">
            <h1 style="font-size: 2.25rem; font-weight: 900; color: var(--greg-text);
                       letter-spacing: -0.03em; margin-bottom: 0;">
                Output Angle Analysis
            </h1>
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-top: 0.6rem;">
                <span style="height: 1px; width: 2.5rem; background: var(--greg-accent);
                             display: inline-block;"></span>
                <span style="font-size: 0.65rem; color: var(--greg-muted);
                             letter-spacing: 0.2em; text-transform: uppercase;
                             font-weight: 700;">output vs incident angle analysis</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        r"**Grating equation:** $m\lambda = d(\sin\alpha + \sin\beta)$"
    )

    beta_current = None
    budget_oa = None
    try:
        beta_current = pc.output_angle(
            order_m, wavelength_nm, int(lines_per_mm), alpha_deg
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.success(f"Current output angle β = **{beta_current:.2f}°**")
        try:
            nm_per_pix = pc.nm_per_pixel(
                p=pixel_pitch_um * 1e-6,
                lines_per_mm=lines_per_mm,
                alpha_deg=alpha_deg,
                lambda_nm=wavelength_nm,
                m=order_m,
                f=focal_length_mm * 1e-3,
                lens_axis=optics_oa["lens_axis"],
            )
            with c2:
                st.info(f"Spectral sampling: **{nm_per_pix:.4f} nm/pixel**")
            with c3:
                st.info(f"Spectral range: **{nm_per_pix * camera_pixels:.2f} nm**")
            budget_oa = pc.line_width_budget(
                p=pixel_pitch_um * 1e-6,
                lines_per_mm=lines_per_mm,
                alpha_deg=alpha_deg,
                lambda_nm=wavelength_nm,
                m=order_m,
                f=focal_length_mm * 1e-3,
                **optics_oa,
            )
            _show_resolution(budget_oa, "Output Angle")
        except Exception as e:
            st.warning(f"Could not calculate nm/pixel: {e}")
    except ValueError as e:
        st.error(str(e))

    alphas = np.arange(alpha_min, alpha_max + alpha_step / 2, alpha_step)
    betas, nm_per_pixel_values = [], []

    for a in alphas:
        try:
            b = pc.output_angle(order_m, wavelength_nm, int(lines_per_mm), a)
            betas.append(b)
            try:
                nm_pp = pc.nm_per_pixel(
                    p=pixel_pitch_um * 1e-6,
                    lines_per_mm=lines_per_mm,
                    alpha_deg=a,
                    lambda_nm=wavelength_nm,
                    m=order_m,
                    f=focal_length_mm * 1e-3,
                    lens_axis=optics_oa["lens_axis"],
                )
                nm_per_pixel_values.append(nm_pp)
            except:
                nm_per_pixel_values.append(np.nan)
        except:
            betas.append(np.nan)
            nm_per_pixel_values.append(np.nan)

    with plt.rc_context({"font.size": 13}):
        fig = Figure(figsize=(10, 4.5), dpi=360, facecolor=PLOT_BG)
        ax = fig.add_subplot(111, facecolor=PLOT_SURFACE)
        ax.tick_params(colors=PLOT_AXIS, which="both", labelsize=13)
        for spine in ax.spines.values():
            spine.set_color("white")

        scatter = ax.scatter(
            alphas,
            betas,
            c=nm_per_pixel_values,
            cmap=colormap,
            s=marker_size,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
        )

        if beta_current is not None:
            ax.scatter(
                [alpha_deg],
                [beta_current],
                color="#FFD10F",
                s=marker_size * 3,
                marker="*",
                edgecolors="#F2B705",
                linewidths=1.5,
                zorder=10,
                label=f"Current: α={alpha_deg}°, β={beta_current:.2f}°",
            )

        if show_littrow:
            try:
                alpha_littrow = pc.littrow_config(
                    order_m, wavelength_nm, int(lines_per_mm)
                )
                if alpha_min <= alpha_littrow <= alpha_max:
                    ax.axvline(
                        alpha_littrow,
                        color="orange",
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.7,
                    )
                    ax.axhline(
                        alpha_littrow,
                        color="orange",
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.7,
                    )
                    ax.text(
                        alpha_littrow,
                        ax.get_ylim()[1] * 0.95,
                        f"Littrow\nα={alpha_littrow:.1f}°",
                        ha="center",
                        va="top",
                        fontsize=13,
                        color="#111111",
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.5",
                            facecolor="orange",
                            alpha=0.85,
                        ),
                    )
            except:
                pass

        ax.set_xlabel("Incident Angle α (°)", fontsize=13, color=PLOT_AXIS)
        ax.set_ylabel("Output Angle β (°)", fontsize=13, color=PLOT_AXIS)
        ax.set_title(
            f"Output Angle vs Incident Angle  ·  {int(lines_per_mm)} lines/mm"
            f"  ·  λ={wavelength_nm:.0f} nm  ·  m={order_m}",
            fontsize=14,
            fontweight="bold",
            color=PLOT_AXIS,
        )
        ax.grid(True, alpha=0.3, linestyle="--", color=PLOT_GRID)
        ax.legend(loc="best", fontsize=13)

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("nm/pixel", color=PLOT_AXIS, fontsize=13)
        cbar.ax.tick_params(labelsize=13, colors=PLOT_AXIS)
        cbar.outline.set_edgecolor(PLOT_AXIS)

        st.pyplot(fig, use_container_width=True)

        try:
            _littrow_export = pc.littrow_config(
                order_m, wavelength_nm, int(lines_per_mm)
            )
        except ValueError:
            _littrow_export = None
        _data_download_button(
            tables={
                "output_angle": {
                    "description": "Output angle and spectral sampling for "
                    "each incident angle in the plotted range (NaN where no "
                    "diffracted order exists).",
                    "columns": {
                        "alpha_deg": alphas,
                        "beta_deg": betas,
                        "nm_per_pixel": nm_per_pixel_values,
                    },
                },
            },
            meta={
                "page": "Output Angle",
                "inputs": {
                    "camera": selected_camera,
                    "pixel_pitch_um": pixel_pitch_um,
                    "camera_pixels": camera_pixels,
                    "lines_per_mm": lines_per_mm,
                    "alpha_deg": alpha_deg,
                    "wavelength_nm": wavelength_nm,
                    "order_m": order_m,
                    "focal_length_mm": focal_length_mm,
                    **beam_source_oa,
                    **optics_oa,
                    "alpha_min_deg": alpha_min,
                    "alpha_max_deg": alpha_max,
                    "alpha_step_deg": alpha_step,
                },
                "results": {
                    "beta_deg": beta_current,
                    "littrow_alpha_deg": _littrow_export,
                    "resolution": budget_oa,
                },
            },
            stem="output_angle",
            key="download_data_oa",
        )

        st.divider()
        st.markdown(
            "<span style='font-size:0.75rem;letter-spacing:0.12em;"
            "text-transform:uppercase;font-weight:700;color:var(--greg-muted);'>"
            "Geometry diagram</span>",
            unsafe_allow_html=True,
        )
        _diag = _make_grating_diagram(alpha_deg, lines_per_mm, wavelength_nm, order_m)
        st.pyplot(_diag, use_container_width=True)
        plt.close(_diag)


# ============================================================================
# PAGE 2: SAMPLING SWEEP
# ============================================================================
elif page == "Sampling Sweep":
    st.markdown(
        """
        <div style="margin-bottom: 1.75rem;">
            <h1 style="font-size: 2.25rem; font-weight: 900; color: var(--greg-text);
                       letter-spacing: -0.03em; margin-bottom: 0;">
                Sampling Sweep
            </h1>
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-top: 0.6rem;">
                <span style="height: 1px; width: 2.5rem; background: var(--greg-accent);
                             display: inline-block;"></span>
                <span style="font-size: 0.65rem; color: var(--greg-muted);
                             letter-spacing: 0.2em; text-transform: uppercase;
                             font-weight: 700;">nm/pixel parametric analysis</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _cos_power = "" if optics_sw["lens_axis"] == pc.LENS_AXIS_BEAM else "^3"
    st.markdown(
        r"**Spectral sampling:** $\frac{\Delta\lambda}{\Delta x} = "
        rf"\frac{{pd\cos{_cos_power}\beta}}{{mf}}$"
    )

    def _budget_sw(**overrides):
        """line_width_budget at the sidebar settings, with some overridden."""
        params = dict(
            p=pixel_pitch_um * 1e-6,
            lines_per_mm=lines_per_mm_sw,
            alpha_deg=alpha_deg_sw,
            lambda_nm=wavelength_nm_sw,
            m=order_m_sw,
            f=focal_length_mm_sw * 1e-3,
            **optics_sw,
        )
        params.update(overrides)
        return pc.line_width_budget(**params)

    current_nm_per_pix = None
    budget_sw = None
    try:
        budget_sw = _budget_sw()
        current_nm_per_pix = budget_sw["nm_per_pixel"]
        _sw_c1, _sw_c2 = st.columns(2)
        with _sw_c1:
            st.success(f"Current: **{current_nm_per_pix:.4f} nm/pixel**")
        with _sw_c2:
            st.info(f"Spectral range: **{current_nm_per_pix * camera_pixels:.2f} nm**")
        _show_resolution(budget_sw, "Sampling Sweep")
    except (ValueError, ZeroDivisionError) as e:
        st.error(str(e))

    sweep_values = np.linspace(sweep_start, sweep_stop, int(sweep_points))
    _sweep_override = {
        "f (mm)": lambda v: {"f": v * 1e-3},
        "lines/mm": lambda v: {"lines_per_mm": v},
        "λ (nm)": lambda v: {"lambda_nm": v},
        "α (deg)": lambda v: {"alpha_deg": v},
    }[sweep_param]
    _sweep_keys = ("nm_per_pixel", "fwhm_nm", "sampling_limit_nm",
                   "resolution_nm", "resolving_power", "sigma_optics_nm",
                   "sigma_angular_nm", "sigma_pixel_nm", "sigma_extra_nm",
                   "sigma_total_nm")
    sweep_budget = {k: [] for k in _sweep_keys}
    for val in sweep_values:
        try:
            _b = _budget_sw(**_sweep_override(val))
        except (ValueError, ZeroDivisionError):  # no such order here
            _b = {}
        for k in _sweep_keys:
            sweep_budget[k].append(_b.get(k, np.nan))
    y_values = sweep_budget["nm_per_pixel"]

    with plt.rc_context({"font.size": 13}):
        fig = Figure(figsize=(10, 4.5), dpi=360, facecolor=PLOT_BG)
        ax = fig.add_subplot(111, facecolor=PLOT_SURFACE)
        ax.tick_params(colors=PLOT_AXIS, which="both", labelsize=13)
        for spine in ax.spines.values():
            spine.set_color("white")

        ax.plot(sweep_values, y_values, linewidth=1, marker="o", markersize=4)

        try:
            if current_nm_per_pix is not None:
                if sweep_param == "f (mm)":
                    current_x = focal_length_mm_sw
                elif sweep_param == "lines/mm":
                    current_x = lines_per_mm_sw
                elif sweep_param == "λ (nm)":
                    current_x = wavelength_nm_sw
                else:
                    current_x = alpha_deg_sw

                if sweep_start <= current_x <= sweep_stop:
                    ax.scatter(
                        [current_x],
                        [current_nm_per_pix],
                        color="#FFD10F",
                        s=150,
                        marker="*",
                        edgecolors="#F2B705",
                        linewidths=1.5,
                        zorder=10,
                        label="Current configuration",
                    )
        except:
            pass

        ax.set_xlabel(sweep_param, fontsize=13, color=PLOT_AXIS)
        ax.set_ylabel("nm/pixel", fontsize=13, color=PLOT_AXIS)
        ax.set_title(
            f"Spectral Sampling vs {sweep_param}  ·  {int(lines_per_mm_sw)} lines/mm"
            f"  ·  α={alpha_deg_sw}°  ·  λ={wavelength_nm_sw:.0f} nm  ·  f={focal_length_mm_sw:.0f} mm",
            fontsize=14,
            fontweight="bold",
            color=PLOT_AXIS,
        )
        ax.grid(True, alpha=0.3, linestyle="--", color=PLOT_GRID)
        ax.legend(loc="best", fontsize=13)

        st.pyplot(fig, use_container_width=True)

        # ── Resolution over the same sweep
        fig_r = Figure(figsize=(10, 4.5), dpi=360, facecolor=PLOT_BG)
        ax_r = fig_r.add_subplot(111, facecolor=PLOT_SURFACE)
        ax_r.tick_params(colors=PLOT_AXIS, which="both", labelsize=13)
        for spine in ax_r.spines.values():
            spine.set_color(PLOT_AXIS)
        ax_r.plot(sweep_values, sweep_budget["resolution_nm"], color=ACCENT,
                  linewidth=3, label="Resolution (larger of the two)")
        ax_r.plot(sweep_values, sweep_budget["fwhm_nm"], color=PLOT_AXIS,
                  linewidth=1.2, linestyle="--", label="Line FWHM (model)")
        ax_r.plot(sweep_values, sweep_budget["sampling_limit_nm"],
                  color=TEXT_MUTED, linewidth=1.2, linestyle=":",
                  label="Sampling limit (2 px)")
        if budget_sw is not None and sweep_start <= current_x <= sweep_stop:
            ax_r.scatter([current_x], [budget_sw["resolution_nm"]],
                         color="#FFD10F", s=150, marker="*",
                         edgecolors="#F2B705", linewidths=1.5, zorder=10,
                         label="Current configuration")
        ax_r.set_xlabel(sweep_param, fontsize=13, color=PLOT_AXIS)
        ax_r.set_ylabel("Δλ (nm)", fontsize=13, color=PLOT_AXIS)
        ax_r.set_title(f"Spectral Resolution vs {sweep_param} (experimental)",
                       fontsize=14,
                       fontweight="bold", color=PLOT_AXIS)
        ax_r.grid(True, alpha=0.3, linestyle="--", color=PLOT_GRID)
        ax_r.legend(loc="best", fontsize=12)
        st.pyplot(fig_r, use_container_width=True)

        _sweep_column = {
            "f (mm)": "focal_length_mm",
            "lines/mm": "lines_per_mm",
            "λ (nm)": "wavelength_nm",
            "α (deg)": "alpha_deg",
        }[sweep_param]
        _data_download_button(
            tables={
                "sampling_sweep": {
                    "description": f"Spectral sampling and modelled "
                    f"resolution vs {_sweep_column}, all other inputs held at "
                    "their run.json values (NaN where no diffracted order "
                    "exists). resolution_nm is the larger of fwhm_nm and "
                    "sampling_limit_nm (2 px).",
                    "columns": {
                        _sweep_column: sweep_values,
                        **sweep_budget,
                    },
                },
            },
            meta={
                "page": "Sampling Sweep",
                "inputs": {
                    "camera": selected_camera,
                    "pixel_pitch_um": pixel_pitch_um,
                    "camera_pixels": camera_pixels,
                    "lines_per_mm": lines_per_mm_sw,
                    "alpha_deg": alpha_deg_sw,
                    "wavelength_nm": wavelength_nm_sw,
                    "order_m": order_m_sw,
                    "focal_length_mm": focal_length_mm_sw,
                    **beam_source_sw,
                    **optics_sw,
                    "sweep_parameter": _sweep_column,
                    "sweep_start": sweep_start,
                    "sweep_stop": sweep_stop,
                    "sweep_points": sweep_points,
                },
                "results": {
                    "nm_per_pixel": current_nm_per_pix,
                    "resolution": budget_sw,
                },
            },
            stem=f"sampling_sweep_{_sweep_column}",
            key="download_data_sw",
        )


# ============================================================================
# PAGE 3: SPECTROMETER ANALYSIS
# ============================================================================
elif page == "Spectrometer Analysis":
    st.markdown(
        """
        <div style="margin-bottom: 1.75rem;">
            <h1 style="font-size: 2.25rem; font-weight: 900; color: var(--greg-text);
                       letter-spacing: -0.03em; margin-bottom: 0;">
                Spectrometer Analysis
            </h1>
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-top: 0.6rem;">
                <span style="height: 1px; width: 2.5rem; background: var(--greg-accent);
                             display: inline-block;"></span>
                <span style="font-size: 0.65rem; color: var(--greg-muted);
                             letter-spacing: 0.2em; text-transform: uppercase;
                             font-weight: 700;">spectral sampling, linearity and stability, resolution</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Raw pickle reference figure ──────────────────────────────────────────
    if "fa_raw_pkl_png" in st.session_state and uploaded_file is not None:
        st.markdown(
            """
            <div style="margin-bottom:0.25rem;">
                <span style="font-size:0.6rem;color:var(--greg-muted);
                             letter-spacing:0.15em;text-transform:uppercase;
                             font-weight:700;">Reference</span>
                <span style="font-size:0.95rem;font-weight:700;
                             color:var(--greg-text);margin-left:0.5rem;">Uploaded Figure</span>
            </div>
            <div style="font-size:0.72rem;color:var(--greg-muted);margin-bottom:0.5rem;">
                Use pixel positions from this plot as reference pixels in the sidebar
            </div>
            """,
            unsafe_allow_html=True,
        )
        _c_left, _col_ref, _c_right = st.columns([1, 3, 1])
        with _col_ref:
            st.image(
                st.session_state["fa_raw_pkl_png"], use_container_width=True
            )
        st.divider()

    # ── Run analysis on button click ─────────────────────────────────────────
    # The figures are cached as PNG bytes, so their colours are fixed at render
    # time. A theme change therefore has to redraw them, which means re-running
    # the analysis \u2014 there is no cheaper route back to a recoloured figure.
    _stale_theme = (
        "fa_figs_png" in st.session_state
        and st.session_state.get("fa_theme") != THEME
    )
    if (run_analysis_btn or _stale_theme) and uploaded_file is not None:
        with st.spinner("Running spectrometer analysis\u2026"):
            try:
                results = ac.run_neon_analysis(
                    pkl_file=uploaded_file,
                    detected_pixels=_detected_pixels,
                    ref_wavelengths_nm=_ref_wavelengths,
                    ref_pixels=_ref_pixels,
                    tolerance_px=tolerance_px_fa,
                    full_sensor=full_sensor_fa,
                )
                # Serialise figures to PNG bytes for session persistence
                _plot_errors = results.get("plot_errors", {})
                figs_png = {}
                for key, fig in results["figs"].items():
                    if fig is None:
                        _err_msg = _plot_errors.get(key, "Unknown error")
                        _ph = plt.Figure(figsize=(16, 10), facecolor=PLOT_BG)
                        _ph_ax = _ph.add_subplot(111, facecolor=PLOT_BG)
                        _ph_ax.set_axis_off()
                        _ph_ax.text(
                            0.5,
                            0.5,
                            f"Plot {key} could not be generated:\n\n{_err_msg}",
                            ha="center",
                            va="center",
                            fontsize=13,
                            color=TEXT,
                            transform=_ph_ax.transAxes,
                            wrap=True,
                        )
                        _pbuf = io.BytesIO()
                        _ph.savefig(
                            _pbuf, format="png", dpi=110, bbox_inches="tight"
                        )
                        _pbuf.seek(0)
                        figs_png[key] = _pbuf.getvalue()
                        plt.close(_ph)
                    else:
                        buf = io.BytesIO()
                        fig.savefig(
                            buf, format="png", dpi=110, bbox_inches="tight"
                        )
                        buf.seek(0)
                        figs_png[key] = buf.getvalue()
                        plt.close(fig)
                st.session_state["fa_figs_png"] = figs_png
                st.session_state["fa_theme"] = THEME
                st.session_state["fa_plot_errors"] = _plot_errors
                st.session_state["fa_data"] = {
                    "matched": results["matched"],
                    "unmatched_pixels": results["unmatched_pixels"],
                    "unmatched_lines": results["unmatched_lines"],
                    "calibration": results["calibration"],
                    "ideal_pixels_used": results["ideal_pixels_used"],
                    "gauss_sigmas_px": results["gauss_fit"]["sigmas_px"],
                    "stats": results["stats"],
                    "sensor_max": 511 if full_sensor_fa else 255,
                    "peaks": results["peaks"],
                    "tables": results["tables"],
                    "export_errors": results["export_errors"],
                    "gauss_popt": results["gauss_fit"]["popt"],
                    "gauss_perr": results["gauss_fit"]["perr"],
                    # Snapshot of the inputs this run used: the sidebar may
                    # change before the data is downloaded.
                    "inputs": {
                        "source_file": uploaded_file.name,
                        "detected_pixels": list(_detected_pixels),
                        "ref_wavelengths_nm": list(_ref_wavelengths),
                        "ref_pixels": list(_ref_pixels),
                        "tolerance_px": tolerance_px_fa,
                        "full_sensor": full_sensor_fa,
                    },
                }
                st.session_state.pop("fa_error", None)
            except Exception as _e:
                st.session_state["fa_error"] = str(_e)
                st.session_state.pop("fa_figs_png", None)
                st.session_state.pop("fa_theme", None)
                st.session_state.pop("fa_data", None)

    # ── Display error if present ─────────────────────────────────────────────
    if "fa_error" in st.session_state:
        st.error(f"Analysis error: {st.session_state['fa_error']}")

    # ── Display results ──────────────────────────────────────────────────────
    if "fa_figs_png" in st.session_state and "fa_data" in st.session_state:
        figs_png = st.session_state["fa_figs_png"]
        data = st.session_state["fa_data"]
        stats = data["stats"]
        cal = data["calibration"]
        matched = data["matched"]

        # ── Summary metrics
        peaks = data["peaks"]
        n_predicted = sum(r["type"] == ac.PEAK_PREDICTED for r in peaks)

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("Detected & matched", len(matched))
        with mc2:
            st.metric("Detected, no match", len(data["unmatched_pixels"]))
        with mc3:
            st.metric("Expected, not found", n_predicted)

        st.divider()

        # ── Combined peaks table
        st.markdown(
            "<span style='font-size:0.75rem;letter-spacing:0.12em;"
            "text-transform:uppercase;font-weight:700;color:var(--greg-muted);'>"
            "All peaks in range</span>",
            unsafe_allow_html=True,
        )

        _TABLE_DET = ACCENT  # teal   — detected & matched to NIST
        _TABLE_DET_NM = "#FFD10F"  # yellow — detected, no NIST match
        _TABLE_PRED = "#FF8C42"  # orange — NIST line expected but not found

        _TYPE_STYLE = {
            ac.PEAK_DETECTED: ("Detected", _TABLE_DET),
            ac.PEAK_NO_MATCH: ("Detected — no match", _TABLE_DET_NM),
            ac.PEAK_PREDICTED: ("Predicted (not found)", _TABLE_PRED),
        }

        def _num(v, fmt):
            return "—" if np.isnan(v) else format(v, fmt)

        _tbl_rows = [
            (
                r["pixel"],
                f"{r['pixel']:.1f}",
                f"{r['wavelength_nm']:.4f}",
                *_TYPE_STYLE[r["type"]],
                _num(r["nist_intensity"], ".0f"),
                _num(r["delta_px"], "+.2f"),
            )
            for r in peaks
        ]

        _th = (
            "background:var(--greg-surface);color:var(--greg-muted);font-size:0.72rem;"
            "font-weight:700;letter-spacing:0.1em;text-transform:uppercase;"
            "padding:0.4rem 0.8rem;border-bottom:1px solid var(--greg-border);"
            "text-align:left;"
        )
        _td = "padding:0.28rem 0.8rem;font-size:0.84rem;border-bottom:1px solid var(--greg-border);"
        _html_rows = "".join(
            f"<tr>"
            f"<td style='{_td}color:var(--greg-muted);'>{i}</td>"
            f"<td style='{_td}color:{color};'>{px_str}</td>"
            f"<td style='{_td}color:{color};'>{wl_str}</td>"
            f"<td style='{_td}color:{color};font-size:0.72rem;'>{label}</td>"
            f"<td style='{_td}color:var(--greg-muted);text-align:right;'>{intens_str}</td>"
            f"<td style='{_td}color:var(--greg-muted);text-align:right;'>{delta_str}</td>"
            f"</tr>"
            for i, (
                _,
                px_str,
                wl_str,
                label,
                color,
                intens_str,
                delta_str,
            ) in enumerate(_tbl_rows, 1)
        )
        st.markdown(
            f"<table style='border-collapse:collapse;width:100%;'>"
            f"<thead><tr>"
            f"<th style='{_th}'>#</th>"
            f"<th style='{_th}'>Pixel</th>"
            f"<th style='{_th}'>Wavelength (nm)</th>"
            f"<th style='{_th}'>Type</th>"
            f"<th style='{_th}text-align:right;'>NIST intensity</th>"
            f"<th style='{_th}text-align:right;cursor:help;' "
            f"title='Residual between the detected pixel and its predicted position "
            f"from the final calibration. Only available for matched peaks. "
            f"Positive = detected peak is to the right of the prediction.'>Δ (px)</th>"
            f"</tr></thead><tbody>{_html_rows}</tbody></table>",
            unsafe_allow_html=True,
        )

        # Six plots in a 2-column grid
        _plot_meta = [
            (
                "A",
                "Scale between neighbors",
                "nm/px scale overlaid on sensor population data",
                None,
            ),
            ("B", "Linearity", "Spectrometer stability", None),
            (
                "C",
                "Scale for all combinations",
                "nm/px for every peak pair, coloured by first peak",
                None,
            ),
            (
                "D",
                "Scale vs Distance",
                "nm/px as a function of distance between peaks",
                None,
            ),
            (
                "E",
                "Ideal Spectrum Overlay",
                "Measured spectrum vs NIST values",
                "The observed line intensities depend on the Ne source and how it is "
                "powered (current, pressure, temperature), so the relative amplitudes "
                "will differ from the NIST tabulated values. Only the peak positions "
                "are used for calibration.",
            ),
            (
                "F",
                "Gaussian Fit",
                "Gaussian fit of the measured spectral lines",
                None,
            ),
        ]
        col_left, col_right = st.columns(2)
        for idx, (key, title, caption, note) in enumerate(_plot_meta):
            col = col_left if idx % 2 == 0 else col_right
            with col:
                _note_html = ""
                if note:
                    _note_html = (
                        f"<style>"
                        f".tt-{key}{{position:relative;display:inline-block;"
                        f"cursor:help;margin-left:0.4rem;font-size:0.8rem;"
                        f"color:var(--greg-muted);}}"
                        f".tt-{key} .tt-box{{visibility:hidden;background:var(--greg-surface);"
                        f"color:var(--greg-text);text-align:left;border-radius:4px;"
                        f"padding:8px 10px;position:absolute;z-index:9999;"
                        f"bottom:130%;left:50%;transform:translateX(-50%);"
                        f"width:280px;font-size:0.72rem;line-height:1.5;"
                        f"border:1px solid var(--greg-border);pointer-events:none;}}"
                        f".tt-{key}:hover .tt-box{{visibility:visible;}}"
                        f"</style>"
                        f'<span class="tt-{key}">ⓘ'
                        f'<span class="tt-box">{note}</span>'
                        f"</span>"
                    )
                st.markdown(
                    f"""
                    <div style="margin-bottom:0.25rem;">
                        <span style="font-size:0.6rem;color:var(--greg-muted);
                                     letter-spacing:0.15em;text-transform:uppercase;
                                     font-weight:700;">Plot {key}</span>
                        <span style="font-size:0.95rem;font-weight:700;
                                     color:var(--greg-text);margin-left:0.5rem;">{title}</span>
                        {_note_html}
                    </div>
                    <div style="font-size:0.72rem;color:var(--greg-muted);
                                margin-bottom:0.5rem;">{caption}</div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(figs_png[key], use_container_width=True)

        # ── Download all plots as a single PNG grid
        _imgs = [Image.open(io.BytesIO(figs_png[k])) for k, *_ in _plot_meta]
        _w = max(i.width for i in _imgs)
        _h = max(i.height for i in _imgs)
        _grid = Image.new("RGB", (_w * 2, _h * 3), GRID_RGB)
        for _i, _img in enumerate(_imgs):
            _grid.paste(_img, ((_i % 2) * _w, (_i // 2) * _h))
        _buf = io.BytesIO()
        _grid.save(_buf, format="PNG")
        _inputs = data["inputs"]
        _stem = _inputs["source_file"].rsplit(".", 1)[0]
        _dl_png, _dl_data = st.columns(2)
        with _dl_png:
            st.download_button(
                "Download plots as PNG",
                data=_buf.getvalue(),
                file_name=f"{_stem}_greg.png",
                mime="image/png",
                on_click="ignore",
            )
        with _dl_data:
            _data_download_button(
                tables=data["tables"],
                meta={
                    "page": "Spectrometer Analysis",
                    "inputs": _inputs,
                    "results": {
                        "calibration": cal,
                        "stats": stats,
                        "ideal_pixels_used": data["ideal_pixels_used"],
                        "gaussian_fit": {
                            "sigmas_px": data["gauss_sigmas_px"],
                            "popt": data["gauss_popt"],
                            "perr": data["gauss_perr"],
                            "popt_layout": "[A1, mu1, sigma1, ..., offset]",
                        },
                    },
                    "plot_errors": st.session_state.get("fa_plot_errors", {}),
                    "export_errors": data["export_errors"],
                },
                stem=_stem,
                key="download_data_fa",
            )
        if data["export_errors"]:
            st.caption(
                "Left out of the data download: "
                + "; ".join(
                    f"{name} ({msg})"
                    for name, msg in data["export_errors"].items()
                )
            )

    elif uploaded_file is None:
        st.info(
            "Upload a pickled sensor-population figure (.pkl) in the sidebar, "
            "then fill in the analysis parameters and click **Run Analysis**."
        )
    else:
        st.info(
            "Set the analysis parameters in the sidebar and click **Run Analysis**."
        )


# ============================================================================
# PAGE 4: SPECTRAL UNITS (ported from Prismatic)
# ============================================================================
else:
    _su_restore()
    ss = st.session_state
    st.markdown(
        """
        <div style="margin-bottom: 1.75rem;">
            <h1 style="font-size: 2.25rem; font-weight: 900; color: var(--greg-text);
                       letter-spacing: -0.03em; margin-bottom: 0;">
                Spectral Units
            </h1>
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-top: 0.6rem;">
                <span style="height: 1px; width: 2.5rem; background: var(--greg-accent);
                             display: inline-block;"></span>
                <span style="font-size: 0.65rem; color: var(--greg-muted);
                             letter-spacing: 0.2em; text-transform: uppercase;
                             font-weight: 700;">wavelength · frequency · wavenumber · coherence</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    def _su_label(text: str) -> None:
        st.markdown(
            "<span style='font-size:0.75rem;letter-spacing:0.12em;"
            "text-transform:uppercase;font-weight:700;color:var(--greg-muted);'>"
            f"{text}</span>",
            unsafe_allow_html=True,
        )

    def _su_tiles(items, min_rem: float = 9.5) -> None:
        """Value tiles in a wrapping grid; st.metric cuts long values to '…'."""
        tiles = "".join(
            '<div style="background:var(--greg-surface);border:1px solid var(--greg-border);'
            'border-left:2px solid var(--greg-accent);border-radius:.5rem;'
            'padding:.7rem 1rem;min-width:0;">'
            f'<div style="font-size:.8rem;color:var(--greg-muted);margin-bottom:.25rem;">'
            f'{label}</div>'
            '<div style="font-size:1.3rem;line-height:1.25;color:var(--greg-text);'
            f'overflow-wrap:anywhere;">{value}</div></div>'
            for label, value in items
        )
        st.markdown(
            '<div style="display:grid;gap:.75rem;margin-bottom:.75rem;'
            f'grid-template-columns:repeat(auto-fit,minmax({min_rem}rem,1fr));">'
            f"{tiles}</div>",
            unsafe_allow_html=True,
        )

    # ── Three linked fields
    _medium_word = "air" if _su_air() else "vacuum"
    _fields = (
        (f"Wavelength ({_medium_word})", "su_wl", "su_wl_unit", su.WL_UNITS,
         su.WL_STEP, su.WL_FMT, _su_on_wl),
        ("Frequency", "su_freq", "su_freq_unit", su.FREQ_UNITS,
         su.FREQ_STEP, su.FREQ_FMT, _su_on_freq),
        ("Wavenumber (vacuum)", "su_wnum", "su_wnum_unit", su.WNUM_UNITS,
         su.WNUM_STEP, su.WNUM_FMT, _su_on_wnum),
    )
    for _col, (_label, _vkey, _ukey, _units, _steps, _fmts, _cb) in zip(
        st.columns(3), _fields
    ):
        with _col:
            _v_col, _u_col = st.columns([3, 2], vertical_alignment="bottom")
            _unit = ss[_ukey]
            with _v_col:
                st.number_input(_label, key=_vkey, on_change=_cb,
                                step=_steps[_unit], format=_fmts[_unit])
            with _u_col:
                st.selectbox(f"{_label} unit", list(_units), key=_ukey,
                             on_change=_su_refresh_fields,
                             label_visibility="collapsed")

    wl_vac = float(ss["su_vac_nm"])
    wl_air = su.vacuum_to_air(wl_vac)

    # ── Spectrum bar
    _VIS_LO, _VIS_HI = 380, 750
    _in_vis = 380 <= wl_vac <= 780
    _pct = max(2.0, min(98.0, (wl_vac - _VIS_LO) / (_VIS_HI - _VIS_LO) * 100))
    _shown = wl_air if _su_air() else wl_vac
    _wl_label = f"{_shown:.2f} nm" if _shown < 10_000 else f"{_shown / 1000:.3g} µm"
    _opacity = "1" if _in_vis else "0.3"
    st.markdown(
        f'<div style="background:var(--greg-surface);border:1px solid var(--greg-border);'
        f'border-radius:4px;padding:1.25rem 1.5rem 1.5rem;margin:.75rem 0;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'flex-wrap:wrap;gap:.5rem;margin-bottom:1.5rem;">'
        f'<span style="font-size:.65rem;font-weight:700;letter-spacing:.08em;'
        f'color:var(--greg-muted);text-transform:uppercase;">'
        f'Visible range 380–750 nm</span>'
        f'<span style="font-size:.6rem;font-weight:700;letter-spacing:.05em;'
        f'color:{"var(--greg-accent)" if _in_vis else "var(--greg-muted)"};">'
        f'{"VISIBLE" if _in_vis else "OUT OF VISIBLE RANGE"}</span></div>'
        f'<div style="position:relative;margin-top:2rem;">'
        f'<div style="position:absolute;left:{_pct}%;top:-26px;transform:translateX(-50%);'
        f'opacity:{_opacity};pointer-events:none;">'
        f'<span style="font-size:.78rem;color:var(--greg-text);font-weight:700;'
        f'white-space:nowrap;">{_wl_label}</span></div>'
        f'<div style="position:absolute;left:{_pct}%;top:-4px;width:2px;'
        f'height:calc(100% + 12px);transform:translateX(-50%);'
        f'background:var(--greg-text);opacity:{_opacity};z-index:2;"></div>'
        f'<div style="height:28px;border-radius:3px;background:linear-gradient(to right,'
        f'#590059 0%,#6600aa 5%,#5500ff 11%,#0000ff 16%,#0090ff 24%,#00ffff 29%,'
        f'#00ff7f 33%,#00ff00 35%,#7cff00 43%,#ffff00 54%,#ffaa00 60%,#ff6d00 65%,'
        f'#ff3500 69%,#ff1000 72%,#ff0000 75%,#ee0000 80%,#d80000 87%,#c20000 91%,'
        f'#a80000 96%,#8c0000 100%);"></div>'
        f'<div style="display:flex;justify-content:space-between;margin-top:.4rem;'
        f'font-size:.68rem;color:var(--greg-muted);">'
        f'<span>380 nm · Violet</span><span>565 nm · Green</span>'
        f'<span>750 nm · Red</span></div></div></div>',
        unsafe_allow_html=True,
    )

    # ── Colour swatch + derived quantities
    _rgb = su.wavelength_to_rgb(wl_vac)
    _region = su.spectral_region(wl_vac)
    _band = su.find_band(wl_vac, su.TELECOM_BANDS)
    _col_sw, _col_ph = st.columns([1, 4])
    with _col_sw:
        if _rgb:
            _hex = "#{:02X}{:02X}{:02X}".format(*_rgb)
            _swatch = (f'<div style="height:56px;border-radius:3px;background:{_hex};'
                       f'margin-bottom:.6rem;"></div>')
        else:
            _swatch = ('<div style="height:56px;border-radius:3px;margin-bottom:.6rem;'
                       'border:1px dashed var(--greg-border);display:flex;'
                       'align-items:center;justify-content:center;font-size:.6rem;'
                       'color:var(--greg-muted);letter-spacing:.05em;">INVISIBLE</div>')
        _band_html = (f'<div style="font-size:.65rem;color:var(--greg-accent);'
                      f'font-weight:700;margin-top:.3rem;">Telecom {_band}</div>'
                      if _band else "")
        st.markdown(
            f'<div style="background:var(--greg-surface);border:1px solid var(--greg-border);'
            f'border-radius:4px;padding:1rem;">{_swatch}'
            f'<div style="font-size:.85rem;font-weight:600;color:var(--greg-text);">'
            f'{_region}</div>{_band_html}</div>',
            unsafe_allow_html=True,
        )
    with _col_ph:
        _su_tiles([
            ("Photon energy", su.fmt_energy(su.photon_energy_ev(wl_vac))),
            ("Period", su.fmt_period(su.period_fs(wl_vac))),
            ("Angular frequency", su.fmt_omega(su.ang_freq_radfs(wl_vac))),
            ("Vacuum λ", f"{wl_vac:.4f} nm") if _su_air()
            else ("Air λ", f"{wl_air:.4f} nm"),
        ])
        _n_air = su.refractive_index_air(wl_vac)
        st.caption(
            f"Air − vacuum: {1e3 * (wl_air - wl_vac):+.1f} pm "
            f"(n_air − 1 = {(_n_air - 1) * 1e6:.2f} × 10⁻⁶). Below 200 nm only "
            "vacuum wavelengths apply."
        )

    # ── Send to the grating pages
    _can_send = WAVELENGTH_MIN_NM <= wl_air <= WAVELENGTH_MAX_NM
    _s1, _s2, _s3 = st.columns([1, 1, 2])
    for _scol, _target in zip((_s1, _s2), WAVELENGTH_KEYS):
        with _scol:
            st.button(
                f"Use in {_target}",
                key=f"su_send_{WAVELENGTH_KEYS[_target]}",
                on_click=_su_send,
                args=(_target,),
                disabled=not _can_send,
                use_container_width=True,
            )
    with _s3:
        st.caption(
            f"Sends the air wavelength {wl_air:.4f} nm and opens the page."
            if _can_send
            else f"Pages 1–2 accept {WAVELENGTH_MIN_NM:.0f}–"
            f"{WAVELENGTH_MAX_NM:.0f} nm."
        )

    st.divider()

    # ── Spectral width at this wavelength
    _su_label("Spectral width at this wavelength")
    _w1, _w2, _w3 = st.columns([2, 1, 3], vertical_alignment="bottom")
    with _w1:
        st.number_input("Width", min_value=0.0, key="su_width", format="%.6g")
    with _w2:
        st.selectbox("Width unit", list(su.WIDTH_UNITS), key="su_width_unit",
                     label_visibility="collapsed")
    with _w3:
        _last = ss.get("_last_resolution")
        if _last:
            st.button(
                f"Use resolution from {_last[1]} ({_fmt_nm(_last[0])})",
                key="su_use_resolution",
                on_click=_su_use_width,
                args=(_last[0],),
            )
    if ss["su_width"] > 0:
        _w = su.width_in(ss["su_width"], ss["su_width_unit"], wl_vac)
        _su_tiles([
            ("Δλ", _fmt_nm(_w["dl_nm"])),
            ("Δν", f"{_w['dnu_ghz']:.4g} GHz"),
            ("Δν̃", f"{_w['dwnum_cm']:.4g} cm⁻¹"),
            ("R = λ/Δλ", f"{_w['resolving_power']:,.0f}"),
            ("Coherence time", su.fmt_time(_w["tau_c_s"])),
            ("Coherence length", su.fmt_length(_w["l_c_m"])),
        ])
        st.caption(
            "Δν = c·Δλ/λ², Δν̃ = Δλ/λ² at the vacuum wavelength. Coherence "
            "time τc = 1/Δν = λ²/(c·Δλ) and length Lc = c·τc; the exact "
            "prefactor depends on the line shape (0.44/Δν for a Gaussian "
            "FWHM), so treat them as scales."
        )

    st.divider()

    # ── Reference data
    _su_label("Reference")
    _t_ne, _t_astro, _t_laser, _t_uv, _t_ir, _t_tel = st.tabs(
        ["Neon (NIST)", "Astronomical lines", "Laser lines", "Ultraviolet bands",
         "Infrared bands", "Telecom bands"]
    )

    def _line_buttons(lines, in_air: bool, prefix: str, ncols: int,
                      digits: "int | None" = None) -> None:
        cols = st.columns(ncols)
        for i, (name, lam, desc) in enumerate(lines):
            with cols[i % ncols]:
                st.button(
                    f"{name} · {lam:g} nm" if digits is None
                    else f"{name} · {lam:.{digits}f} nm",
                    key=f"su_ref_{prefix}_{i}",
                    on_click=_su_jump,
                    args=(lam, in_air),
                    help=desc,
                    use_container_width=True,
                )

    def _band_range(lo: float, hi: float) -> str:
        if lo >= 3000:
            return f"{lo / 1000:g}–{hi / 1000:g} µm"
        return f"{lo:g}–{hi:g} nm"

    def _band_rows(bands) -> None:
        for name, lo, hi, desc in bands:
            active = lo <= wl_vac < hi
            border = "var(--greg-accent)" if active else "var(--greg-border)"
            badge = (" <span style='font-size:.58rem;font-weight:700;"
                     "color:var(--greg-accent);'>ACTIVE</span>" if active else "")
            st.markdown(
                f'<div style="border:1px solid {border};border-radius:4px;'
                f'padding:.6rem 1rem;margin-bottom:.4rem;display:flex;flex-wrap:wrap;'
                f'gap:.5rem;justify-content:space-between;align-items:center;">'
                f'<div><span style="font-size:.8rem;font-weight:700;'
                f'color:var(--greg-text);">{name}</span>{badge}<br>'
                f'<span style="font-size:.65rem;color:var(--greg-muted);">'
                f'{_band_range(lo, hi)} (vacuum)</span></div>'
                f'<span style="font-size:.72rem;color:var(--greg-muted);'
                f'max-width:60%;text-align:right;">{desc}</span></div>',
                unsafe_allow_html=True,
            )

    with _t_ne:
        st.caption("Strong Ne I lines from NIST, air wavelengths (the "
                   "calibration reference used on page 3).")
        _line_buttons(su.neon_lines(10_000), True, "ne", 4, digits=3)
    with _t_astro:
        st.caption("Air wavelengths.")
        _line_buttons(su.SPECTRAL_LINES, True, "astro", 3)
    with _t_laser:
        st.caption("Nominal wavelengths as usually quoted (air).")
        _line_buttons(su.LASER_LINES, True, "laser", 3)
    with _t_uv:
        _band_rows(su.UV_BANDS)
    with _t_ir:
        _band_rows(su.IR_BANDS)
    with _t_tel:
        _band_rows(su.TELECOM_BANDS)

    _su_save()


# ============================================================================
# FOOTER
# ============================================================================
st.markdown(
    """
    <div style="text-align: center; color: var(--greg-muted); padding: 2rem 0 1rem;
                font-size: 0.75rem; border-top: 1px solid var(--greg-border);
                margin-top: 2rem; opacity: 0.6; font-family: var(--greg-font-ui);">
        <strong style="color: var(--greg-text);">GREG</strong> &nbsp;·&nbsp;
        Grating Equation Generator
    </div>
    """,
    unsafe_allow_html=True,
)
