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
import physics_core as pc

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

# Camera options: display name → {pitch_um, pixels}
CAMERAS = {
    "LinoSPAD2 (26.2 µm)": {"pitch_um": 26.2, "pixels": 256},
    "Timepix3 (55 µm)": {"pitch_um": 55.0, "pixels": 256},
    "Moravian CMOS (3.45 µm)": {"pitch_um": 3.45, "pixels": 8336},
}


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
    pixel_pitch_um = _cam["pitch_um"]
    camera_pixels = _cam["pixels"]

    page = st.radio(
        "",
        ["Output Angle", "Sampling Sweep", "Spectrometer Analysis"],
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
            min_value=300.0,
            max_value=1000.0,
            value=650.0,
            step=1.0,
            key="wavelength_oa",
        )
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
        beam_width_mm = st.number_input(
            "Input beam diameter (mm)",
            min_value=1.0,
            max_value=200.0,
            value=3.0,
            step=1.0,
            key="beam_width_oa",
        )
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
            min_value=300.0,
            max_value=1000.0,
            value=650.0,
            step=1.0,
            key="wavelength_sw",
        )
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
        beam_width_mm_sw = st.number_input(
            "Input beam diameter (mm)",
            min_value=1.0,
            max_value=200.0,
            value=3.0,
            step=1.0,
            key="beam_width_sw",
        )
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

    # Spectrometer Analysis params
    else:
        st.subheader("Data File")
        uploaded_file = st.file_uploader(
            "Choose a data file",
            type=["pkl", "pickle"],
            help="Upload a pickled matplotlib sensor-population figure (.pkl)",
        )
        if uploaded_file is not None:
            st.success(f"Loaded: **{uploaded_file.name}**")
            st.caption(f"{uploaded_file.size:,} bytes")
            # Auto-extract peak pixels from the pickled figure's legend
            _pkl_id = f"{uploaded_file.name}_{uploaded_file.size}"
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
            )
            with c2:
                st.info(f"Spectral sampling: **{nm_per_pix:.4f} nm/pixel**")
            with c3:
                st.info(f"Spectral range: **{nm_per_pix * camera_pixels:.2f} nm**")
            grating_footprint_mm = beam_width_mm / np.cos(np.radians(alpha_deg))
            sensor_res_nm = 2.0 * nm_per_pix
            diff_res_nm = wavelength_nm / (abs(order_m) * lines_per_mm * grating_footprint_mm)
            r1, r2, r3 = st.columns(3)
            def _fmt(v):
                return f"{v * 1000:.1f} pm" if v < 1.0 else f"{v:.3f} nm"
            with r1:
                st.info(f"Sensor-limited (Nyquist): **{_fmt(sensor_res_nm)}**")
            with r2:
                st.info(f"Diffraction-limited: **{_fmt(diff_res_nm)}**  ·  beam footprint **{grating_footprint_mm:.1f} mm**")
            with r3:
                if sensor_res_nm >= diff_res_nm:
                    st.warning("Bottleneck: **sensor** (pixel size)")
                else:
                    st.warning("Bottleneck: **diffraction** (beam width)")
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
    st.markdown(
        r"**Spectral sampling:** $\frac{\Delta\lambda}{\Delta x} = \frac{pd\cos^3\beta}{mf}$"
    )

    current_nm_per_pix = None
    try:
        current_nm_per_pix = pc.nm_per_pixel(
            p=pixel_pitch_um * 1e-6,
            lines_per_mm=lines_per_mm_sw,
            alpha_deg=alpha_deg_sw,
            lambda_nm=wavelength_nm_sw,
            m=order_m_sw,
            f=focal_length_mm_sw * 1e-3,
        )
        _sw_c1, _sw_c2, _sw_c3 = st.columns(3)
        with _sw_c1:
            st.success(f"Current: **{current_nm_per_pix:.4f} nm/pixel**")
        with _sw_c2:
            st.info(f"Spectral range: **{current_nm_per_pix * camera_pixels:.2f} nm**")
        grating_footprint_mm_sw = beam_width_mm_sw / np.cos(np.radians(alpha_deg_sw))
        sensor_res_nm_sw = 2.0 * current_nm_per_pix
        diff_res_nm_sw = wavelength_nm_sw / (abs(order_m_sw) * lines_per_mm_sw * grating_footprint_mm_sw)
        def _fmt_sw(v):
            return f"{v * 1000:.1f} pm" if v < 1.0 else f"{v:.3f} nm"
        with _sw_c3:
            if sensor_res_nm_sw >= diff_res_nm_sw:
                st.warning("Bottleneck: **sensor**")
            else:
                st.warning("Bottleneck: **diffraction**")
        _sw_r1, _sw_r2 = st.columns(2)
        with _sw_r1:
            st.info(f"Sensor-limited (Nyquist): **{_fmt_sw(sensor_res_nm_sw)}**")
        with _sw_r2:
            st.info(f"Diffraction-limited: **{_fmt_sw(diff_res_nm_sw)}**  ·  beam footprint **{grating_footprint_mm_sw:.1f} mm**")
    except ValueError as e:
        st.error(str(e))

    sweep_values = np.linspace(sweep_start, sweep_stop, int(sweep_points))
    y_values = []

    for val in sweep_values:
        try:
            f_val = focal_length_mm_sw * 1e-3
            lines_val = lines_per_mm_sw
            lambda_val = wavelength_nm_sw
            alpha_val = alpha_deg_sw

            if sweep_param == "f (mm)":
                f_val = val * 1e-3
            elif sweep_param == "lines/mm":
                lines_val = val
            elif sweep_param == "λ (nm)":
                lambda_val = val
            elif sweep_param == "α (deg)":
                alpha_val = val

            y_values.append(
                pc.nm_per_pixel(
                    p=pixel_pitch_um * 1e-6,
                    lines_per_mm=lines_val,
                    alpha_deg=alpha_val,
                    lambda_nm=lambda_val,
                    m=order_m_sw,
                    f=f_val,
                )
            )
        except:
            y_values.append(np.nan)

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


# ============================================================================
# PAGE 3: SPECTROMETER ANALYSIS
# ============================================================================
else:
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
    if run_analysis_btn and uploaded_file is not None:
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
                }
                st.session_state.pop("fa_error", None)
            except Exception as _e:
                st.session_state["fa_error"] = str(_e)
                st.session_state.pop("fa_figs_png", None)
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
        a_cal = cal["a_nm_per_pixel"]
        b_cal = cal["b_nm"]
        unmatched_lines = data.get("unmatched_lines", {})
        sensor_max = data.get("sensor_max", 255)

        all_px = list(matched.keys()) + data["unmatched_pixels"]
        px_lo = 0
        px_hi = sensor_max

        visible_predicted = {
            wl: intensity
            for wl, intensity in unmatched_lines.items()
            if px_lo <= (wl - b_cal) / a_cal <= px_hi
        }

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("Detected & matched", len(matched))
        with mc2:
            st.metric("Detected, no match", len(data["unmatched_pixels"]))
        with mc3:
            st.metric("Expected, not found", len(visible_predicted))

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

        _tbl_rows = []
        for px, info in matched.items():
            _tbl_rows.append(
                (
                    float(px),
                    f"{px:.1f}",
                    f"{info['wavelength_nm']:.4f}",
                    "Detected",
                    _TABLE_DET,
                    f"{info['intensity']:.0f}",
                    f"{info['delta_px']:+.2f}",
                )
            )
        for px in data["unmatched_pixels"]:
            pred_wl = a_cal * float(px) + b_cal
            _tbl_rows.append(
                (
                    float(px),
                    f"{float(px):.1f}",
                    f"{pred_wl:.4f}",
                    "Detected — no match",
                    _TABLE_DET_NM,
                    "—",
                    "—",
                )
            )
        for wl, intensity in visible_predicted.items():
            pred_px = (wl - b_cal) / a_cal
            _tbl_rows.append(
                (
                    pred_px,
                    f"{pred_px:.1f}",
                    f"{wl:.4f}",
                    "Predicted (not found)",
                    _TABLE_PRED,
                    f"{intensity:.0f}",
                    "—",
                )
            )
        _tbl_rows.sort(key=lambda r: r[0])

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
        _stem = (
            uploaded_file.name.rsplit(".", 1)[0]
            if uploaded_file
            else "analysis"
        )
        st.download_button(
            "Download plots as PNG",
            data=_buf.getvalue(),
            file_name=f"{_stem}_greg.png",
            mime="image/png",
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
