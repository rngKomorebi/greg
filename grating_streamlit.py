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

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
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

# Design tokens
BG = "#131313"
SIDEBAR_BG = "#1c1b1b"
NAV_ACTIVE = "#201f1f"
SURFACE = "#2a2a2a"
ACCENT = "#42e5b0"
TEXT = "#e5e2e1"
TEXT_MUTED = "#bbcac1"
BORDER = "#3c4a43"
PLOT_BG = "#0e1117"
PLOT_SURFACE = "#0e1117"
PLOT_AXIS = "white"
PLOT_GRID = "gray"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100;300;400;500;600;700;800;900&display=swap');

    html, body, .stApp {{
        background-color: {BG} !important;
        font-family: 'Inter', sans-serif !important;
        color: {TEXT} !important;
    }}
    .main .block-container {{
        background-color: {BG} !important;
        padding: 2rem 2.5rem !important;
        max-width: 100% !important;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {TEXT} !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
    }}
    p, span, label, div, .stMarkdown {{
        font-family: 'Inter', sans-serif !important;
    }}
    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"],
    [data-testid="stSidebar"] .block-container {{
        background-color: {SIDEBAR_BG} !important;
        border-right: 1px solid {BORDER} !important;
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
        color: {TEXT_MUTED} !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        margin: 0 !important;
        width: 100% !important;
        transition: all 0.15s ease !important;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
        background-color: {NAV_ACTIVE} !important;
        color: {TEXT} !important;
        border-left-color: {ACCENT} !important;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
        background-color: {NAV_ACTIVE} !important;
        color: {ACCENT} !important;
        border-left: 2px solid {ACCENT} !important;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {{
        display: none !important;
    }}
    [data-testid="stSidebar"] h3 {{
        color: {TEXT_MUTED} !important;
        font-size: 0.6rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.18em !important;
        text-transform: uppercase !important;
        padding: 0 1.5rem !important;
        margin-top: 1.25rem !important;
        margin-bottom: 0.25rem !important;
    }}
    .stNumberInput input, .stTextInput input {{
        background-color: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        color: {TEXT} !important;
        border-radius: 0.25rem !important;
    }}
    .stNumberInput input:focus, .stTextInput input:focus {{
        border-color: {ACCENT} !important;
        box-shadow: 0 0 0 2px rgba(163, 166, 255, 0.15) !important;
    }}
    [data-testid="stSelectbox"] > div > div {{
        background-color: {SURFACE} !important;
        border-color: {BORDER} !important;
        color: {TEXT} !important;
    }}
    [data-testid="stMetric"] {{
        background-color: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-left: 2px solid {ACCENT} !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem 1rem !important;
    }}
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
        color: {TEXT} !important;
    }}
    [data-testid="stFileUploader"] {{
        background-color: {SURFACE} !important;
        border: 1px dashed {BORDER} !important;
        border-radius: 0.25rem !important;
    }}
    hr {{ border-color: {BORDER} !important; opacity: 0.4 !important; }}
    [data-testid="collapsedControl"],
    button[kind="header"][aria-label="Close sidebar"],
    [data-testid="stSidebarCollapseButton"] {{
        display: none !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding: 1.5rem 1.5rem 0.75rem; font-family: Inter, sans-serif;">
            <div style="font-size: 1.4rem; font-weight: 900; color: {TEXT};
                        letter-spacing: -0.04em;">GREG</div>
            <div style="font-size: 0.6rem; color: {TEXT_MUTED}; letter-spacing: 0.2em;
                        text-transform: uppercase; margin-top: 0.2rem;">
                Grating Equation Generator
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "",
        ["Output Angle", "Sampling Sweep", "Spectrometer Analysis"],
        label_visibility="collapsed",
    )

    st.divider()

    # Output Angle params
    if page == "Output Angle":
        st.subheader("Instrument Parameters")
        pixel_pitch_um = st.number_input(
            "Pixel pitch p (µm)",
            min_value=0.1,
            max_value=1000.0,
            value=26.2,
            step=0.1,
            format="%.3f",
            key="pixel_pitch_oa",
        )
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
        pixel_pitch_um_sw = st.number_input(
            "Pixel pitch p (µm)",
            min_value=0.1,
            max_value=1000.0,
            value=26.2,
            step=0.1,
            format="%.3f",
            key="pixel_pitch_sw",
        )
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
            f"""
            <style>
            div[data-testid="stSidebarContent"] div[data-testid="stButton"] > button {{
                display: block;
                margin: 0.4rem auto 0.2rem auto;
                font-size: 1.15rem;
                font-weight: 700;
                border: 2px solid {ACCENT};
                padding: 0.45rem 2rem;
                width: 80%;
            }}
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
        f"""
        <div style="margin-bottom: 1.75rem;">
            <h1 style="font-size: 2.25rem; font-weight: 900; color: {TEXT};
                       letter-spacing: -0.03em; margin-bottom: 0;">
                Output Angle Analysis
            </h1>
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-top: 0.6rem;">
                <span style="height: 1px; width: 2.5rem; background: {ACCENT};
                             display: inline-block;"></span>
                <span style="font-size: 0.65rem; color: {TEXT_MUTED};
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
        c1, c2 = st.columns(2)
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

    with plt.rc_context({"font.size": 10}):
        fig = Figure(figsize=(10, 4.5), facecolor=PLOT_BG)
        ax = fig.add_subplot(111, facecolor=PLOT_SURFACE)
        ax.tick_params(colors=PLOT_AXIS, which="both", labelsize=10)
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
                        fontsize=10,
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

        ax.set_xlabel("Incident Angle α (°)", fontsize=11, color=PLOT_AXIS)
        ax.set_ylabel("Output Angle β (°)", fontsize=11, color=PLOT_AXIS)
        ax.set_title(
            f"Output Angle vs Incident Angle  ·  {int(lines_per_mm)} lines/mm"
            f"  ·  λ={wavelength_nm:.0f} nm  ·  m={order_m}",
            fontsize=12,
            fontweight="bold",
            color=PLOT_AXIS,
        )
        ax.grid(True, alpha=0.3, linestyle="--", color=PLOT_GRID)
        ax.legend(loc="best", fontsize=9)

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("nm/pixel", color=PLOT_AXIS, fontsize=10)
        cbar.ax.tick_params(labelsize=9, colors=PLOT_AXIS)
        cbar.outline.set_edgecolor(PLOT_AXIS)

        st.pyplot(fig, use_container_width=True)


# ============================================================================
# PAGE 2: SAMPLING SWEEP
# ============================================================================
elif page == "Sampling Sweep":
    st.markdown(
        f"""
        <div style="margin-bottom: 1.75rem;">
            <h1 style="font-size: 2.25rem; font-weight: 900; color: {TEXT};
                       letter-spacing: -0.03em; margin-bottom: 0;">
                Sampling Sweep
            </h1>
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-top: 0.6rem;">
                <span style="height: 1px; width: 2.5rem; background: {ACCENT};
                             display: inline-block;"></span>
                <span style="font-size: 0.65rem; color: {TEXT_MUTED};
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
            p=pixel_pitch_um_sw * 1e-6,
            lines_per_mm=lines_per_mm_sw,
            alpha_deg=alpha_deg_sw,
            lambda_nm=wavelength_nm_sw,
            m=order_m_sw,
            f=focal_length_mm_sw * 1e-3,
        )
        st.success(f"Current: **{current_nm_per_pix:.4f} nm/pixel**")
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
                    p=pixel_pitch_um_sw * 1e-6,
                    lines_per_mm=lines_val,
                    alpha_deg=alpha_val,
                    lambda_nm=lambda_val,
                    m=order_m_sw,
                    f=f_val,
                )
            )
        except:
            y_values.append(np.nan)

    with plt.rc_context({"font.size": 10}):
        fig = Figure(figsize=(10, 4.5), facecolor=PLOT_BG)
        ax = fig.add_subplot(111, facecolor=PLOT_SURFACE)
        ax.tick_params(colors=PLOT_AXIS, which="both", labelsize=10)
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

        ax.set_xlabel(sweep_param, fontsize=11, color=PLOT_AXIS)
        ax.set_ylabel("nm/pixel", fontsize=11, color=PLOT_AXIS)
        ax.set_title(
            f"Spectral Sampling vs {sweep_param}  ·  {int(lines_per_mm_sw)} lines/mm"
            f"  ·  α={alpha_deg_sw}°  ·  λ={wavelength_nm_sw:.0f} nm  ·  f={focal_length_mm_sw:.0f} mm",
            fontsize=12,
            fontweight="bold",
            color=PLOT_AXIS,
        )
        ax.grid(True, alpha=0.3, linestyle="--", color=PLOT_GRID)
        ax.legend(loc="best", fontsize=9)

        st.pyplot(fig, use_container_width=True)


# ============================================================================
# PAGE 3: SPECTROMETER ANALYSIS
# ============================================================================
else:
    st.markdown(
        f"""
        <div style="margin-bottom: 1.75rem;">
            <h1 style="font-size: 2.25rem; font-weight: 900; color: {TEXT};
                       letter-spacing: -0.03em; margin-bottom: 0;">
                Spectrometer Analysis
            </h1>
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-top: 0.6rem;">
                <span style="height: 1px; width: 2.5rem; background: {ACCENT};
                             display: inline-block;"></span>
                <span style="font-size: 0.65rem; color: {TEXT_MUTED};
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
            f"""
            <div style="margin-bottom:0.25rem;">
                <span style="font-size:0.6rem;color:{TEXT_MUTED};
                             letter-spacing:0.15em;text-transform:uppercase;
                             font-weight:700;">Reference</span>
                <span style="font-size:0.95rem;font-weight:700;
                             color:{TEXT};margin-left:0.5rem;">Uploaded Figure</span>
            </div>
            <div style="font-size:0.72rem;color:{TEXT_MUTED};margin-bottom:0.5rem;">
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
            f"<span style='font-size:0.75rem;letter-spacing:0.12em;"
            f"text-transform:uppercase;font-weight:700;color:{TEXT_MUTED};'>"
            f"All peaks in range</span>",
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
            f"background:{SURFACE};color:{TEXT_MUTED};font-size:0.72rem;"
            f"font-weight:700;letter-spacing:0.1em;text-transform:uppercase;"
            f"padding:0.4rem 0.8rem;border-bottom:1px solid {BORDER};"
            f"text-align:left;"
        )
        _td = f"padding:0.28rem 0.8rem;font-size:0.84rem;border-bottom:1px solid {BORDER};"
        _html_rows = "".join(
            f"<tr>"
            f"<td style='{_td}color:{TEXT_MUTED};'>{i}</td>"
            f"<td style='{_td}color:{color};'>{px_str}</td>"
            f"<td style='{_td}color:{color};'>{wl_str}</td>"
            f"<td style='{_td}color:{color};font-size:0.72rem;'>{label}</td>"
            f"<td style='{_td}color:{TEXT_MUTED};text-align:right;'>{intens_str}</td>"
            f"<td style='{_td}color:{TEXT_MUTED};text-align:right;'>{delta_str}</td>"
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
                        f"color:{TEXT_MUTED};}}"
                        f".tt-{key} .tt-box{{visibility:hidden;background:{SURFACE};"
                        f"color:{TEXT};text-align:left;border-radius:4px;"
                        f"padding:8px 10px;position:absolute;z-index:9999;"
                        f"bottom:130%;left:50%;transform:translateX(-50%);"
                        f"width:280px;font-size:0.72rem;line-height:1.5;"
                        f"border:1px solid {BORDER};pointer-events:none;}}"
                        f".tt-{key}:hover .tt-box{{visibility:visible;}}"
                        f"</style>"
                        f'<span class="tt-{key}">ⓘ'
                        f'<span class="tt-box">{note}</span>'
                        f"</span>"
                    )
                st.markdown(
                    f"""
                    <div style="margin-bottom:0.25rem;">
                        <span style="font-size:0.6rem;color:{TEXT_MUTED};
                                     letter-spacing:0.15em;text-transform:uppercase;
                                     font-weight:700;">Plot {key}</span>
                        <span style="font-size:0.95rem;font-weight:700;
                                     color:{TEXT};margin-left:0.5rem;">{title}</span>
                        {_note_html}
                    </div>
                    <div style="font-size:0.72rem;color:{TEXT_MUTED};
                                margin-bottom:0.5rem;">{caption}</div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(figs_png[key], use_container_width=True)

        # ── Download all plots as a single PNG grid
        _imgs = [Image.open(io.BytesIO(figs_png[k])) for k, *_ in _plot_meta]
        _w = max(i.width for i in _imgs)
        _h = max(i.height for i in _imgs)
        _grid = Image.new("RGB", (_w * 2, _h * 3), (14, 17, 23))
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
    f"""
    <div style='text-align: center; color: {TEXT_MUTED}; padding: 2rem 0 1rem;
                font-size: 0.75rem; border-top: 1px solid {BORDER};
                margin-top: 2rem; opacity: 0.6; font-family: Inter, sans-serif;'>
        <strong style="color: {TEXT};">GREG</strong> &nbsp;·&nbsp;
        Grating Equation Generator
    </div>
    """,
    unsafe_allow_html=True,
)
