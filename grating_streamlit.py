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
        ["Output Angle", "Sampling Sweep", "File Analysis"],
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

    # File Analysis params
    else:
        st.subheader("Data File")
        uploaded_file = st.file_uploader(
            "Choose a data file",
            type=["pkl", "pickle"],
            help="Upload a pickle (.pkl) file containing the spectrum data",
        )
        if uploaded_file is not None:
            st.success(f"Loaded: **{uploaded_file.name}**")
            st.caption(f"{uploaded_file.size} bytes")
        else:
            st.info("No file uploaded yet")


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
    cbar.set_label("nm/pixel", color=PLOT_AXIS)
    cbar.ax.yaxis.set_tick_params(color=PLOT_AXIS)
    cbar.outline.set_edgecolor(PLOT_AXIS)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=PLOT_AXIS)

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
# PAGE 3: FILE ANALYSIS
# ============================================================================
else:
    st.markdown(
        f"""
        <div style="margin-bottom: 1.75rem;">
            <h1 style="font-size: 2.25rem; font-weight: 900; color: {TEXT};
                       letter-spacing: -0.03em; margin-bottom: 0;">
                File Analysis
            </h1>
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-top: 0.6rem;">
                <span style="height: 1px; width: 2.5rem; background: {ACCENT};
                             display: inline-block;"></span>
                <span style="font-size: 0.65rem; color: {TEXT_MUTED};
                             letter-spacing: 0.2em; text-transform: uppercase;
                             font-weight: 700;">multi-plot data visualization</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if uploaded_file is not None:
        try:
            import pandas as pd

            uploaded_file.seek(0)
            content = uploaded_file.getvalue().decode("utf-8-sig")
            stringio = io.StringIO(content)
            try:
                df = pd.read_csv(stringio, header=None)
                data_stats = (
                    df.values.flatten() if df.shape[1] == 1 else df.values
                )
            except:
                stringio.seek(0)
                data_stats = np.loadtxt(stringio, delimiter=",")
            sc1, sc2 = st.columns(2)
            with sc1:
                st.metric("Mean Value", f"{np.mean(data_stats):.3f}")
            with sc2:
                st.metric("Std Deviation", f"{np.std(data_stats):.3f}")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    if uploaded_file is not None:
        try:
            import pandas as pd

            uploaded_file.seek(0)
            content = uploaded_file.getvalue().decode("utf-8-sig")
            stringio = io.StringIO(content)
            try:
                df = pd.read_csv(stringio, header=None)
                data = df.values
            except:
                stringio.seek(0)
                data = np.loadtxt(stringio, delimiter=",")

            if data.ndim == 1:
                x = np.arange(len(data))
                plot_data = [
                    ("Raw Data", x, data),
                    ("Cumulative Sum", x, np.cumsum(data)),
                    (
                        "Moving Average",
                        x[49:],
                        np.convolve(data, np.ones(50) / 50, mode="valid"),
                    ),
                    ("Histogram", None, data),
                    ("Normalized", x, (data - np.mean(data)) / np.std(data)),
                    ("Derivative", x[:-1], np.diff(data)),
                ]
            else:
                plot_data = [
                    (f"Column {i+1}", np.arange(len(data)), data[:, i])
                    for i in range(min(6, data.shape[1]))
                ]
        except:
            x = np.linspace(0, 100, 200)
            plot_data = [
                ("Sine Wave", x, np.sin(x * 0.1)),
                ("Cosine Wave", x, np.cos(x * 0.1)),
                ("Random Walk", x, np.cumsum(np.random.randn(len(x))) * 0.1),
                ("Exponential", x, np.exp(-x / 50)),
                ("Gaussian", x, np.exp(-((x - 50) ** 2) / 200)),
                ("Linear", x, 0.5 * x + np.random.randn(len(x)) * 5),
            ]
    else:
        x = np.linspace(0, 100, 200)
        plot_data = [
            ("Sine Wave", x, np.sin(x * 0.1)),
            ("Cosine Wave", x, np.cos(x * 0.1)),
            ("Random Walk", x, np.cumsum(np.random.randn(len(x))) * 0.1),
            ("Exponential", x, np.exp(-x / 50)),
            ("Gaussian", x, np.exp(-((x - 50) ** 2) / 200)),
            ("Linear", x, 0.5 * x + np.random.randn(len(x)) * 5),
        ]

    cols = st.columns(3)
    plot_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    ]

    for i, (title, x_data, y_data) in enumerate(plot_data[:6]):
        with cols[i % 3]:
            fig = Figure(figsize=(5, 3.5), facecolor=PLOT_BG)
            ax = fig.add_subplot(111, facecolor=PLOT_SURFACE)
            ax.tick_params(colors=PLOT_AXIS, which="both", labelsize=10)
            for spine in ax.spines.values():
                spine.set_color("white")

            if title == "Histogram":
                ax.hist(
                    y_data,
                    bins=30,
                    color=plot_colors[i],
                    alpha=0.7,
                    edgecolor="white",
                )
                ax.set_xlabel("Value", fontsize=11, color=PLOT_AXIS)
                ax.set_ylabel("Frequency", fontsize=11, color=PLOT_AXIS)
            else:
                ax.plot(x_data, y_data, color=plot_colors[i], linewidth=1.75)
                ax.set_xlabel("X", fontsize=11, color=PLOT_AXIS)
                ax.set_ylabel("Y", fontsize=11, color=PLOT_AXIS)

            ax.set_title(
                title, fontsize=12, fontweight="bold", color=PLOT_AXIS
            )
            ax.grid(True, alpha=0.3, linestyle="--", color=PLOT_GRID)

            st.pyplot(fig)


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
