"""
Streamlit-based Grating Explorer
A simpler, web-based alternative to the Qt GUI for exploring diffraction grating properties.
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
    page_icon="🌈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Title
st.title("🌈 GREG")
st.markdown(
    "**Grating Equation Generator** - Interactive diffraction grating calculator and visualizer"
)

# Tabs
tab1, tab2, tab3 = st.tabs(
    ["📐 Output Angle", "📊 Sampling Sweep", "📁 File Analysis"]
)


# ============================================================================
# TAB 1: OUTPUT ANGLE
# ============================================================================
with tab1:
    st.header("Output Angle Analysis")

    # Create columns for layout
    col1, col2 = st.columns([2, 1])

    with col2:
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
            value=30.0,
            step=1.0,
            key="alpha_deg_oa",
        )

        wavelength_nm = st.number_input(
            "Wavelength λ (nm)",
            min_value=450.0,
            max_value=820.0,
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

    with col1:
        # Calculate current output angle
        beta_current = None
        try:
            beta_current = pc.output_angle(
                order_m, wavelength_nm, int(lines_per_mm), alpha_deg
            )
            st.success(f"✓ Current output angle β = **{beta_current:.2f}°**")

            # Calculate nm/pixel
            try:
                nm_per_pix = pc.nm_per_pixel(
                    p=pixel_pitch_um * 1e-6,
                    lines_per_mm=lines_per_mm,
                    alpha_deg=alpha_deg,
                    lambda_nm=wavelength_nm,
                    m=order_m,
                    f=focal_length_mm * 1e-3,
                )
                st.info(f"📏 Spectral sampling: **{nm_per_pix:.4f} nm/pixel**")
            except Exception as e:
                st.warning(f"⚠️ Could not calculate nm/pixel: {e}")

        except ValueError as e:
            st.error(f"✗ {e}")

        # Generate plot data
        alphas = np.arange(alpha_min, alpha_max + alpha_step / 2, alpha_step)
        betas = []
        wavelengths_for_color = []

        for a in alphas:
            try:
                b = pc.output_angle(
                    order_m, wavelength_nm, int(lines_per_mm), a
                )
                betas.append(b)
                wavelengths_for_color.append(wavelength_nm)
            except:
                betas.append(np.nan)
                wavelengths_for_color.append(np.nan)

        # Create figure
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        # Plot β vs α with colormap
        scatter = ax.scatter(
            alphas,
            betas,
            c=wavelengths_for_color,
            cmap=colormap,
            s=marker_size,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
        )

        # Mark current point
        if beta_current is not None:
            ax.scatter(
                [alpha_deg],
                [beta_current],
                color="red",
                s=marker_size * 3,
                marker="*",
                edgecolors="black",
                linewidths=1.5,
                zorder=10,
                label=f"Current: α={alpha_deg}°, β={beta_current:.2f}°",
            )

        # Show Littrow configuration
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
                        linewidth=2,
                        alpha=0.7,
                    )
                    ax.axhline(
                        alpha_littrow,
                        color="orange",
                        linestyle="--",
                        linewidth=2,
                        alpha=0.7,
                    )
                    ax.text(
                        alpha_littrow,
                        ax.get_ylim()[1] * 0.95,
                        f"Littrow\nα={alpha_littrow:.1f}°",
                        ha="center",
                        va="top",
                        color="orange",
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.5",
                            facecolor="white",
                            alpha=0.7,
                        ),
                    )
            except:
                pass

        ax.set_xlabel("Incident Angle α (°)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Output Angle β (°)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Output Angle vs Incident Angle\n{int(lines_per_mm)} lines/mm, λ={wavelength_nm:.0f} nm, m={order_m}",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best")

        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, label="Wavelength (nm)")

        st.pyplot(fig)


# ============================================================================
# TAB 2: SAMPLING SWEEP
# ============================================================================
with tab2:
    st.header("Sampling Sweep Analysis")

    # Create columns for layout
    col1, col2 = st.columns([2, 1])

    with col2:
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
            min_value=450.0,
            max_value=820.0,
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

        st.subheader("Plot Options")

        plot_type = st.selectbox(
            "Y variable",
            ["nm per pixel", "Δλ for N pixels"],
            index=0,
            key="plot_type",
        )

        n_pixels = st.number_input(
            "N pixels",
            min_value=1,
            max_value=100,
            value=2,
            step=1,
            key="n_pixels",
        )

        sweep_param = st.selectbox(
            "Sweep parameter",
            ["f (mm)", "lines/mm", "λ (nm)", "α (deg)"],
            index=0,
            key="sweep_param",
        )

        sweep_start = st.number_input(
            "Sweep start", value=50.0, step=10.0, key="sweep_start"
        )

        sweep_stop = st.number_input(
            "Sweep stop", value=500.0, step=10.0, key="sweep_stop"
        )

        sweep_points = st.number_input(
            "Points",
            min_value=2,
            max_value=5000,
            value=50,
            step=10,
            key="sweep_points",
        )

    with col1:
        # Calculate current value
        try:
            current_nm_per_pix = pc.nm_per_pixel(
                p=pixel_pitch_um_sw * 1e-6,
                lines_per_mm=lines_per_mm_sw,
                alpha_deg=alpha_deg_sw,
                lambda_nm=wavelength_nm_sw,
                m=order_m_sw,
                f=focal_length_mm_sw * 1e-3,
            )

            if plot_type == "nm per pixel":
                st.success(f"✓ Current: **{current_nm_per_pix:.4f} nm/pixel**")
            else:
                delta_lambda = current_nm_per_pix * n_pixels
                st.success(
                    f"✓ Current: **{delta_lambda:.4f} nm** for {n_pixels} pixels"
                )

        except ValueError as e:
            st.error(f"✗ {e}")

        # Generate sweep data
        sweep_values = np.linspace(sweep_start, sweep_stop, int(sweep_points))
        y_values = []

        for val in sweep_values:
            try:
                # Set parameters based on sweep
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

                nm_per_pix = pc.nm_per_pixel(
                    p=pixel_pitch_um_sw * 1e-6,
                    lines_per_mm=lines_val,
                    alpha_deg=alpha_val,
                    lambda_nm=lambda_val,
                    m=order_m_sw,
                    f=f_val,
                )

                if plot_type == "nm per pixel":
                    y_values.append(nm_per_pix)
                else:
                    y_values.append(nm_per_pix * n_pixels)

            except:
                y_values.append(np.nan)

        # Create figure
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        # Plot sweep
        ax.plot(
            sweep_values, y_values, "c-", linewidth=2, marker="o", markersize=4
        )

        # Mark current value
        try:
            if sweep_param == "f (mm)":
                current_x = focal_length_mm_sw
            elif sweep_param == "lines/mm":
                current_x = lines_per_mm_sw
            elif sweep_param == "λ (nm)":
                current_x = wavelength_nm_sw
            else:  # α (deg)
                current_x = alpha_deg_sw

            if plot_type == "nm per pixel":
                current_y = current_nm_per_pix
            else:
                current_y = current_nm_per_pix * n_pixels

            ax.scatter(
                [current_x],
                [current_y],
                color="red",
                s=150,
                marker="*",
                edgecolors="black",
                linewidths=1.5,
                zorder=10,
                label="Current configuration",
            )
        except:
            pass

        ax.set_xlabel(f"{sweep_param}", fontsize=12, fontweight="bold")

        if plot_type == "nm per pixel":
            ylabel = "nm/pixel"
        else:
            ylabel = f"Δλ for {n_pixels} pixels (nm)"
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")

        title = f"Spectral Sampling vs {sweep_param}\n"
        title += f"{int(lines_per_mm_sw)} lines/mm, α={alpha_deg_sw}°, λ={wavelength_nm_sw:.0f} nm, f={focal_length_mm_sw:.0f} mm"
        ax.set_title(title, fontsize=14, fontweight="bold")

        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best")

        st.pyplot(fig)


# ============================================================================
# TAB 3: FILE ANALYSIS
# ============================================================================
with tab3:
    st.header("File Analysis")
    st.markdown("Upload and analyze data files with multi-plot visualization")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a data file",
        type=["txt", "csv", "dat"],
        help="Upload a text or CSV file containing numerical data",
    )

    # Layout: File info and results
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("File Information")
        if uploaded_file is not None:
            st.success(f"✓ File loaded: **{uploaded_file.name}**")
            st.info(f"Size: {uploaded_file.size} bytes")
        else:
            st.info("No file uploaded yet")

    with col2:
        st.subheader("Analysis Results")
        if uploaded_file is not None:
            # Try to load and analyze the file
            try:
                # Read the file
                stringio = io.StringIO(
                    uploaded_file.getvalue().decode("utf-8")
                )
                data = np.loadtxt(stringio)

                # Calculate statistics
                mean_val = np.mean(data)
                std_val = np.std(data)

                st.metric("Mean Value", f"{mean_val:.3f}")
                st.metric("Std Deviation", f"{std_val:.3f}")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.info("Showing demo plots instead")
                mean_val = np.random.rand() * 100
                std_val = np.random.rand() * 50
                st.metric("Mean Value (demo)", f"{mean_val:.3f}")
                st.metric("Std Deviation (demo)", f"{std_val:.3f}")
        else:
            st.info("Upload a file to see results")

    # Plot section
    st.markdown("---")
    st.subheader("Analysis Plots")

    if uploaded_file is not None:
        try:
            # Try to load the file
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            data = np.loadtxt(stringio)

            # If data is 1D, create sample plots
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
                # 2D data - plot columns
                plot_data = []
                for i in range(min(6, data.shape[1])):
                    plot_data.append(
                        (f"Column {i+1}", np.arange(len(data)), data[:, i])
                    )

        except:
            # Demo mode - show example plots
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
        # Demo mode - show example plots
        x = np.linspace(0, 100, 200)
        plot_data = [
            ("Sine Wave", x, np.sin(x * 0.1)),
            ("Cosine Wave", x, np.cos(x * 0.1)),
            ("Random Walk", x, np.cumsum(np.random.randn(len(x))) * 0.1),
            ("Exponential", x, np.exp(-x / 50)),
            ("Gaussian", x, np.exp(-((x - 50) ** 2) / 200)),
            ("Linear", x, 0.5 * x + np.random.randn(len(x)) * 5),
        ]

    # Create 2x3 grid of plots
    cols = st.columns(3)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, (title, x_data, y_data) in enumerate(plot_data[:6]):
        with cols[i % 3]:
            fig = Figure(figsize=(5, 3.5))
            ax = fig.add_subplot(111)

            if title == "Histogram":
                ax.hist(
                    y_data,
                    bins=30,
                    color=colors[i],
                    alpha=0.7,
                    edgecolor="black",
                )
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
            else:
                ax.plot(x_data, y_data, color=colors[i], linewidth=2)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")

            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3, linestyle="--")

            st.pyplot(fig)


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p><strong>GREG</strong> - Grating Equation Generator</p>
    <p>For diffraction grating analysis and spectrometer design</p>
</div>
""",
    unsafe_allow_html=True,
)
