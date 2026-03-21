# GREG - Grating Equation Generator

**GREG** is an interactive web app for diffraction grating analysis and spectrometer design. Built with Streamlit and Python, it provides real-time calculations and visualizations for optical engineers and researchers.

**Try it live**: [gregapp.streamlit.app](https://gregapp.streamlit.app/)

## Features

### Output Angle Analysis
- Calculate output angles (β) vs incident angles (α) for diffraction gratings
- Visualize grating behavior with interactive scatter plots colored by nm/pixel
- Display grating equation: *m·λ = d(sin α + sin β)*
- Identify Littrow configurations
- Calculate spectral sampling (nm/pixel)
- Support for multiple diffraction orders
- Dark theme optimized for visualization

### Sampling Sweep
- Display spectral sampling equation: *dλ/dx = (d·cos³β)/(mf)*
- Sweep parameters: focal length, groove density, wavelength, incident angle
- Dynamic parameter ranges that adjust based on selected sweep parameter
- Analyze spectral sampling (nm/pixel) vs system parameters
- Optimize spectrometer design
- Mark current configuration on sweep plots

### Spectrometer Analysis
- Upload a pickled matplotlib sensor-population figure (`.pkl`)
- Automatically extract peak pixel positions from the figure legend
- Match detected peaks against the built-in Neon reference spectrum (NIST)
- Six analysis plots in a 2×3 layout:
  - **(A)** Neighbour nm/px scale vs pixel position
  - **(B)** All-pair nm/px scale, coloured by first peak in pair
  - **(C)** nm/px scale as a function of distance between peaks
  - **(D)** NIST values vs measured data (normalised overlay)
  - **(E)** Linearity — wavelength vs pixel with residuals panel
  - **(F)** Gaussian fit of measured peaks with per-peak σ annotation
- Pixel-to-wavelength calibration with residuals
- Matched peaks summary table with residual column

## Quick Start

### Online (Recommended)
Visit the deployed app: **[gregapp.streamlit.app](https://gregapp.streamlit.app/)**

### Local Installation
```bash
# Clone the repository
git clone https://github.com/rngKomorebi/greg.git
cd greg

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run grating_streamlit.py
```

The app will open in your browser at http://localhost:8501

## Mobile Access
GREG is fully responsive and works on mobile devices. Simply visit the app URL from your phone's browser.

## Physics Background

GREG uses the grating equation for reflection gratings in an in-plane mount:

```
m·λ = d(sin α + sin β)
```

Where:
- **m** = diffraction order
- **λ** = wavelength
- **d** = groove spacing (1/lines_per_mm)
- **α** = incident angle
- **β** = output angle

Spectral sampling is calculated using:

```
dλ/dx = (d·cos³β)/(m·f)
```

Where **f** is the focal length and **x = f·tan β** is the detector position.

## Requirements
- Python 3.9+
- streamlit >= 1.28.0
- numpy >= 1.24.0
- matplotlib >= 3.9.0, < 3.10.0
- scipy >= 1.10.0

## File Structure
```
greg/
├── grating_streamlit.py    # Main Streamlit app (3 pages)
├── physics_core.py         # Grating equation & sampling calculations
├── analysis_core.py        # Spectrometer calibration, 6 plot functions
├── .streamlit/
│   └── config.toml         # Streamlit config (auto-rerun on save)
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
└── README.md               # This file
```

## License
MIT License - feel free to use for academic or commercial projects

## Author
**Sergei Kulkov** ([@rngKomorebi](https://github.com/rngKomorebi))

## Acknowledgements

The spectrometer analysis module (`analysis_core.py`) — including peak matching, dispersion calculations, Gaussian fitting, and the six analysis plots — was developed by **Sarlote Simsone** as part of her work on this project.

## Links
- **Live App**: [gregapp.streamlit.app](https://gregapp.streamlit.app/)
- **Repository**: [github.com/rngKomorebi/greg](https://github.com/rngKomorebi/greg)

---

