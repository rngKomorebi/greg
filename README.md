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
- Display spectral sampling equation: *dλ/dx = (d·cosβ)/(mf)*, or
  *(d·cos³β)/(mf)* with the lens along the grating normal
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
  - **(B)** Linearity — wavelength vs pixel with residuals panel
  - **(C)** All-pair nm/px scale, coloured by first peak in pair
  - **(D)** nm/px scale as a function of distance between peaks
  - **(E)** NIST values vs measured data (normalised overlay)
  - **(F)** Gaussian fit of measured peaks with per-peak σ annotation
- Pixel-to-wavelength calibration with residuals
- Matched peaks summary table with residual column

### Resolution Model (Output Angle and Sampling Sweep, experimental)
- Predicted line width from independent blurs added in quadrature: the
  far field of the single-mode beam (including the grating's anamorphic
  factor), optional extra angular blur, the pixel response, and optional
  extra detector blur
- Beam diameter from the collimator focal length and fibre mode-field
  diameter; a grating narrower than the beam footprint cuts the beam
- Line σ and FWHM in nm, resolution (larger of FWHM and 2 px), R = λ/Δλ,
  and the limiting term
- Resolution vs the swept parameter on Sampling Sweep
- Camera lens axis option: along the diffracted beam (cosβ, default) or
  along the grating normal (cos³β)
- Editable pixel pitch and sensor size, seeded from the selected camera, to
  project onto other cameras

### Spectral Units
- Linked wavelength / frequency / wavenumber fields with unit selectors
- Air vs vacuum wavelengths; photon energy, period, angular frequency
- Spectral width ↔ Δν ↔ Δν̃, resolving power, coherence time and length
- Clickable reference lines (Neon, astronomical, lasers) and bands
- Send a wavelength straight to Output Angle or Sampling Sweep

### Data Export
- Every page has a **Download data (CSV + JSON)** button for replotting or
  combining runs elsewhere
- The ZIP holds one CSV per plot, `run.json` with the inputs and results, and
  a `README.txt`
- Load with `pandas.read_csv("<name>.csv")` or
  `numpy.genfromtxt("<name>.csv", delimiter=",", names=True)`

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
dλ/dx = (d·cosβ)/(m·f)
```

Where **f** is the focal length of a camera lens pointed along the
diffracted beam. With the lens along the grating normal (x = f·tan β) it is
(d·cos³β)/(m·f).

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

