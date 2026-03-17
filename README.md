# GREG - Grating Equation Generator

**GREG** is an interactive web app for diffraction grating analysis and spectrometer design. Built with Streamlit and Python, it provides real-time calculations and visualizations for optical engineers and researchers.

🌐 **Try it live**: [gregapp.streamlit.app](https://gregapp.streamlit.app/)

## 🌟 Features

### 📐 Output Angle Analysis
- Calculate output angles (β) vs incident angles (α) for diffraction gratings
- Visualize grating behavior with interactive scatter plots colored by nm/pixel
- Display grating equation: *m·λ = d(sin α + sin β)*
- Identify Littrow configurations
- Calculate spectral sampling (nm/pixel)
- Support for multiple diffraction orders
- Dark theme optimized for visualization

### 📊 Sampling Sweep
- Display spectral sampling equation: *dλ/dx = (d·cos³β)/(mf)*
- Sweep parameters: focal length, groove density, wavelength, incident angle
- Dynamic parameter ranges that adjust based on selected sweep parameter
- Analyze spectral sampling (nm/pixel) vs system parameters
- Optimize spectrometer design
- Mark current configuration on sweep plots

### 📁 File Analysis
- Upload and analyze data files (txt, csv, dat)
- 6-plot grid visualization (2×3 layout)
- Statistical analysis (mean, std deviation)
- Support for 1D and 2D datasets
- Demo mode with example plots

## 🚀 Quick Start

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

## 📱 Mobile Access
GREG is fully responsive and works on mobile devices. Simply visit the app URL from your phone's browser.

## 🧮 Physics Background

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

## 📦 Requirements
- Python 3.8+
- streamlit >= 1.28.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0

## 🛠️ Technology Stack
- **Frontend**: Streamlit
- **Computation**: NumPy
- **Visualization**: Matplotlib with dark theme
- **Physics**: Custom grating calculations (physics_core.py)

## 📄 File Structure
```
greg/
├── grating_streamlit.py   # Main Streamlit app (3 tabs)
├── physics_core.py         # Grating physics calculations
├── requirements.txt        # Python dependencies
├── example_spectrum.csv    # Sample data for testing
├── LICENSE                 # MIT License
└── README.md              # This file
```

## 🔄 Deployment
The app automatically deploys to Streamlit Cloud when changes are pushed to the main branch on GitHub. No manual deployment steps required.

## 🤝 Contributing
Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📝 License
MIT License - feel free to use for academic or commercial projects

## 👤 Author
Built for diffraction grating analysis and spectrometer design

## 🔗 Links
- **Live App**: [gregapp.streamlit.app](https://gregapp.streamlit.app/)
- **Repository**: [github.com/rngKomorebi/greg](https://github.com/rngKomorebi/greg)

---

**GREG** - *Making grating calculations simple and visual* 🌈
