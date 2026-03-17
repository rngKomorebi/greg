# GREG - Grating Equation Generator

**GREG** is an interactive web app for diffraction grating analysis and spectrometer design. Built with Streamlit and Python, it provides real-time calculations and visualizations for optical engineers and researchers.

## 🌟 Features

### 📐 Output Angle Analysis
- Calculate output angles (β) vs incident angles (α) for diffraction gratings
- Visualize grating behavior with interactive scatter plots
- Identify Littrow configurations
- Calculate spectral sampling (nm/pixel)
- Support for multiple diffraction orders

### 📊 Sampling Sweep
- Sweep parameters: focal length, groove density, wavelength, incident angle
- Analyze spectral sampling vs system parameters
- Optimize spectrometer design
- Calculate Δλ for N pixels

### 📁 File Analysis
- Upload and analyze data files (txt, csv, dat)
- 6-plot grid visualization (2×3 layout)
- Statistical analysis (mean, std deviation)
- Support for 1D and 2D datasets
- Demo mode with example plots

## 🚀 Quick Start

### Online (Recommended)
Visit the deployed app: **[greg.streamlit.app](https://greg.streamlit.app)** *(link coming soon)*

### Local Installation
```powershell
# Clone the repository
git clone https://github.com/YOUR_USERNAME/greg.git
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

utf8m\lambda = d(\sin\alpha + \sin\beta)utf8

Where:
- $ = diffraction order
- $\lambda$ = wavelength
- $ = groove spacing (1/lines_per_mm)
- $\alpha$ = incident angle
- $\beta$ = output angle

Spectral sampling is calculated using:

utf8\frac{d\lambda}{dx} = \frac{d \cos^3\beta}{mf}utf8

Where $ is the focal length and  = f\tan\beta$ is the detector position.

## 📦 Requirements
- Python 3.8+
- streamlit >= 1.28.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0

## 🛠️ Technology Stack
- **Frontend**: Streamlit
- **Computation**: NumPy
- **Visualization**: Matplotlib
- **Physics**: Custom grating calculations (physics_core.py)

## 📄 File Structure
```
greg/
├── grating_streamlit.py   # Main Streamlit app
├── physics_core.py         # Grating physics calculations
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

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
- **App**: [greg.streamlit.app](https://greg.streamlit.app) *(coming soon)*
- **Repository**: [github.com/YOUR_USERNAME/greg](https://github.com/YOUR_USERNAME/greg)

---

**GREG** - *Making grating calculations simple and visual* 🌈
