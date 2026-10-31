# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.5] - 2026-10-31

### Added

- **Data export on every page.** A **Download data (CSV + JSON)** button
  saves the numbers behind the plots as a ZIP: one CSV per plot, `run.json`
  with the inputs and results, and a `README.txt` on loading them. CSV and
  JSON open outside Python and are safe to load, unlike a pickled figure.
  - Output Angle: `output_angle.csv` (α, β, nm/px).
  - Sampling Sweep: `sampling_sweep.csv` (swept parameter, nm/px).
  - Spectrometer Analysis: the raw sensor trace, the peaks table, and
    plots A–F (`C_D_all_pairs.csv` serves both C and D; E and F have two
    files each). The inputs in `run.json` are the ones the analysis ran
    with, even if the sidebar has changed since.
- New `export_core.py` builds the ZIP. `run_neon_analysis` returns the
  tables as `tables` (plus `peaks` and `export_errors`), computed by the
  same helpers that draw the plots.
- **Resolution model on Output Angle and Sampling Sweep**, labelled
  experimental until it is checked against more measurements.
  `physics_core.line_width_budget` predicts the line width as independent
  blurs added in quadrature: the far field of the single-mode beam (with the
  grating's anamorphic factor cos β / cos α), an optional extra angular blur
  (µrad, scales with f), the pixel response (a/√12) and an optional extra
  detector blur (µm). When the beam footprint D / cos α is wider than the
  grating, the cut beam's far field is used, so the resolution stops
  improving with α. The pages show line σ and FWHM in nm, the resolution
  (larger of FWHM and the 2 px sampling limit), R = λ/Δλ, and which term
  limits it. Sampling Sweep adds a resolution plot over the sweep, and both
  exports carry the results.
- New sidebar inputs for the model: camera lens axis, collimator focal
  length and fibre mode-field diameter (beam D = 4λf/(π·MFD); defaults
  Thorlabs PF2F63 + SM600, D ≈ 2.1 mm), grating width (25 mm),
  microlenses, extra angular and detector blur (both 0). Cameras have a fill
  factor (LinoSPAD2 0.58, with microlenses).
- **Editable pixel pitch and sensor size** in the sidebar, seeded from the
  selected camera, to project the instrument onto other cameras.
- `physics_core.collimated_beam_diameter_mm` (fibre + collimator → beam D).
- `scripts/resolution_vs_scale.py` fits α to measured Ne line widths
  (2400 gr/mm / 400 mm: σ 0.79 px at 0.0191 nm/px; 2400 gr/mm / 200 mm:
  σ 17 pm at 0.043 nm/px; 1200 gr/mm / 200 mm: σ 48 pm at 0.11 nm/px) and
  reports the extra angular blur needed at the lab α (65° from the normal).
  The fit gives 65.6° with almost no extra blur.
- **Spectral Units page**, ported from Prismatic: linked wavelength /
  frequency / wavenumber fields with unit selectors, spectrum bar, photon
  energy, period, angular frequency, and reference tabs (Neon from greg's
  NIST table, astronomical lines, laser lines, ultraviolet, infrared
  and telecom bands), with full names rather than bare abbreviations.
  - Air vs vacuum wavelengths (IAU standard formula, Morton 2000);
    frequency and wavenumber always refer to vacuum.
  - Spectral width at the current wavelength in nm, pm, GHz, MHz or cm⁻¹,
    with resolving power, coherence time and coherence length; it can take
    the resolution just computed on Output Angle or Sampling Sweep.
  - "Use in Output Angle / Sampling Sweep" sends the air wavelength to
    those pages and opens them.
- Pages 1–2 show ν and ν̃ under the wavelength input. New module
  `spectral_units.py` holds the conversions and reference data.
- Gaussian fits keep their parameter uncertainties (`perr`), exported as
  `sigma_px_err` / `sigma_nm_err` in `F_gaussian_peaks.csv`.

### Changed

- `nm_per_pixel` takes `lens_axis`: "beam" (cosβ, camera lens along the
  diffracted beam, now the default) or "normal" (cos³β, the previous
  behaviour, only valid for a detector centred on the grating normal).
  Order m = 0 now raises a clear `ValueError` instead of dividing by zero.
- The "Sensor-limited (Nyquist)" / "Diffraction-limited" cards are replaced
  by the resolution model above.
- `fit_gaussians` returns `(sigmas, popt, perr)`.

- Downloads no longer rerun the app (`on_click="ignore"`), so
  `requirements.txt` now needs `streamlit>=1.43.0`.
- The peaks table above the analysis plots is built from
  `analysis_core.peak_rows`, the same rows that go into `peaks.csv`.

### Fixed

- README plot list now uses the same A–F letters as the app.

## [0.0.4] - 2026-04-20

### Added

- **Spectrometer Analysis — peaks table** gains a **Δ (px)** column showing
  the signed residual (detected pixel − NIST-predicted pixel) for each
  matched peak under the final calibration. Unmatched rows show `—`.
  Column header has a hover tooltip explaining the value.

- **Spectrometer Analysis — Plot E** (Ideal Spectrum Overlay) gets a hover
  info icon `ⓘ` next to its title explaining that relative line intensities
  differ from NIST because they depend on the Ne source and its drive
  conditions; only peak *positions* are used for calibration.

- **Matching tolerance** sidebar input gains a `help` tooltip describing the
  parameter: maximum pixel distance allowed between a detected peak and its
  NIST-predicted position, and the trade-off between false matches and
  missed peaks.

### Changed

- **Spectrometer Analysis — six plots** are now individually fault-tolerant:
  each plot is generated in its own `try/except` block inside
  `run_neon_analysis`. A failing plot renders as a dark placeholder image
  with the error message; the remaining five plots are unaffected.
  `run_neon_analysis` returns an additional `plot_errors` dict keyed A–F.

- **Gaussian fit** (step 3 of the analysis pipeline) is likewise wrapped:
  failure sets Plot F to a placeholder and records the error, without
  aborting the other five plots.

- **All legend positions** in the six analysis plots changed from
  `"upper right"` to `"best"` to avoid occluding data points.

- **Plot C legend** labels changed from generic `Peak 1 / Peak 2 / …` to
  the actual pixel position of each group's first peak (`px 42`, `px 87`, …),
  matching the pixel-space axis of the plot.

## [0.0.3] - 2026-04-14

### Changed

- **Spectrometer Analysis** — reordered the six analysis plots from the
  original A–B–C–D–E–F sequence to A–E–B–C–D–F (i.e. Linearity promoted
  to the top-right slot), then relabelled the figures A–F to match the
  new reading order:
  - **(A)** Neighbour nm/px scale (unchanged position).
  - **(B)** Linearity / spectrometer stability (was E).
  - **(C)** All-pair nm/px scale, coloured by first peak (was B).
  - **(D)** nm/px scale vs distance between peaks (was C).
  - **(E)** NIST overlay — measured vs ideal spectrum (was D).
  - **(F)** Gaussian fit of measured peaks (unchanged position).

### Added

- **Download plots as PNG** button in the Spectrometer Analysis results
  section: stitches all six analysis figures into a single 2×3 PNG grid
  and offers it as `<original_filename>_greg.png` via `st.download_button`.

## [0.0.2] - 2026-03-21

Added third tab: spectrometer analysis for scale, linearity, resolution.

### Added

- **Spectrometer Analysis** page: upload a pickled matplotlib
  sensor-population figure, auto-extract peak pixel positions from the
  figure legend, and run the full Neon calibration pipeline.

- `analysis_core.py` — spectrometer calibration module (Sarlote Simsone):
  - Two-point seed calibration (`check_linearity`).
  - Iterative Neon peak matching against the built-in NIST reference
    spectrum (`match_peaks_to_neon`).
  - Neighbour and all-pairs nm/px dispersion analysis
    (`dispersion_neighbours`, `dispersion_all_pairs`).
  - Ideal pixel prediction under uniform dispersion (`_ideal_pixels`).
  - Multi-Gaussian fitting with collision-avoiding σ label annotations
    (`fit_gaussians`, `_annotate_sigma_labels`).
  - Six analysis figures produced by a single `run_neon_analysis` call:
    - **(A)** Neighbour nm/px scale overlaid on sensor-population figure.
    - **(B)** All-pair nm/px scale, coloured by first peak in pair.
    - **(C)** nm/px scale as a function of distance between peaks.
    - **(D)** NIST values vs measured data (normalised overlay).
    - **(E)** Linearity — wavelength vs pixel with residuals sub-panel
      captioned "Stability of the spectrometer".
    - **(F)** Gaussian fit of measured peaks with per-peak σ annotation.
  - `extract_pixels_from_legend` — auto-extracts peak pixel positions
    from a pickled figure's legend.

- `.streamlit/config.toml` with `runOnSave = true` for automatic reruns
  when source files change.

- `matplotlib >= 3.9.0, < 3.10.0` pin in requirements to ensure pickle
  compatibility with figures generated by the same matplotlib series.

## [0.0.1] - 2026-03-17

First public release of GREG — Grating Equation Generator. Includes two
analysis pages, for output angle after diffraction grating, and scaling 
of the spectral sampling with different parameters of the spectrometer.