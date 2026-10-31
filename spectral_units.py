"""
spectral_units.py — spectral unit conversions and reference data for GREG.

Ported from Prismatic (Spectral Unit Explorer) and extended with spectral
widths, coherence time and air/vacuum wavelengths.

Conventions
-----------
Wavelengths are in nm. The canonical wavelength is the vacuum wavelength;
frequency and wavenumber are always derived from it. "Air" means dry standard air
at 15 °C and 101 325 Pa, the convention of NIST line lists and of most
astronomical line tables above 200 nm.

Public API
----------
Positions
    freq_thz(wl_nm), wnum_cm(wl_nm), wl_from_freq(thz), wl_from_wnum(cm)
    photon_energy_ev(wl_nm), period_fs(wl_nm), ang_freq_radfs(wl_nm)
Air / vacuum
    air_to_vacuum(wl_air_nm), vacuum_to_air(wl_vac_nm), refractive_index_air(wl_vac_nm)
Widths at a centre wavelength
    width_from_nm(dl_nm, wl_nm) / width_from_ghz / width_from_cm → dict
Display helpers
    wavelength_to_rgb, spectral_region, find_band, fmt_energy, fmt_period, fmt_omega, fmt_time
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── Physical constants ───────────────────────────────────────────────────────
C_M_S = 299_792_458.0       # c [m/s]
C_NM_THZ = 299_792.458      # c [nm·THz]
HC_EV_NM = 1_239.841_984    # h·c [eV·nm]

# The air/vacuum formulas below are fitted for λ ≥ 200 nm; shorter
# wavelengths only exist in vacuum, so they are returned unchanged.
AIR_MIN_NM = 200.0

# ─── Unit tables ──────────────────────────────────────────────────────────────
# canonical = display × FACTOR   (canonical: nm, THz, cm⁻¹)
WL_UNITS = {"m": 1e9, "cm": 1e7, "mm": 1e6, "µm": 1e3, "nm": 1.0, "pm": 1e-3, "Å": 0.1}
WL_STEP = {"m": 1e-9, "cm": 1e-7, "mm": 1e-6, "µm": 1e-3, "nm": 0.1, "pm": 100.0, "Å": 1.0}
WL_FMT = {"m": "%.6g", "cm": "%.6g", "mm": "%.6g", "µm": "%.5f", "nm": "%.4f",
          "pm": "%.2f", "Å": "%.3f"}

FREQ_UNITS = {"Hz": 1e-12, "kHz": 1e-9, "MHz": 1e-6, "GHz": 1e-3, "THz": 1.0, "PHz": 1e3}
FREQ_STEP = {"Hz": 1e9, "kHz": 1e6, "MHz": 1000.0, "GHz": 1.0, "THz": 1e-3, "PHz": 1e-6}
FREQ_FMT = {"Hz": "%.6g", "kHz": "%.6g", "MHz": "%.4f", "GHz": "%.6f", "THz": "%.4f",
            "PHz": "%.8f"}

WNUM_UNITS = {"m⁻¹": 0.01, "cm⁻¹": 1.0, "mm⁻¹": 10.0}
WNUM_STEP = {"m⁻¹": 1e5, "cm⁻¹": 1.0, "mm⁻¹": 0.1}
WNUM_FMT = {"m⁻¹": "%.6g", "cm⁻¹": "%.3f", "mm⁻¹": "%.4f"}

# Units a spectral width can be entered in, and what they measure.
WIDTH_UNITS = {"nm": "wavelength", "pm": "wavelength", "GHz": "frequency",
               "MHz": "frequency", "cm⁻¹": "wavenumber"}


# ─── Positions ────────────────────────────────────────────────────────────────
def freq_thz(wl_nm: float) -> float:
    return C_NM_THZ / wl_nm


def wnum_cm(wl_nm: float) -> float:
    return 1e7 / wl_nm


def wl_from_freq(thz: float) -> float:
    return C_NM_THZ / thz


def wl_from_wnum(cm: float) -> float:
    return 1e7 / cm


def photon_energy_ev(wl_nm: float) -> float:
    return HC_EV_NM / wl_nm


def period_fs(wl_nm: float) -> float:
    return 1e3 / freq_thz(wl_nm)


def ang_freq_radfs(wl_nm: float) -> float:
    return 2.0 * math.pi * freq_thz(wl_nm) * 1e-3


# ─── Air / vacuum ─────────────────────────────────────────────────────────────
def refractive_index_air(wl_vac_nm: float) -> float:
    """Refractive index of standard air at a vacuum wavelength.

    The IAU standard formula (Morton 2000, after Edlén 1966), as used by the
    VALD database. Returns 1 below AIR_MIN_NM.
    """
    if wl_vac_nm < AIR_MIN_NM:
        return 1.0
    s2 = (1e3 / wl_vac_nm) ** 2  # (1/λ in µm)²
    return 1.0 + 8.34254e-5 + 2.406147e-2 / (130.0 - s2) + 1.5998e-4 / (38.9 - s2)


def vacuum_to_air(wl_vac_nm: float) -> float:
    """Vacuum → standard-air wavelength (Morton 2000). Unchanged below 200 nm."""
    return wl_vac_nm / refractive_index_air(wl_vac_nm)


def air_to_vacuum(wl_air_nm: float) -> float:
    """Standard-air → vacuum wavelength (N. Piskunov's inverse, VALD).

    Accurate to well below 1 pm against vacuum_to_air. Unchanged below 200 nm.
    """
    if wl_air_nm < AIR_MIN_NM:
        return wl_air_nm
    s2 = (1e3 / wl_air_nm) ** 2
    n = (1.0 + 8.336624212083e-5 + 2.408926869968e-2 / (130.1065924522 - s2)
         + 1.599740894897e-4 / (38.92568793293 - s2))
    return wl_air_nm * n


# ─── Widths ───────────────────────────────────────────────────────────────────
def width_from_nm(dl_nm: float, wl_nm: float) -> Dict[str, float]:
    """Express a spectral width Δλ at vacuum centre wavelength λ in every unit.

    Uses the small-width relations Δν = c·Δλ/λ² and Δν̃ = Δλ/λ². Coherence
    time and length follow the common definition τc = λ²/(c·Δλ) = 1/Δν,
    Lc = c·τc; the exact prefactor depends on the line shape (for example
    0.44/Δν for a Gaussian FWHM), so treat them as order-of-magnitude scales.
    """
    dnu_thz = C_NM_THZ * dl_nm / wl_nm ** 2
    return {
        "dl_nm": dl_nm,
        "dnu_ghz": dnu_thz * 1e3,
        "dwnum_cm": 1e7 * dl_nm / wl_nm ** 2,
        "resolving_power": wl_nm / dl_nm,
        "tau_c_s": 1.0 / (dnu_thz * 1e12),
        "l_c_m": (wl_nm * 1e-9) ** 2 / (dl_nm * 1e-9),
    }


def width_from_ghz(dnu_ghz: float, wl_nm: float) -> Dict[str, float]:
    return width_from_nm(dnu_ghz * 1e-3 * wl_nm ** 2 / C_NM_THZ, wl_nm)


def width_from_cm(dwnum_cm: float, wl_nm: float) -> Dict[str, float]:
    return width_from_nm(dwnum_cm * wl_nm ** 2 / 1e7, wl_nm)


def width_in(value: float, unit: str, wl_nm: float) -> Dict[str, float]:
    """Width entered in one of WIDTH_UNITS → width_from_nm() dict."""
    if unit == "nm":
        return width_from_nm(value, wl_nm)
    if unit == "pm":
        return width_from_nm(value * 1e-3, wl_nm)
    if unit == "GHz":
        return width_from_ghz(value, wl_nm)
    if unit == "MHz":
        return width_from_ghz(value * 1e-3, wl_nm)
    if unit == "cm⁻¹":
        return width_from_cm(value, wl_nm)
    raise ValueError(f"Unknown width unit {unit!r}")


# ─── Reference data ───────────────────────────────────────────────────────────
# Each table carries the medium its wavelengths are quoted in, so a click can be
# converted to the canonical vacuum wavelength correctly.
LASER_LINES = [  # nominal wavelengths, as usually quoted (air, ±0.1 nm)
    ("Argon fluoride excimer (ArF)", 193.0,
     "Deep-ultraviolet lithography and LASIK (laser-assisted in situ keratomileusis) eye surgery"),
    ("Krypton fluoride excimer (KrF)", 248.0,
     "Ultraviolet lithography for semiconductor fabrication (250 nm node era)"),
    ("Xenon chloride excimer (XeCl)", 308.0,
     "Ultraviolet phototherapy and pulsed laser deposition"),
    ("Helium–cadmium (HeCd)", 325.0,
     "Ultraviolet line of the helium–cadmium laser — photolithography, fluorescence"),
    ("Nd:YAG third harmonic", 355.0,
     "Neodymium-doped yttrium aluminium garnet, frequency-tripled in a nonlinear crystal"),
    ("Argon ion, blue", 488.0,
     "Flow cytometry, confocal microscopy, single-molecule fluorescence"),
    ("Argon ion, green", 514.5,
     "Holography and Raman excitation"),
    ("Nd:YAG second harmonic", 532.0,
     "Neodymium-doped yttrium aluminium garnet, frequency-doubled — green laser pointers"),
    ("Helium–neon (HeNe)", 632.8,
     "Red helium–neon laser — alignment and metrology standard, holography"),
    ("Ruby", 694.3,
     "Chromium-doped sapphire — the first laser (Maiman, 1960); dermatology"),
    ("Rubidium D₂", 780.24,
     "Rubidium D₂ transition — laser cooling and frequency references"),
    ("Titanium:sapphire (Ti:Sa)", 800.0,
     "Femtosecond pulses; tunable 690–1080 nm"),
    ("Nd:YAG fundamental", 1064.0,
     "Neodymium-doped yttrium aluminium garnet — the most common solid-state laser"),
    ("Erbium-doped fiber", 1550.0,
     "Telecom C-band — erbium-doped fiber amplifiers and lasers"),
    ("Carbon dioxide (CO₂)", 10600.0,
     "Long-wave infrared — industrial cutting and welding, surgery"),
]

SPECTRAL_LINES = [  # air wavelengths
    ("Calcium K (Ca II)", 393.37,
     "Singly ionised calcium — strong solar absorption, stellar classification"),
    ("Calcium H (Ca II)", 396.85,
     "Singly ionised calcium — solar absorption, G- and K-type stars"),
    ("Hydrogen gamma (Hγ)", 434.05, "Hydrogen Balmer series, n = 5 → 2"),
    ("Hydrogen beta (Hβ)", 486.13,
     "Hydrogen Balmer series, n = 4 → 2 — emission nebulae, ionised hydrogen (H II) regions"),
    ("Magnesium b₁ (Mg I)", 518.36, "Neutral magnesium triplet — solar and stellar spectra"),
    ("Oxygen green (O I)", 557.73, "Neutral oxygen — green aurora and airglow"),
    ("Sodium D₂ (Na I)", 588.995, "Sodium doublet — yellow street lamps"),
    ("Sodium D₁ (Na I)", 589.592, "Sodium doublet — classic optical wavelength standard"),
    ("Oxygen red (O I)", 630.03, "Neutral oxygen — red aurora, upper atmosphere"),
    ("Hydrogen alpha (Hα)", 656.28,
     "Hydrogen Balmer series, n = 3 → 2 — red colour of emission nebulae"),
    ("Nitrogen (N II)", 658.35, "Singly ionised nitrogen — ionised gas nebulae"),
]

# ITU-T single-mode fibre bands
TELECOM_BANDS = [  # vacuum, nm
    ("O-band (Original)", 1260, 1360,
     "First band used — zero dispersion of standard single-mode fiber at 1310 nm"),
    ("E-band (Extended)", 1360, 1460,
     "Needs low-water-peak fiber (hydroxyl absorption peak near 1383 nm)"),
    ("S-band (Short wavelength)", 1460, 1530,
     "Raman and thulium-doped fiber amplification"),
    ("C-band (Conventional)", 1530, 1565,
     "Lowest fiber loss; gain peak of the erbium-doped fiber amplifier (EDFA)"),
    ("L-band (Long wavelength)", 1565, 1625,
     "Extends dense wavelength-division multiplexing (DWDM) capacity"),
    ("U-band (Ultra-long wavelength)", 1625, 1675,
     "Fiber monitoring and maintenance, specialist links"),
]

UV_BANDS = [  # vacuum, nm (ISO 21348)
    ("Extreme ultraviolet (EUV)", 10, 121,
     "Lithography at 13.5 nm; absorbed by every material, reflective optics only"),
    ("Vacuum ultraviolet (VUV)", 121, 200,
     "Absorbed by oxygen in air — needs vacuum or a nitrogen purge"),
    ("Ultraviolet C (UV-C)", 200, 280, "Germicidal — blocked by the ozone layer"),
    ("Ultraviolet B (UV-B)", 280, 315, "Sunburn and vitamin D synthesis"),
    ("Ultraviolet A (UV-A)", 315, 400, "Tanning; the least biologically active UV"),
]

IR_BANDS = [  # vacuum, nm
    ("Near infrared (NIR)", 750, 1400,
     "Silicon detectors to ~1100 nm; low tissue absorption — biological imaging window"),
    ("Short-wave infrared (SWIR)", 1400, 3000,
     "Indium gallium arsenide (InGaAs) detectors; contains the telecom bands"),
    ("Mid-wave infrared (MWIR)", 3000, 8000,
     "Thermal imaging of hot objects in the 3–5 µm atmospheric window; "
     "indium antimonide (InSb) and mercury cadmium telluride (HgCdTe) detectors"),
    ("Long-wave infrared (LWIR)", 8000, 15000,
     "Room-temperature thermal imaging in the 8–14 µm atmospheric window; "
     "microbolometers; carbon dioxide laser at 10.6 µm"),
    ("Far infrared (FIR)", 15000, 1_000_000,
     "Molecular rotations, cold dust in astronomy; merges into the terahertz range"),
]


def neon_lines(min_intensity: float = 5000.0) -> List[Tuple[str, float, str]]:
    """Strong Ne I lines from greg's NIST table (air wavelengths, 400–900 nm)."""
    with open(Path(__file__).parent / "nist_neon.json") as fh:
        lines = json.load(fh)["neon_lines"]
    return [
        ("Neon", float(wl), "Neutral neon (Ne I), air wavelength")
        for wl, i in lines
        if float(i) >= min_intensity
    ]


# ─── Display helpers ──────────────────────────────────────────────────────────
def wavelength_to_rgb(wl: float) -> Optional[Tuple[int, int, int]]:
    """Wavelength [nm] → (r, g, b) 0–255; None outside 380–780 nm."""
    if not (380 <= wl <= 780):
        return None
    gamma = 0.8
    if wl < 440:
        r, g, b = -(wl - 440) / (440 - 380), 0.0, 1.0
    elif wl < 490:
        r, g, b = 0.0, (wl - 440) / (490 - 440), 1.0
    elif wl < 510:
        r, g, b = 0.0, 1.0, -(wl - 510) / (510 - 490)
    elif wl < 580:
        r, g, b = (wl - 510) / (580 - 510), 1.0, 0.0
    elif wl < 645:
        r, g, b = 1.0, -(wl - 645) / (645 - 580), 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0
    if wl < 420:
        factor = 0.3 + 0.7 * (wl - 380) / 40
    elif wl >= 700:
        factor = 0.3 + 0.7 * (780 - wl) / 80
    else:
        factor = 1.0

    def adj(c):
        return 0 if c == 0 else int(255 * ((c * factor) ** gamma))

    return adj(r), adj(g), adj(b)


def spectral_region(wl: float) -> str:
    for upper, name in [
        (0.01, "Gamma ray"), (0.1, "Hard X-ray"), (10, "Soft X-ray"),
        (121, "Extreme ultraviolet (EUV)"), (200, "Vacuum ultraviolet (VUV)"),
        (280, "Ultraviolet C (UV-C)"), (315, "Ultraviolet B (UV-B)"),
        (380, "Ultraviolet A (UV-A)"), (420, "Violet"), (450, "Blue-violet"),
        (495, "Blue"), (530, "Cyan-green"), (560, "Green"), (590, "Yellow"),
        (625, "Orange"), (750, "Red"), (780, "Deep red"),
        (1400, "Near infrared (NIR)"), (3000, "Short-wave infrared (SWIR)"),
        (8000, "Mid-wave infrared (MWIR)"), (15000, "Long-wave infrared (LWIR)"),
        (1_000_000, "Far infrared (FIR) / terahertz"),
    ]:
        if wl < upper:
            return name
    return "Microwave"


def find_band(wl: float, bands) -> Optional[str]:
    for name, lo, hi, _ in bands:
        if lo <= wl < hi:
            return name
    return None


def fmt_energy(ev: float) -> str:
    if ev >= 1e3:
        return f"{ev / 1000:.3g} keV"
    if ev >= 1:
        return f"{ev:.4g} eV"
    if ev >= 1e-3:
        return f"{ev * 1000:.3g} meV"
    return f"{ev * 1e6:.3g} µeV"


def fmt_period(fs: float) -> str:
    if fs < 1:
        return f"{fs * 1000:.3g} as"
    if fs < 1e3:
        return f"{fs:.4g} fs"
    if fs < 1e6:
        return f"{fs / 1e3:.4g} ps"
    return f"{fs / 1e6:.4g} ns"


def fmt_omega(radfs: float) -> str:
    if radfs >= 0.001:
        return f"{radfs:.4g} rad/fs"
    return f"{radfs * 1e3:.3g} rad/ps"


def fmt_time(s: float) -> str:
    for scale, unit in ((1e-15, "fs"), (1e-12, "ps"), (1e-9, "ns"), (1e-6, "µs"),
                        (1e-3, "ms")):
        if s < scale * 1e3:
            return f"{s / scale:.3g} {unit}"
    return f"{s:.3g} s"


def fmt_length(m: float) -> str:
    for scale, unit in ((1e-9, "nm"), (1e-6, "µm"), (1e-3, "mm")):
        if m < scale * 1e3:
            return f"{m / scale:.3g} {unit}"
    if m < 1e3:
        return f"{m:.3g} m"
    return f"{m / 1e3:.3g} km"
