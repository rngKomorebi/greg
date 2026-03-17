import math

import numpy as np


def nm_per_pixel(
    p: float = 26.2e-6,
    lines_per_mm: float = 1200.0,
    alpha_deg: float = 30.0,
    lambda_nm: float = 640.0,
    m: int = 1,
    f: float = 200e-3,
) -> float:
    """
    nm/pixel for a reflection grating in an in-plane mount, computed
    from α and λ. Uses the exact mapping x = f tan β, leading to
    dx/dλ = f m / (d cos^3 β). Raises ValueError if the chosen order
    does not exist at the given angles/wavelength.
    """
    d = 1e-3 / lines_per_mm  # [m]
    alpha = math.radians(alpha_deg)  # [rad]
    lam = lambda_nm * 1e-9  # [m]
    sin_beta = m * lam / d - math.sin(alpha)
    if sin_beta < -1.0 or sin_beta > 1.0:
        raise ValueError(
            "Chosen order does not exist for these α, λ, and grating."
        )
    beta = math.asin(sin_beta)
    nm_per_pix = (p * d * (math.cos(beta) ** 3)) / (m * f) * 1e9
    return nm_per_pix


def output_angle(
    m: int, lambda_nm: float, lines_per_mm: int, alpha_deg: float
) -> float:
    """
    Output angle β (degrees) for a reflection grating in an in-plane mount.
    Uses m λ = d (sin α + sin β). Inputs λ in nm, lines_per_mm in lines/mm.
    Raises ValueError if |sinβ|>1.
    """
    d = 1e-3 / lines_per_mm  # [m]
    lam = lambda_nm * 1e-9  # [m]
    alpha = math.radians(alpha_deg)
    s = (m * lam / d) - math.sin(alpha)
    if s < -1.0 or s > 1.0:
        raise ValueError("No physical β (|sinβ|>1) for these parameters.")
    return math.degrees(math.asin(s))


def littrow_config(m: int, lambda_nm: float, lines_per_mm: int) -> float:
    """
    Littrow incidence angle α_L (degrees) where β=α. Condition: m λ = 2 d sin α_L.
    """
    d = 1e-3 / lines_per_mm  # [m]
    lam = lambda_nm * 1e-9  # [m]
    s = m * lam / (2.0 * d)
    if s < -1.0 or s > 1.0:
        raise ValueError("No physical Littrow angle for these parameters.")
    return math.degrees(math.asin(s))


def grating_resolving_power(
    lines_per_mm: float = 1200.0,
    illuminated_width_mm: float = 25.0,
    m: int = 1,
) -> float:
    """
    Theoretical resolving power R = λ/Δλ = m·N where N is the total
    number of illuminated grooves on the grating.

    Args:
        lines_per_mm: groove density
        illuminated_width_mm: width of the beam on the grating
        m: diffraction order

    Returns:
        Resolving power R (dimensionless)
    """
    N = lines_per_mm * illuminated_width_mm  # total illuminated grooves
    return abs(m) * N


def spectral_resolution_rms(
    lambda_nm: float = 640.0,
    lines_per_mm: float = 1200.0,
    illuminated_width_mm: float = 25.0,
    m: int = 1,
) -> float:
    """
    Theoretical spectral resolution σ (RMS, Gaussian sigma) in nm,
    limited by the grating resolving power.

    For a diffraction-limited Gaussian line profile:
    σ = λ / (R × 2.355) = λ / (m·N × 2.355)

    where FWHM = 2.355 × σ for a Gaussian.

    R = λ/FWHM, so σ = λ/(R × 2.355)

    This is the diffraction-limited resolution (RMS). Actual resolution
    may be worse due to slit width, aberrations, pixel sampling, etc.

    Args:
        lambda_nm: wavelength in nm
        lines_per_mm: groove density
        illuminated_width_mm: beam width on grating
        m: diffraction order

    Returns:
        σ in nm (Gaussian RMS width)
    """
    R = grating_resolving_power(lines_per_mm, illuminated_width_mm, m)
    # R = λ/FWHM, and FWHM = 2.355·σ, so σ = λ/(R × 2.355)
    return lambda_nm / (R * 2.355)


def detector_limited_resolution_rms(
    p: float = 26.2e-6,
    lines_per_mm: float = 1200.0,
    alpha_deg: float = 30.0,
    lambda_nm: float = 640.0,
    m: int = 1,
    f: float = 200e-3,
) -> float:
    """
    Detector sampling limited resolution σ (RMS) in nm.

    For Nyquist sampling, the pixel size contributes to the PSF.
    If we approximate the pixel as a top-hat function of width p_λ (in nm),
    then convolving with a Gaussian gives:

    σ_pixel ≈ p_λ / √12  (RMS of uniform distribution)

    where p_λ = nm/pixel is the spectral sampling.

    However, in practice with Gaussian fitting, you typically observe:
    σ_measured ≈ nm/pixel when the line is barely sampled (2-3 pixels).

    This uses the pragmatic estimate: σ_detector ≈ nm_per_pixel

    Returns:
        σ in nm (approximate RMS width from detector sampling)
    """
    nmpp = nm_per_pixel(p, lines_per_mm, alpha_deg, lambda_nm, m, f)
    # Pragmatic: σ ≈ pixel_size for marginally sampled lines
    # More conservative: σ = nmpp / sqrt(12) ≈ 0.29 * nmpp for top-hat pixel
    return nmpp  # Use this for typical experimental conditions


def effective_resolution_rms(
    p: float = 26.2e-6,
    lines_per_mm: float = 1200.0,
    alpha_deg: float = 30.0,
    lambda_nm: float = 640.0,
    m: int = 1,
    f: float = 200e-3,
    illuminated_width_mm: float = 25.0,
) -> dict:
    """
    Calculate the effective spectral resolution σ (RMS), considering both
    grating diffraction limit and detector sampling.

    When both contribute, they add in quadrature:
    σ_total = √(σ_grating² + σ_detector²)

    Returns dict with:
        - 'grating_limited_rms': diffraction-limited σ (nm)
        - 'detector_limited_rms': sampling-limited σ (nm)
        - 'effective_rms': total σ (nm) - quadrature sum
        - 'resolving_power': R = λ/(2.355·σ) (using FWHM definition)
        - 'limiting_factor': 'grating', 'detector', or 'both'
    """
    sigma_grating = spectral_resolution_rms(
        lambda_nm, lines_per_mm, illuminated_width_mm, m
    )
    sigma_detector = detector_limited_resolution_rms(
        p, lines_per_mm, alpha_deg, lambda_nm, m, f
    )

    # Add in quadrature (independent Gaussian contributions)
    sigma_effective = np.sqrt(sigma_grating**2 + sigma_detector**2)

    # Determine limiting factor
    if sigma_grating > 2 * sigma_detector:
        limiting = "grating"
    elif sigma_detector > 2 * sigma_grating:
        limiting = "detector"
    else:
        limiting = "both"

    # Resolving power from effective FWHM
    fwhm_effective = 2.355 * sigma_effective
    R_effective = lambda_nm / fwhm_effective

    return {
        "grating_limited_rms": sigma_grating,
        "detector_limited_rms": sigma_detector,
        "effective_rms": sigma_effective,
        "resolving_power": R_effective,
        "limiting_factor": limiting,
    }
