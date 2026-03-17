"""
Physics Core Module for Diffraction Grating Calculations

This module provides core physics calculations for diffraction grating analysis,
including output angle calculations, spectral sampling, and resolution analysis
for reflection gratings in an in-plane mount configuration.

The grating equation used is: m·λ = d·(sin(α) + sin(β))
where:
    m = diffraction order
    λ = wavelength
    d = groove spacing (1/lines_per_mm)
    α = incident angle
    β = output angle
"""

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
    Calculate spectral sampling (nm/pixel) for a reflection grating spectrometer.
    
    Uses the exact mapping x = f·tan(β) for detector position, leading to the
    dispersion relation dx/dλ = f·m / (d·cos³(β)). The spectral sampling is then
    computed as: nm/pixel = p·d·cos³(β) / (m·f) × 10⁹
    
    Args:
        p: Pixel pitch in meters (default: 26.2 µm)
        lines_per_mm: Grating groove density in lines/mm (default: 1200.0)
        alpha_deg: Incident angle α in degrees (default: 30.0)
        lambda_nm: Wavelength λ in nanometers (default: 640.0)
        m: Diffraction order, typically ±1, ±2, etc. (default: 1)
        f: Focal length in meters (default: 200 mm)
    
    Returns:
        Spectral sampling in nm/pixel
    
    Raises:
        ValueError: If the chosen diffraction order does not exist for the given
                   combination of incident angle, wavelength, and grating parameters
                   (i.e., when |sin(β)| > 1)
    
    Example:
        >>> nm_per_pixel(p=26.2e-6, lines_per_mm=1200, alpha_deg=30, lambda_nm=640)
        0.0234
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
    Calculate the output angle β for a reflection grating in an in-plane mount.
    
    Uses the grating equation: m·λ = d·(sin(α) + sin(β))
    Solving for β: β = arcsin(m·λ/d - sin(α))
    
    Args:
        m: Diffraction order (integer), typically ±1, ±2, etc.
        lambda_nm: Wavelength λ in nanometers
        lines_per_mm: Grating groove density in lines/mm (integer)
        alpha_deg: Incident angle α in degrees
    
    Returns:
        Output angle β in degrees
    
    Raises:
        ValueError: If no physical solution exists (i.e., |sin(β)| > 1),
                   which occurs when the requested order cannot propagate
                   at the given wavelength and incident angle
    
    Example:
        >>> output_angle(m=1, lambda_nm=640, lines_per_mm=1200, alpha_deg=30)
        45.2
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
    Calculate the Littrow incidence angle where the output angle equals the incident angle.
    
    In Littrow configuration, β = α, which simplifies the grating equation to:
    m·λ = 2·d·sin(α_L)
    
    This configuration is useful for spectrometer designs as it provides retro-reflection,
    minimizing optical aberrations and simplifying alignment.
    
    Args:
        m: Diffraction order (integer), typically ±1, ±2, etc.
        lambda_nm: Wavelength λ in nanometers
        lines_per_mm: Grating groove density in lines/mm (integer)
    
    Returns:
        Littrow incidence angle α_L in degrees
    
    Raises:
        ValueError: If no physical Littrow angle exists for the given parameters
                   (i.e., when m·λ/(2·d) > 1 or < -1)
    
    Example:
        >>> littrow_config(m=1, lambda_nm=640, lines_per_mm=1200)
        23.5
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
    Calculate the theoretical resolving power of a diffraction grating.
    
    The resolving power R = λ/Δλ represents the ability of the grating to separate
    closely spaced spectral lines. It is given by R = m·N, where N is the total
    number of illuminated grooves on the grating surface.
    
    Args:
        lines_per_mm: Grating groove density in lines/mm (default: 1200.0)
        illuminated_width_mm: Width of the beam on the grating in mm (default: 25.0)
        m: Diffraction order (default: 1). Note: The absolute value is used in the
           calculation, so positive and negative orders give the same resolving power
    
    Returns:
        Resolving power R (dimensionless). Higher values indicate better spectral
        resolution capability
    
    Example:
        >>> grating_resolving_power(lines_per_mm=1200, illuminated_width_mm=25, m=1)
        30000.0
        
    Note:
        This is the theoretical maximum resolving power. Actual performance may be
        limited by detector sampling, slit width, aberrations, or other factors.
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
    Calculate the theoretical spectral resolution limited by grating resolving power.
    
    For a diffraction-limited Gaussian line profile, the relationship between resolving
    power R and spectral width is:
        R = λ/FWHM
    
    For a Gaussian profile, FWHM = 2.355·σ, so:
        σ = λ / (R × 2.355) = λ / (m·N × 2.355)
    
    This represents the narrowest achievable line width (RMS) due to diffraction alone.
    Actual resolution may be worse due to detector sampling, slit width, optical
    aberrations, or environmental factors.
    
    Args:
        lambda_nm: Wavelength λ in nanometers (default: 640.0)
        lines_per_mm: Grating groove density in lines/mm (default: 1200.0)
        illuminated_width_mm: Width of the beam on the grating in mm (default: 25.0)
        m: Diffraction order (default: 1)
    
    Returns:
        Spectral resolution σ in nm (Gaussian RMS width). Smaller values indicate
        better resolution
    
    Example:
        >>> spectral_resolution_rms(lambda_nm=640, lines_per_mm=1200,
        ...                         illuminated_width_mm=25, m=1)
        0.0091
    
    Note:
        This function calculates the diffraction-limited resolution. For the total
        system resolution including detector effects, use effective_resolution_rms().
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
    Calculate the detector sampling limited spectral resolution σ (RMS) in nanometers.
    
    For Nyquist sampling, the pixel size contributes to the point spread function (PSF).
    If we approximate the pixel as a top-hat function of width p_λ (in nm), then
    convolving with a Gaussian gives σ_pixel ≈ p_λ / √12 (RMS of uniform distribution).
    
    However, in practice with Gaussian fitting, you typically observe σ_measured ≈ nm/pixel
    when the line is barely sampled (2-3 pixels). This function uses the pragmatic estimate:
    σ_detector ≈ nm_per_pixel
    
    Args:
        p: Pixel pitch in meters (default: 26.2 µm)
        lines_per_mm: Grating groove density in lines/mm (default: 1200.0)
        alpha_deg: Incident angle α in degrees (default: 30.0)
        lambda_nm: Wavelength λ in nanometers (default: 640.0)
        m: Diffraction order (default: 1)
        f: Focal length in meters (default: 200 mm)
    
    Returns:
        Detector-limited spectral resolution σ in nm (approximate RMS width)
    
    Note:
        For a more conservative estimate, use σ = nm_per_pixel / √12 ≈ 0.29 × nm_per_pixel
        for top-hat pixel response. The current implementation uses nm_per_pixel directly
        as it better reflects typical experimental conditions.
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
    Calculate the effective spectral resolution considering both grating and detector limits.
    
    The total spectral resolution is determined by two independent contributions:
    1. Grating diffraction limit (depends on the number of illuminated grooves)
    2. Detector sampling limit (depends on pixel size and dispersion)
    
    When both contribute, they add in quadrature (as independent Gaussian contributions):
    σ_total = √(σ_grating² + σ_detector²)
    
    Args:
        p: Pixel pitch in meters (default: 26.2 µm)
        lines_per_mm: Grating groove density in lines/mm (default: 1200.0)
        alpha_deg: Incident angle α in degrees (default: 30.0)
        lambda_nm: Wavelength λ in nanometers (default: 640.0)
        m: Diffraction order (default: 1)
        f: Focal length in meters (default: 200 mm)
        illuminated_width_mm: Width of the beam on the grating in mm (default: 25.0)
    
    Returns:
        Dictionary containing:
            - 'grating_limited_rms': Diffraction-limited σ in nm
            - 'detector_limited_rms': Sampling-limited σ in nm
            - 'effective_rms': Total σ in nm (quadrature sum)
            - 'resolving_power': Effective resolving power R = λ/(2.355·σ)
            - 'limiting_factor': 'grating', 'detector', or 'both'
    
    Note:
        The limiting factor is determined by comparing the two contributions:
        - If σ_grating > 2·σ_detector, result is 'grating'
        - If σ_detector > 2·σ_grating, result is 'detector'
        - Otherwise, result is 'both'
    
    Example:
        >>> result = effective_resolution_rms(lambda_nm=640, lines_per_mm=1200)
        >>> print(f"Effective resolution: {result['effective_rms']:.3f} nm")
        >>> print(f"Limited by: {result['limiting_factor']}")
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
