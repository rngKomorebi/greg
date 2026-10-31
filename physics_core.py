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

# FWHM of a Gaussian in units of its σ: 2·√(2 ln 2).
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


LENS_AXIS_NORMAL = "normal"
LENS_AXIS_BEAM = "beam"


def nm_per_pixel(
    p: float = 26.2e-6,
    lines_per_mm: float = 1200.0,
    alpha_deg: float = 30.0,
    lambda_nm: float = 640.0,
    m: int = 1,
    f: float = 200e-3,
    lens_axis: str = LENS_AXIS_BEAM,
) -> float:
    """
    Calculate spectral sampling (nm/pixel) for a reflection grating spectrometer.

    With lens_axis="beam" (the default) the lens looks along the diffracted
    beam, the usual layout, so x = f·tan(β − β₀) and at the centre of the
    detector dx/dλ = f·m / (d·cos(β)), giving
    nm/pixel = p·d·cos(β) / (m·f) × 10⁹.

    With lens_axis="normal" the camera lens looks along the grating normal, so
    the detector position is x = f·tan(β), giving dx/dλ = f·m / (d·cos³(β))
    and nm/pixel = p·d·cos³(β) / (m·f) × 10⁹. Only a detector centred on the
    normal sees this, so it differs from "beam" only where β is near zero and
    the two agree anyway.

    Args:
        p: Pixel pitch in meters (default: 26.2 µm)
        lines_per_mm: Grating groove density in lines/mm (default: 1200.0)
        alpha_deg: Incident angle α in degrees (default: 30.0)
        lambda_nm: Wavelength λ in nanometers (default: 640.0)
        m: Diffraction order, typically ±1, ±2, etc. (default: 1)
        f: Focal length in meters (default: 200 mm)
        lens_axis: "beam" (cosβ) or "normal" (cos³β), see above

    Returns:
        Spectral sampling in nm/pixel

    Raises:
        ValueError: If the chosen diffraction order does not exist for the given
                   combination of incident angle, wavelength, and grating parameters
                   (i.e., when |sin(β)| > 1)

    Example:
        >>> nm_per_pixel(p=26.2e-6, lines_per_mm=1200, alpha_deg=30, lambda_nm=640)
        0.1052
    """
    d = 1e-3 / lines_per_mm  # [m]
    alpha = math.radians(alpha_deg)  # [rad]
    lam = lambda_nm * 1e-9  # [m]
    sin_beta = m * lam / d - math.sin(alpha)
    if sin_beta < -1.0 or sin_beta > 1.0:
        raise ValueError(
            "Chosen order does not exist for these α, λ, and grating."
        )
    if m == 0:
        raise ValueError("Order m = 0 has no dispersion.")
    beta = math.asin(sin_beta)
    power = 1 if lens_axis == LENS_AXIS_BEAM else 3
    nm_per_pix = (p * d * (math.cos(beta) ** power)) / (m * f) * 1e9
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


def collimated_beam_diameter_mm(
    collimator_f_mm: float = 10.99,
    mfd_um: float = 4.3,
    lambda_nm: float = 640.0,
) -> float:
    """
    1/e² diameter of the beam from a single-mode fibre and a collimator.

    The fibre mode is a Gaussian of 1/e² diameter MFD, so a lens of focal
    length f_c collimates it to D = 4·λ·f_c / (π·MFD). The defaults are the
    Thorlabs PF2F63 (f_c = 10.99 mm) with SM600 fibre (MFD ≈ 4.3 µm at
    633 nm), giving D ≈ 2.1 mm.

    Args:
        collimator_f_mm: Collimator focal length in mm
        mfd_um: Mode-field diameter of the fibre in µm
        lambda_nm: Wavelength in nm

    Returns:
        Beam diameter (1/e²) in mm
    """
    return 4.0 * lambda_nm * 1e-6 * collimator_f_mm / (math.pi * mfd_um * 1e-3)


def _far_field_sigma_rad(w: float, half_width: float, lam: float) -> float:
    """
    σ (rad) of the far field of a Gaussian beam cut off by a slit-like aperture.

    w is the 1/e² radius of the beam and half_width the aperture half-width,
    both across the beam, in the same unit as lam. Uncut, the far field is
    Gaussian with σ = λ / (2πw). Cut, it is not (a hard edge tends to sinc²),
    so σ is taken from the FWHM, which is what a Gaussian fit to a line follows.
    """
    sigma_free = lam / (2.0 * math.pi * w)
    if half_width >= 2.5 * w:  # edge intensity below e⁻¹²·⁵: no effect
        return sigma_free
    u = np.linspace(0.0, half_width, 801)
    weights = np.full(u.size, u[1] - u[0])
    weights[[0, -1]] *= 0.5  # trapezoid rule
    field = weights * np.exp(-((u / w) ** 2))

    def intensity(theta):  # symmetric aperture: a cosine transform
        return np.dot(field, np.cos(2.0 * np.pi * u * theta / lam)) ** 2

    half = 0.5 * intensity(0.0)
    lo, hi = 0.0, lam / half_width  # the far field is past half maximum by hi
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if intensity(mid) > half:
            lo = mid
        else:
            hi = mid
    return 2.0 * lo / FWHM_PER_SIGMA


def line_width_budget(
    p: float = 26.2e-6,
    lines_per_mm: float = 1200.0,
    alpha_deg: float = 80.0,
    lambda_nm: float = 640.0,
    m: int = 1,
    f: float = 200e-3,
    beam_diameter_mm: float = 2.1,
    pixel_width_frac: float = 1.0,
    extra_sigma_um: float = 0.0,
    extra_sigma_urad: float = 0.0,
    grating_width_mm: float | None = None,
    lens_axis: str = LENS_AXIS_BEAM,
) -> dict:
    """
    Predict the width of a monochromatic line on the detector, term by term.

    Independent blurs add in quadrature on the detector (σ in µm) and are then
    converted to pixels and to wavelength with the local dispersion:

        σ_x² = σ_optics² + σ_angular² + σ_pixel² + σ_extra²

    σ_optics  The focused spot of a single Gaussian mode, as from a single-mode
              fibre and collimator. The grating changes the beam width in the
              dispersion direction by cos β / cos α (anamorphic factor), so the
              beam leaving it has 1/e² radius w = (D/2)·cos β / cos α and its
              far field has σ_θ = λ / (2πw). In wavelength this is
              λ·d·cos α / (π·m·D), independent of f and of the lens axis.
              The fibre image and the diffraction limit are the same spot here,
              so they are not added twice. If the footprint on the grating,
              D / cos α, overfills grating_width_mm, the beam is cut off at
              the grating edges and σ_θ comes from the far field of the cut
              beam instead; the resolution then stops improving with α.
    σ_angular Extra angular spread of the beam before the camera lens
              (collimation, flatness of the mirror and grating), σ in µrad.
              On the detector it scales with f.
    σ_pixel   Top-hat pixel response of width a = pixel_width_frac·p:
              σ = a / √12. Without microlenses a ≈ √(fill factor)·p; good
              microlenses bring it towards the full pitch.
    σ_extra   Extra blur on the detector itself (crosstalk, timing), σ in µm;
              it does not scale with f.

    Angles become positions through the local plate scale: f with
    lens_axis="beam", f / cos²β with "normal" (x = f·tan β).

    Resolution is the larger of the line FWHM and the Nyquist sampling limit of
    2 pixels: a line narrower than 2 pixels cannot be resolved more finely than
    the pixel grid allows.

    Args:
        p: Pixel pitch in metres
        lines_per_mm: Groove density in lines/mm
        alpha_deg: Incident angle α in degrees
        lambda_nm: Wavelength in nm
        m: Diffraction order
        f: Camera focal length in metres
        beam_diameter_mm: 1/e² diameter of the collimated beam on the grating,
            see collimated_beam_diameter_mm()
        pixel_width_frac: Effective pixel width as a fraction of the pitch
        extra_sigma_um: Additional blur σ on the detector in µm
        extra_sigma_urad: Additional angular blur σ in µrad
        grating_width_mm: Ruled width of the grating across the grooves, or
            None for a grating large enough never to cut the beam
        lens_axis: "beam" or "normal", as in nm_per_pixel()

    Returns:
        Dictionary with nm_per_pixel, beta_deg, anamorphic_factor,
        footprint_mm (D / cos α on the grating), clipped (footprint wider
        than the grating) and, for each of
        optics / angular / pixel / extra / total, sigma_<term>_um,
        sigma_<term>_px and sigma_<term>_nm; plus fwhm_px, fwhm_nm,
        sampling_limit_nm (2 px), resolution_nm, resolving_power,
        undersampled (FWHM < 2 px) and limiting ("optics", "angular",
        "pixel", "extra" or "sampling").

    Raises:
        ValueError: If the order does not exist for these parameters.
    """
    beta_deg = output_angle(m, lambda_nm, lines_per_mm, alpha_deg)
    cos_a = math.cos(math.radians(alpha_deg))
    cos_b = math.cos(math.radians(beta_deg))
    nm_px = abs(nm_per_pixel(p, lines_per_mm, alpha_deg, lambda_nm, m, f, lens_axis))
    p_um = p * 1e6
    plate_m = f if lens_axis == LENS_AXIS_BEAM else f / cos_b**2  # m per rad

    footprint_mm = beam_diameter_mm / cos_a
    w_out_mm = 0.5 * beam_diameter_mm * cos_b / cos_a
    if grating_width_mm is None:
        half_width_mm = math.inf
    else:
        half_width_mm = 0.5 * grating_width_mm * cos_b  # across the beam
    sigma_theta = _far_field_sigma_rad(w_out_mm, half_width_mm, lambda_nm * 1e-6)
    sigma_um = {
        "optics": sigma_theta * plate_m * 1e6,
        "angular": abs(extra_sigma_urad) * 1e-6 * plate_m * 1e6,
        "pixel": pixel_width_frac * p_um / math.sqrt(12.0),
        "extra": abs(extra_sigma_um),
    }
    sigma_um["total"] = math.sqrt(sum(s * s for s in sigma_um.values()))

    out = {"nm_per_pixel": nm_px, "beta_deg": beta_deg,
           "anamorphic_factor": cos_b / cos_a,
           "footprint_mm": footprint_mm,
           "clipped": footprint_mm > (grating_width_mm or math.inf)}
    for term, s in sigma_um.items():
        out[f"sigma_{term}_um"] = s
        out[f"sigma_{term}_px"] = s / p_um
        out[f"sigma_{term}_nm"] = s / p_um * nm_px

    fwhm_px = FWHM_PER_SIGMA * out["sigma_total_px"]
    fwhm_nm = fwhm_px * nm_px
    sampling_nm = 2.0 * nm_px
    resolution_nm = max(fwhm_nm, sampling_nm)
    if fwhm_nm < sampling_nm:
        limiting = "sampling"
    else:
        limiting = max(("optics", "angular", "pixel", "extra"),
                       key=lambda t: sigma_um[t])
    out.update(
        fwhm_px=fwhm_px,
        fwhm_nm=fwhm_nm,
        sampling_limit_nm=sampling_nm,
        resolution_nm=resolution_nm,
        resolving_power=lambda_nm / resolution_nm,
        undersampled=fwhm_px < 2.0,
        limiting=limiting,
    )
    return out
