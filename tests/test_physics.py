"""Grating equation, dispersion and the resolution budget, checked against
relations that hold without any reference value."""
import math

import pytest

import physics_core as pc

CASES = [  # (m, lambda_nm, lines_per_mm, alpha_deg)
    (1, 640.0, 1200, 30.0),
    (1, 650.0, 2400, 45.0),
    (2, 500.0, 600, 10.0),
    (-1, 500.0, 600, 10.0),
]


@pytest.mark.parametrize("m, lam, g, alpha", CASES)
def test_output_angle_satisfies_grating_equation(m, lam, g, alpha):
    beta = pc.output_angle(m, lam, g, alpha)
    d_nm = 1e6 / g
    lhs = m * lam / d_nm
    rhs = math.sin(math.radians(alpha)) + math.sin(math.radians(beta))
    assert lhs == pytest.approx(rhs, abs=1e-12)


@pytest.mark.parametrize("m, lam, g", [(1, 640.0, 1200), (1, 650.0, 2400), (2, 500.0, 600)])
def test_littrow_returns_beam_along_itself(m, lam, g):
    alpha = pc.littrow_config(m, lam, g)
    assert pc.output_angle(m, lam, g, alpha) == pytest.approx(alpha, abs=1e-9)


def test_missing_order_raises():
    with pytest.raises(ValueError):
        pc.output_angle(3, 900.0, 2400, 10.0)
    with pytest.raises(ValueError):
        pc.nm_per_pixel(lines_per_mm=2400, alpha_deg=10.0, lambda_nm=900.0, m=3)


def test_order_zero_raises():
    with pytest.raises(ValueError):
        pc.nm_per_pixel(m=0)


@pytest.mark.parametrize("m, lam, g, alpha", CASES)
def test_nm_per_pixel_matches_numerical_dispersion(m, lam, g, alpha):
    """Beam-axis lens: one pixel is an angle p/f, and dλ/dβ = d·cosβ/m."""
    p, f = 26.2e-6, 0.2
    analytic = pc.nm_per_pixel(p=p, lines_per_mm=g, alpha_deg=alpha,
                               lambda_nm=lam, m=m, f=f)
    d_nm = 1e6 / g
    beta = math.radians(pc.output_angle(m, lam, g, alpha))
    dbeta = p / f

    def lam_of(b):
        return d_nm * (math.sin(math.radians(alpha)) + math.sin(b)) / m

    numeric = (lam_of(beta + dbeta / 2) - lam_of(beta - dbeta / 2))
    assert abs(analytic) == pytest.approx(abs(numeric), rel=1e-4)


def test_normal_axis_adds_cos_squared():
    kw = dict(p=26.2e-6, lines_per_mm=1200, alpha_deg=30.0, lambda_nm=640.0, m=1, f=0.2)
    beam = pc.nm_per_pixel(lens_axis=pc.LENS_AXIS_BEAM, **kw)
    normal = pc.nm_per_pixel(lens_axis=pc.LENS_AXIS_NORMAL, **kw)
    cos_b = math.cos(math.radians(pc.output_angle(1, 640.0, 1200, 30.0)))
    assert normal == pytest.approx(beam * cos_b ** 2, rel=1e-12)


def test_line_width_budget_adds_in_quadrature():
    out = pc.line_width_budget(extra_sigma_um=5.0, extra_sigma_urad=20.0)
    terms = ["optics", "angular", "pixel", "extra"]
    total = math.sqrt(sum(out[f"sigma_{t}_um"] ** 2 for t in terms))
    assert out["sigma_total_um"] == pytest.approx(total, rel=1e-12)
    assert out["resolution_nm"] == pytest.approx(
        max(out["fwhm_nm"], out["sampling_limit_nm"]), rel=1e-12
    )
    assert out["limiting"] in {"sampling", *terms}
