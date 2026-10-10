"""Independent continuum references for acoustic verification.

These functions use plane-wave potentials, impedance matching, and an exact
elastic layer propagator. They never import the finite-element implementation.
The convention is exp(i*k*x-i*omega*t), with positive compressive pressure.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq


def fluid_impedance_coefficients(rho_1, c_1, rho_2, c_2):
    """Normal-incidence pressure reflection/transmission and energy fractions."""
    z_1, z_2 = rho_1 * c_1, rho_2 * c_2
    reflection = (z_2 - z_1) / (z_2 + z_1)
    transmission = 2 * z_2 / (z_2 + z_1)
    return reflection, transmission, reflection**2, z_1 / z_2 * transmission**2


def fluid_solid_coefficients(rho_f, c_f, rho_s, c_p, c_s, angle):
    """Fluid incidence onto an isotropic solid, including P/S conversion.

    ``angle`` is in radians from the interface normal. The fluid is above the
    solid. The incident pressure amplitude is one Pa. Transmitted amplitudes
    are particle velocities in m/s along the downward P and SV polarizations.
    This 3x3 boundary calculation applies below and above critical angles.
    """
    sx_f = np.sin(angle)
    sx_p, sx_s = c_p / c_f * sx_f, c_s / c_f * sx_f
    cz_f = np.cos(angle)
    cz_p, cz_s = np.lib.scimath.sqrt(1 - sx_p**2), np.lib.scimath.sqrt(1 - sx_s**2)
    mu = rho_s * c_s**2
    lam = rho_s * c_p**2 - 2 * mu
    tau_p, tau_s = 2 * mu * sx_p * cz_p / c_p, mu * (cz_s**2 - sx_s**2) / c_s
    normal_p = -(lam * sx_p**2 + (lam + 2 * mu) * cz_p**2) / c_p
    normal_s = 2 * mu * cz_s * sx_s / c_s
    admittance = cz_f / (rho_f * c_f)
    system = np.array(
        [[admittance, cz_p, -sx_s], [1, normal_p, normal_s], [0, tau_p, tau_s]],
        dtype=complex,
    )
    reflection, v_p, v_s = np.linalg.solve(system, [admittance, -1, 0])
    transmitted_normal = -cz_p * v_p + sx_s * v_s
    transmitted_tangential = sx_p * v_p + cz_s * v_s
    fluid_normal = -admittance * (1 - reflection)
    fluid_tangential = sx_f / (rho_f * c_f) * (1 + reflection)
    return {
        "reflection": reflection,
        "velocity_p": v_p,
        "velocity_s": v_s,
        "normal_velocity_residual": transmitted_normal - fluid_normal,
        "normal_traction_residual": normal_p * v_p + normal_s * v_s + 1 + reflection,
        "tangential_traction": tau_p * v_p + tau_s * v_s,
        "tangential_slip": fluid_tangential - transmitted_tangential,
        "reflected_energy": float(abs(reflection) ** 2),
        "p_energy": float(rho_s * c_p * np.real(cz_p) * abs(v_p) ** 2 / admittance),
        "s_energy": float(rho_s * c_s * np.real(cz_s) * abs(v_s) ** 2 / admittance),
    }


def scholte_speed(rho_f, c_f, rho_s, c_p, c_s):
    """Nonzero subsonic root of the fluid-loaded Rayleigh secular equation."""

    def secular(c):
        s2 = (c / c_s) ** 2
        a = np.sqrt(1 - (c / c_p) ** 2)
        b = np.sqrt(1 - s2)
        f = np.sqrt(1 - (c / c_f) ** 2)
        return (2 - s2) ** 2 - 4 * a * b + rho_f / rho_s * s2**2 * a / f

    upper = min(c_f, c_s) * (1 - 1e-10)
    return brentq(secular, upper * 0.01, upper, xtol=1e-10)


def plate_secular(c, k, thickness, rho_f, c_f, rho_s, c_p, c_s, symmetry="A"):
    """Exact determinant for a plate between identical fluid halfspaces.

    Uses a scaled displacement/traction state propagated from the plate
    midplane to its upper face. Restriction to c < c_f selects nonleaky waves;
    the fluid normal impedance is rho_f*omega**2/sqrt(k**2-omega**2/c_f**2).
    ``A`` selects flexural symmetry; ``S`` selects extensional symmetry.
    """
    if not 0 < c < min(c_f, c_s):
        raise ValueError("The reference requires 0 < phase speed < min(c_f, c_s)")
    ratio = (c_p / c_s) ** 2
    lam_over_l2m = (ratio - 2) / ratio
    s2 = (c / c_s) ** 2
    state_matrix = np.array(
        [
            [0, -1, 1, 0],
            [lam_over_l2m, 0, 0, 1 / ratio],
            [ratio - (ratio - 2) ** 2 / ratio - s2, 0, 0, -lam_over_l2m],
            [0, -s2, 1, 0],
        ]
    )
    propagated = expm(state_matrix * k * thickness / 2)
    columns = [1, 2] if symmetry == "A" else [0, 3]
    z = rho_f / rho_s * s2 / np.sqrt(1 - (c / c_f) ** 2)
    boundary = np.vstack(
        [propagated[2, columns], propagated[3, columns] - z * propagated[1, columns]]
    )
    return float(np.linalg.det(boundary))


def fluid_loaded_plate_speed(k, thickness, rho_f, c_f, rho_s, c_p, c_s, symmetry="A"):
    """Lowest positive subsonic plate root, obtained without FEM matrices."""
    points = np.linspace(0.01, 1 - 1e-8, 600) * min(c_f, c_s)
    kwargs = (k, thickness, rho_f, c_f, rho_s, c_p, c_s, symmetry)
    values = [plate_secular(c, *kwargs) for c in points]
    for a, b, va, vb in zip(points[:-1], points[1:], values[:-1], values[1:], strict=True):
        if va * vb < 0:
            return brentq(plate_secular, a, b, args=kwargs, xtol=1e-9)
    raise ValueError("No subsonic plate root in the requested interval")


def elastic_parameters(rho, c_p, c_s):
    """Convert physical P/S speeds to isotropic E and nu for benchmark inputs."""
    mu = rho * c_s**2
    lam = rho * (c_p**2 - 2 * c_s**2)
    return {"rho": rho, "E": mu * (3 * lam + 2 * mu) / (lam + mu), "nu": lam / (2 * (lam + mu))}
