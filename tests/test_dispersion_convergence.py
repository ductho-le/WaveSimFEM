"""End-to-end FEM checks against independent continuum dispersion references.

These tests exercise assembly, integration, point sampling, and saved histories.
They deliberately supply mesh, time, and recording parameters: passing them does
not establish that these meshes resolve every guided mode of every model.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.optimize import brentq

from guidedwavefem import run_simulation

E, NU, RHO, H, FREQUENCY = 70e9, 0.33, 2700.0, 1e-3, 100e3


def _rayleigh_lamb_a0() -> float:
    """Solve the dimensional free-surface traction determinant independently.

    With p² = ω²/c_L² - k², q² = ω²/c_T² - k², and half-thickness h,
    the antisymmetric condition is
      (q²-k²)² sin(ph) cos(qh) + 4k²pq cos(ph) sin(qh) = 0.
    See Eq. (2), https://pmc.ncbi.nlm.nih.gov/articles/PMC6631679/.
    This uses complex trigonometric functions, not the production A0 helper,
    its sub-shear hyperbolic form, or its asymptotic approximations.
    The benchmark frequency is below the first higher-mode cutoff.
    """
    c_t = np.sqrt(E / (2 * (1 + NU) * RHO))
    c_l = np.sqrt(E * (1 - NU) / ((1 + NU) * (1 - 2 * NU) * RHO))
    omega = 2 * np.pi * FREQUENCY

    def determinant(k: float) -> float:
        p = np.lib.scimath.sqrt((omega / c_l) ** 2 - k**2)
        q = np.lib.scimath.sqrt((omega / c_t) ** 2 - k**2)
        value = (q**2 - k**2) ** 2 * np.sin(p * H / 2) * np.cos(q * H / 2)
        value += 4 * k**2 * p * q * np.cos(p * H / 2) * np.sin(q * H / 2)
        return float(value.imag)

    return brentq(determinant, 1.0001 * omega / c_t, 10 * omega / c_t, xtol=1e-10)


def _measure_propagation(path, component: str) -> tuple[float, float, float, float]:
    """Fit harmonic phase and log-amplitude versus distance, without FFT bins."""
    with np.load(path) as data:
        times, x = data["times"], data["sensor_x"]
        history = data[f"sensor_data_{component}"]
        dt = float(data["dt"])
    harmonic = trapezoid(history * np.exp(-2j * np.pi * FREQUENCY * times)[:, None], times, axis=0)
    assert np.isfinite(harmonic).all() and np.all(np.abs(harmonic) > 0)
    phase = np.unwrap(np.angle(harmonic))
    phase_fit = np.polyfit(x, phase, 1)
    attenuation = -np.polyfit(x, np.log(np.abs(harmonic)), 1)[0]
    residual = np.std(phase - np.polyval(phase_fit, x))
    return -phase_fit[0], attenuation, residual, dt


def _a0_run(output_dir, solver: str, nx: int, rows: int, dt: float | None, cfl=0.5):
    """A transverse surface burst; mid-plane u_y suppresses symmetric modes."""
    path, _ = run_simulation(
        cfg={
            "solver": {"type": solver, "device": "cpu", "duration": 100e-6, "dt": dt, "CFL": cfl},
            "geometry": {"sensing_length": 0.1, "absorb_length": 0.04},
            "structure": {
                "mode": "1_layer",
                "single_layer": {"thickness": H, "rho": RHO, "E": E, "nu": NU},
            },
            "excitation": {
                "f_center": FREQUENCY,
                "source_type": "tone_burst",
                "n_cycles": 5,
                "excitation_width": 0.002,
                "source_fx": 0.0,
                "source_fy": 1.0,
            },
            "mesh": {
                "nx": nx,
                "ny_per_region": [rows],
                "max_frequency": 1.5 * FREQUENCY,
                "degree": 2,
            },
            "sensors": {
                "n_sensors": 41,
                "sensor_offset": 0.02,
                "sensor_pitch": 0.0005,
                "sensor_y": H / 2,
                "sample_rate": 8e6,
            },
            "postprocess": {"run_postprocess": False},
            "output": {"output_dir": str(output_dir)},
        }
    )
    return _measure_propagation(path, "y")


@pytest.mark.slow
@pytest.mark.parametrize("solver", ["explicit", "implicit"])
def test_a0_transient_mesh_convergence(tmp_path, solver):
    """Both solvers converge to the continuum A0 wavenumber as Q2 cells shrink.

    The implicit step is fixed; explicit CFL steps are much smaller than a
    period, so fourth-order temporal dispersion is negligible in this check.
    Spatial phase fitting resolves errors far below a space/time FFT bin.
    """
    reference = _rayleigh_lamb_a0()
    errors = []
    for nx, rows in ((48, 1), (95, 1), (189, 2)):
        k, _, residual, dt = _a0_run(
            tmp_path / str(nx),
            solver,
            nx,
            rows,
            0.125e-6 if solver == "implicit" else None,
        )
        errors.append(abs(k / reference - 1))
        if solver == "explicit":
            assert dt * FREQUENCY < 0.005
    assert errors[1] < 0.3 * errors[0], errors
    assert errors[2] < 0.35 * errors[1], errors
    assert errors[2] < 0.003, errors
    assert residual < 0.025, residual


@pytest.mark.slow
def test_efficient_q2_resolution_preserves_a0_phase(tmp_path):
    """Check phase against continuum and actual work against the former settings.

    Compare separate mesh and time changes; wall-clock thresholds would be noisy.
    This is a below-cutoff free-plate benchmark, not a universal accuracy claim.
    """
    reference = _rayleigh_lamb_a0()
    results = []
    work = []
    for nx, cfl in ((236, 0.5), (142, 0.5), (142, 0.8)):
        output = tmp_path / f"{nx}-{cfl}"
        k, _, residual, dt = _a0_run(output, "explicit", nx, 2, None, cfl)
        assert abs(k / reference - 1) < 0.003
        assert residual < 0.025
        with np.load(output / "sensor_data.npz") as data:
            import yaml

            runtime = yaml.safe_load(str(data["simulation_config_yaml"]))["runtime"]
        work.append(runtime["ndofs"] / dt)
        results.append(k)
    assert abs(results[2] / results[1] - 1) < 1e-4
    assert work[2] < 0.5 * work[0]


@pytest.mark.slow
def test_a0_transient_newmark_time_convergence(tmp_path):
    """Halving Newmark's step reduces phase error on one fixed fine mesh."""
    reference = _rayleigh_lamb_a0()
    errors = []
    for dt in (1e-6, 0.5e-6, 0.25e-6):
        k, _, residual, _ = _a0_run(tmp_path / str(dt), "implicit", 189, 2, dt)
        errors.append(abs(k / reference - 1))
        assert residual < 0.025, residual
    # Second-order temporal convergence until finite-mesh/boundary errors dominate.
    # Use generous bounds rather than requiring an asymptotic factor of four.
    assert errors[1] < 0.55 * errors[0], errors
    assert errors[2] < 0.55 * errors[1], errors
    assert errors[2] < 0.003, errors


@pytest.mark.slow
@pytest.mark.parametrize("relaxation_time", [0.0, 3e-7])
def test_orthotropic_extensional_speed_and_kelvin_voigt_attenuation(tmp_path, relaxation_time):
    """Check an orthotropic plate and its proportional Kelvin-Voigt extension.

    In the long-wavelength extensional limit, zero transverse stress gives
    ε_yy = -C12/C22 ε_xx and E_eff = C11 - C12²/C22. With Cv = τ C,
    k = ω sqrt(ρ / (E_eff (1+iωτ))). Thus Re(k) is phase wavenumber and
    -Im(k) is attenuation per metre. This independent continuum reference
    checks engineering-shear conventions, anisotropic assembly, and damping.
    It covers principal-axis orthotropy, not arbitrary rotated/layered media.
    """
    stiffness = np.array([150e9, 8e9, 0, 20e9, 0, 5e9])
    density = 1600.0
    material = {
        "thickness": H,
        "rho": density,
        "material_model": "viscoelastic" if relaxation_time else "anisotropic",
        "C": stiffness.tolist(),
    }
    if relaxation_time:
        material["Cv"] = (relaxation_time * stiffness).tolist()
    path, _ = run_simulation(
        cfg={
            "solver": {"type": "implicit", "duration": 70e-6, "dt": 0.125e-6},
            # Place both ends far enough away that the earliest reflected
            # extensional wave arrives after the complete recording window.
            "geometry": {"sensing_length": 0.6, "absorb_length": 0.2},
            "structure": {"mode": "1_layer", "single_layer": material},
            "excitation": {
                "f_center": FREQUENCY,
                "source_type": "tone_burst",
                "n_cycles": 5,
                "excitation_width": 0.003,
                "source_x": 0.4,
                "source_fx": 1.0,
                "source_fy": 0.0,
            },
            "mesh": {
                "nx": 510,
                "ny_per_region": [2],
                "max_frequency": 1.5 * FREQUENCY,
                "degree": 2,
            },
            "sensors": {
                "n_sensors": 41,
                "sensor_offset": 0.03,
                "sensor_pitch": 0.001,
                "sensor_y": H / 2,
                "sample_rate": 8e6,
            },
            "postprocess": {"run_postprocess": False},
            "output": {"output_dir": str(tmp_path)},
        }
    )
    k, attenuation, residual, _ = _measure_propagation(path, "x")
    effective_modulus = stiffness[0] - stiffness[1] ** 2 / stiffness[3]
    omega = 2 * np.pi * FREQUENCY
    reference = omega * np.sqrt(density / (effective_modulus * (1 + 1j * omega * relaxation_time)))
    assert k == pytest.approx(reference.real, rel=0.002), (k, reference)
    assert residual < 0.01, residual
    if relaxation_time:
        assert attenuation == pytest.approx(-reference.imag, rel=0.02), (attenuation, reference)
    else:
        assert abs(attenuation) < 0.1, attenuation
