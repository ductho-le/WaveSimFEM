"""Acoustic source conservation and incompatible boundary-data checks."""

from __future__ import annotations

from math import erf

import numpy as np
import pytest

from wavesimfem.acoustic_time import CoupledNewmark, CoupledRK4
from wavesimfem.acoustics import build_acoustic_problem
from wavesimfem.config import parse_and_resolve_config


def _source_config():
    return {
        "solver_type": "explicit",
        "device": "cpu",
        "CFL": 0.5,
        "duration": 6e-6,
        "sensing_length": 0.01,
        "absorb_length": 0.0,
        "absorb_alpha": 0.0,
        "absorb_beta": 0.0,
        "f_center": 1e5,
        "source_type": "tone_burst",
        "n_cycles": 2,
        "source_x": 0.004,
        "excitation_width": 0.006,
        "source_kind": "acoustic_volume",
        "source_amplitude": 1.5e6,
        "source_region": "water",
        "sensor_y": 0.0005,
        "sensor_offset": 0.001,
        "sensor_pitch": 0.001,
        "n_sensors": 3,
        "nx": 16,
        "ny_per_region": [2, 2],
        "degree": 1,
        "structure_mode": "N_layers",
        "layers": [
            {"name": "water", "thickness": 0.001, "rho": 1000.0, "c": 1500.0},
            {"name": "oil", "thickness": 0.001, "rho": 800.0, "c": 1200.0},
        ],
        "run_postprocess": False,
        "write_vtx": False,
    }


@pytest.mark.parametrize("method", [CoupledRK4, CoupledNewmark])
def test_volume_source_satisfies_closed_cavity_integrated_pressure_balance(method):
    """Integral(p/K)'' = Integral(q) across two fluids with unequal K and rho.

    The oracle integrates the specified Gaussian in space using erf and the
    two-cycle tone burst in time using its three sine harmonics. It does not
    reuse the production waveform or forcing vector as its reference.
    """
    cfg = parse_and_resolve_config(custom_cfg=_source_config())
    problem = build_acoustic_problem(cfg)
    end, steps = cfg["duration"], 240
    dt = end / steps
    stepper = (
        method(problem.operators, dt, forcing=problem.forcing)
        if method is CoupledNewmark
        else method(problem.operators, forcing=problem.forcing)
    )
    weighted = problem.operators.Mp.createVecLeft()
    try:
        for i in range(steps):
            stepper.step(i * dt, dt)
        # exp(-(x-c)^2 / (2 sigma^2)) = exp(-((x-c)/width)^2) with width = sqrt(2) sigma.
        width, center = np.sqrt(2.0) * cfg["excitation_width"] / 3.0, cfg["source_x"]
        spatial_integral = (
            0.001
            * width
            * np.sqrt(np.pi)
            / 2
            * (erf((0.01 - center) / width) - erf(-center / width))
        )
        omega = 2 * np.pi * cfg["f_center"]
        envelope_omega = omega / 2
        frequencies = np.array([omega, omega + envelope_omega, omega - envelope_omega])
        coefficients = np.array([0.5, -0.25, -0.25])
        first_integral = np.sum(coefficients * (1 - np.cos(frequencies * end)) / frequencies)
        second_integral = np.sum(
            coefficients * (end / frequencies - np.sin(frequencies * end) / frequencies**2)
        )
        tolerance = 8e-5 if method is CoupledNewmark else 2e-8
        for state, temporal_integral in (
            (stepper.p, second_integral),
            (stepper.pdot, first_integral),
        ):
            problem.operators.Mp.mult(state, weighted)
            expected = cfg["source_amplitude"] * spatial_integral * temporal_integral
            assert weighted.sum() == pytest.approx(expected, rel=tolerance, abs=0.0)
    finally:
        weighted.destroy()
        stepper.destroy()
        problem.destroy()


@pytest.mark.parametrize(
    "source_kind,boundary_kind,message",
    [
        ("acoustic_pressure", "pressure_release", "driven pressure face"),
        ("acoustic_pressure", "radiation", "driven pressure face"),
        ("acoustic_normal_acceleration", "radiation", "normal-acceleration face"),
        ("acoustic_normal_acceleration", "pressure_release", "normal-acceleration face"),
    ],
)
def test_acoustic_boundary_drive_rejects_incompatible_boundary_condition(
    source_kind, boundary_kind, message
):
    raw = _source_config()
    raw.update(
        source_kind=source_kind,
        source_region="oil",
        source_boundary="top",
        acoustic_boundaries={"top": boundary_kind},
    )
    with pytest.raises(ValueError, match=message):
        parse_and_resolve_config(custom_cfg=raw)
    # Assembly also enforces the boundary conflict directly when passed a mutated dict.
    valid = _source_config()
    valid.update(source_kind=source_kind, source_region="oil", source_boundary="top")
    cfg = parse_and_resolve_config(custom_cfg=valid)
    cfg["acoustic_boundaries"]["top"] = boundary_kind
    with pytest.raises(ValueError, match=message):
        build_acoustic_problem(cfg)


def test_driven_pressure_rejects_incompatible_release_data_at_shared_corner():
    raw = _source_config()
    raw.update(
        source_kind="acoustic_pressure",
        source_region="oil",
        source_boundary="top",
        acoustic_boundaries={"left": "pressure_release"},
    )
    with pytest.raises(ValueError, match="Driven pressure intersects a pressure-release boundary"):
        parse_and_resolve_config(custom_cfg=raw)
    valid = _source_config()
    valid.update(source_kind="acoustic_pressure", source_region="oil", source_boundary="top")
    cfg = parse_and_resolve_config(custom_cfg=valid)
    cfg["acoustic_boundaries"]["left"] = "pressure_release"
    with pytest.raises(ValueError, match="Driven pressure intersects a pressure-release boundary"):
        build_acoustic_problem(cfg)


def test_pre_elimination_gll_mass_lumping_and_lifting_at_dirichlet_faces():
    """Explicit GLL lumping precedes Dirichlet elimination and omits Mp_fD lifting."""
    from wavesimfem.acoustic_time import CoupledArrayRK4

    raw = {
        "solver": {"type": "explicit", "device": "cpu", "CFL": 0.5, "duration": 2e-4},
        "geometry": {"sensing_length": 1.0, "absorb_length": 0.0},
        "absorbing_boundaries": {"alpha": 0.0, "beta": 0.0},
        "structure": {
            "mode": "1_layer",
            "single_layer": {
                "name": "unit_fluid",
                "thickness": 1.0,
                "rho": 1.0,
                "c": 1.0,
                "material_model": "fluid",
            },
        },
        "excitation": {
            "source_kind": "acoustic_pressure",
            "source_boundary": "left",
            "source_region": "unit_fluid",
            "source_amplitude": 1.0,
            "source_type": "tone_burst",
            "n_cycles": 2,
            "f_center": 100.0,
        },
        "mesh": {"nx": 1, "ny_per_region": [1], "degree": 2, "max_frequency": 200.0},
        "sensors": {
            "observable": "pressure",
            "n_sensors": 2,
            "sensor_offset": 0.2,
            "sensor_pitch": 0.3,
        },
        "postprocess": {"run_postprocess": False},
    }
    problem = build_acoustic_problem(parse_and_resolve_config(custom_cfg=raw))
    try:
        # On [0,1]^2 with Q2 and x=0 eliminated, horizontal GLL weights at x=0.5 and x=1.0
        # are 2/3 and 1/6 (not post-elimination row sums 3/5 and 1/5).
        x = problem.p_coords[:, 0]
        y = problem.p_coords[:, 1]
        wx = np.where(np.isclose(x, 0.5), 2.0 / 3.0, 1.0 / 6.0)
        wy = np.where(np.isclose(y, 0.5), 2.0 / 3.0, 1.0 / 6.0)
        np.testing.assert_allclose(problem.operators.lumped_mp.array_r, wx * wy, rtol=1e-14)
        # Consistent forcing retains -g'' Mp_fD P_D; lumped explicit forcing omits it.
        assert problem.forcing.pressure[1] is not None
        assert problem.forcing.lumped_pressure == (
            problem.forcing.pressure[0],
            None,
            problem.forcing.pressure[2],
        )
        petsc = CoupledRK4(problem.operators, forcing=problem.forcing)
        array = CoupledArrayRK4(problem.operators, forcing=problem.forcing, device="cpu")
        try:
            dt = 0.5 * petsc.stable_dt()
            for step in range(25):
                petsc.step(step * dt, dt)
                array.step(step * dt, dt)
            np.testing.assert_allclose(array.p.array_r, petsc.p.array_r, rtol=1e-12, atol=1e-14)
        finally:
            array.destroy()
            petsc.destroy()
    finally:
        problem.destroy()


def test_explicit_prescribed_pressure_and_release_spatial_convergence():
    """Explicit RK4 with nonzero prescribed pressure and pressure-release converges in L2/H1."""
    import ufl
    from dolfinx import fem

    from wavesimfem.acoustics import AcousticSampler, acoustic_waveform

    length, height, rho, speed, amplitude = 1.0, 0.1, 1000.0, 1500.0, 1000.0
    t_end = 1.35 * length / speed

    def exact_pressure(x: np.ndarray, cfg: dict) -> np.ndarray:
        total = np.zeros_like(x, dtype=float)
        max_m = int(np.ceil(speed * t_end / (2.0 * length)) + 1)
        for m in range(max_m):
            tau_plus = t_end - (2.0 * m * length + x) / speed
            tau_minus = t_end - (2.0 * (m + 1) * length - x) / speed
            g_plus = np.array([acoustic_waveform(float(tp), cfg)[0] for tp in tau_plus])
            g_minus = np.array([acoustic_waveform(float(tm), cfg)[0] for tm in tau_minus])
            total += amplitude * (g_plus - g_minus)
        return total

    for degree, min_l2_order, min_h1_order in ((1, 1.9, 0.95), (2, 2.8, 1.9)):
        l2_errors, h1_errors = [], []
        for nx in (8, 16, 32):
            raw = {
                "solver": {"type": "explicit", "device": "cpu", "CFL": 0.25, "duration": t_end},
                "geometry": {"sensing_length": length, "absorb_length": 0.0},
                "absorbing_boundaries": {"alpha": 0.0, "beta": 0.0},
                "structure": {
                    "mode": "1_layer",
                    "single_layer": {
                        "name": "water",
                        "thickness": height,
                        "rho": rho,
                        "c": speed,
                        "material_model": "fluid",
                    },
                },
                "acoustic_boundaries": {"right": "pressure_release"},
                "excitation": {
                    "source_kind": "acoustic_pressure",
                    "source_boundary": "left",
                    "source_region": "water",
                    "source_amplitude": amplitude,
                    "source_type": "tone_burst",
                    "n_cycles": 3,
                    "f_center": 600.0,
                },
                "mesh": {
                    "nx": nx,
                    "ny_per_region": [1],
                    "degree": degree,
                    "max_frequency": 1200.0,
                },
                "sensors": {
                    "observable": "pressure",
                    "n_sensors": 3,
                    "sensor_offset": 0.2,
                    "sensor_pitch": 0.2,
                },
                "postprocess": {"run_postprocess": False},
            }
            cfg = parse_and_resolve_config(custom_cfg=raw)
            problem = build_acoustic_problem(cfg)
            try:
                stepper = CoupledRK4(problem.operators, forcing=problem.forcing)
                sampler = AcousticSampler(problem)
                try:
                    steps = max(200, int(np.ceil(t_end / (0.15 * stepper.stable_dt()))))
                    dt = t_end / steps
                    for i in range(steps):
                        stepper.step(i * dt, dt)
                    sampler.update(stepper, t_end, 0.0)
                    ref_space = fem.functionspace(problem.domain, ("Q", degree + 3))
                    p_ref = fem.Function(ref_space)
                    p_ref.x.array[:] = exact_pressure(
                        ref_space.tabulate_dof_coordinates()[:, 0], cfg
                    )
                    diff = sampler.pressure - p_ref
                    dx_hi = ufl.Measure(
                        "dx",
                        domain=problem.domain,
                        metadata={"quadrature_degree": 2 * (degree + 3)},
                    )
                    l2_errors.append(
                        float(
                            np.sqrt(
                                fem.assemble_scalar(fem.form(ufl.inner(diff, diff) * dx_hi))
                                / fem.assemble_scalar(fem.form(ufl.inner(p_ref, p_ref) * dx_hi))
                            )
                        )
                    )
                    h1_errors.append(
                        float(
                            np.sqrt(
                                fem.assemble_scalar(
                                    fem.form(ufl.inner(ufl.grad(diff), ufl.grad(diff)) * dx_hi)
                                )
                                / fem.assemble_scalar(
                                    fem.form(ufl.inner(ufl.grad(p_ref), ufl.grad(p_ref)) * dx_hi)
                                )
                            )
                        )
                    )
                finally:
                    sampler.destroy()
                    stepper.destroy()
                if degree == 2 and nx == 16:
                    # Shared AcousticProblem also supports consistent-mass CoupledNewmark.
                    nm_steps = 600
                    nm_dt = t_end / nm_steps
                    newmark = CoupledNewmark(problem.operators, nm_dt, forcing=problem.forcing)
                    try:
                        for i in range(nm_steps):
                            newmark.step(i * nm_dt, nm_dt)
                        p_nodal = exact_pressure(problem.p_coords[:, 0], cfg)
                        rel_nm = np.linalg.norm(newmark.p.array_r - p_nodal) / np.linalg.norm(
                            p_nodal
                        )
                        assert rel_nm < 6e-4
                    finally:
                        newmark.destroy()
            finally:
                problem.destroy()
        assert np.log2(l2_errors[-2] / l2_errors[-1]) > min_l2_order, l2_errors
        assert np.log2(h1_errors[-2] / h1_errors[-1]) > min_h1_order, h1_errors
