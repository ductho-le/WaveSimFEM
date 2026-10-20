#!/usr/bin/env python3
"""Reproducible numerical verification of the production acoustic FEM assembly.

Run in the DOLFINx environment, on one MPI rank. References are independent
continuum solutions in acoustic_reference.py. The legacy comparison uses the
existing regularized-elastic production assembly on the same mesh.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from acoustic_reference import (
    elastic_parameters,
    fluid_impedance_coefficients,
    fluid_loaded_plate_speed,
    fluid_solid_coefficients,
    scholte_speed,
)


def benchmark_config(layers, nx, rows, length=1.0, degree=2):
    """Undamped structured mesh with rigid acoustic boundaries."""
    from wavesimfem.config import parse_and_resolve_config

    return parse_and_resolve_config(
        custom_cfg={
            "solver": {"type": "explicit", "device": "cpu", "duration": 0.001, "CFL": 0.7},
            "geometry": {"sensing_length": length - 2e-12, "absorb_length": 1e-12},
            "structure": {"mode": "N_layers", "layers": layers},
            "absorbing_boundaries": {"alpha": 0.0, "beta": 0.0},
            "excitation": {
                "source_kind": "acoustic_normal_acceleration",
                "source_amplitude": 0.0,
                "source_boundary": "top",
                "f_center": 300.0,
                "source_x": 0.1 * length,
                "excitation_width": 0.1 * length,
            },
            "mesh": {"nx": nx, "ny_per_region": rows, "degree": degree, "max_frequency": 500.0},
            "sensors": {
                "n_sensors": 3,
                "sensor_offset": 0.1 * length,
                "sensor_pitch": 0.1 * length,
                "sensor_y": sum(layer["thickness"] for layer in layers)
                - 0.1 * layers[-1]["thickness"],
                "observable": "pressure",
                "region": layers[-1]["name"],
            },
            "postprocess": {"run_postprocess": False},
            "output": {"quick_plot": False, "usetex": False},
        }
    )


def fluid(name="water", thickness=1.0, rho=1000.0, c=1500.0):
    return {"name": name, "thickness": thickness, "material_model": "fluid", "rho": rho, "c": c}


def solid(name="solid", thickness=1.0):
    return {"name": name, "thickness": thickness, **elastic_parameters(2000, 2500, 1200)}


def csr(matrix):
    if matrix is None:
        return None
    if sp.issparse(matrix):
        return sp.csr_matrix(matrix)
    if matrix.getComm().getSize() != 1:
        raise ValueError("Verification matrix extraction requires one MPI rank")
    indptr, indices, data = matrix.getValuesCSR()
    return sp.csr_matrix((data.copy(), indices.copy(), indptr.copy()), shape=matrix.getSize())


def build(cfg):
    from wavesimfem.acoustics import build_acoustic_problem

    return build_acoustic_problem(cfg)


def operators(problem):
    op = problem.operators
    return {name: csr(getattr(op, name)) for name in ("Ms", "Ks", "Mp", "Kp", "C")}


def coupled_matrices(problem):
    op = operators(problem)
    ms, ks, mp, kp, c = (op[n] for n in ("Ms", "Ks", "Mp", "Kp", "C"))
    if ms is None or not ms.shape[0]:
        return kp, mp
    return (
        sp.bmat([[ks, c], [None, kp]], format="csr"),
        sp.bmat([[ms, None], [-c.T, mp]], format="csr"),
    )


def dirichlet_solve(matrix, indices, values):
    indices = np.asarray(indices, dtype=int)
    free = np.setdiff1d(np.arange(matrix.shape[0]), indices)
    solution = np.zeros(matrix.shape[0], dtype=np.result_type(matrix.dtype, values))
    solution[indices] = values
    solution[free] = sla.spsolve(matrix[free][:, free], -matrix[free][:, indices] @ values)
    return solution


def relative_error(value, reference):
    return float(np.linalg.norm(value - reference) / np.linalg.norm(reference))


def cavity_check():
    """Consistent-mass Q1/Q2 cavity eigenfrequencies, including the zero mode."""
    results = []
    for degree in (1, 2):
        errors = []
        for n in (4, 8, 16):
            problem = build(benchmark_config([fluid()], n, [n], degree=degree))
            op = operators(problem)
            eigenvalues = la.eigvalsh(
                op["Kp"].toarray(), op["Mp"].toarray(), subset_by_index=[0, 3]
            )
            exact = 1500 / 2
            measured = np.sqrt(eigenvalues[1]) / (2 * np.pi)
            error = abs(measured / exact - 1)
            errors.append(error)
            constant_residual = float(np.linalg.norm(op["Kp"] @ np.ones(op["Kp"].shape[0])))
            results.append(
                {
                    "degree": degree,
                    "elements_per_side": n,
                    "frequency_hz": measured,
                    "relative_error": error,
                    "constant_mode_residual": constant_residual,
                    "pressure_dofs": op["Mp"].shape[0],
                }
            )
            problem.destroy()
        order = float(np.log(errors[-2] / errors[-1]) / np.log(2))
        assert order > (1.8 if degree == 1 else 3.8), (degree, order)
        assert errors[-1] < (0.002 if degree == 1 else 2e-6)
        results[-1]["observed_order"] = order
    return {"status": "pass", "reference": "f_10=c/(2L), rigid square cavity", "runs": results}


def fluid_fluid_check():
    layers = [fluid("lower", rho=1000, c=1500), fluid("upper", rho=650, c=2100)]
    problem = build(benchmark_config(layers, 3, [24, 24]))
    op = operators(problem)
    xy = problem.p_coords
    y = xy[:, 1] - 1
    omega = 2 * np.pi * 500
    reflection, transmission, er, et = fluid_impedance_coefficients(1000, 1500, 650, 2100)
    exact = np.where(
        y <= 0,
        np.exp(1j * omega / 1500 * y) + reflection * np.exp(-1j * omega / 1500 * y),
        transmission * np.exp(1j * omega / 2100 * y),
    )
    fixed = np.flatnonzero(np.isclose(np.abs(y), 1))
    solution = dirichlet_solve(op["Kp"] - omega**2 * op["Mp"], fixed, exact[fixed])
    lower = y < -0.05
    waves = np.column_stack(
        [np.exp(1j * omega / 1500 * y[lower]), np.exp(-1j * omega / 1500 * y[lower])]
    )
    incident, reflected = np.linalg.lstsq(waves, solution[lower], rcond=None)[0]
    r_measured = reflected / incident
    upper = y > 0.05
    transmitted = np.vdot(np.exp(1j * omega / 2100 * y[upper]), solution[upper]) / np.count_nonzero(
        upper
    )
    t_measured = transmitted / incident
    velocity_jump = (1 - r_measured) / (1000 * 1500) - t_measured / (650 * 2100)
    energy_sum = abs(r_measured) ** 2 + 1000 * 1500 / (650 * 2100) * abs(t_measured) ** 2
    error = relative_error(solution, exact)
    problem.destroy()
    assert error < 2e-5, error
    assert abs(r_measured - reflection) < 1e-5
    assert abs(t_measured - transmission) < 1e-5
    assert abs(velocity_jump) * 1000 * 1500 < 1e-5
    assert abs(energy_sum - 1) < 1e-5
    return {
        "status": "pass",
        "reference": "two-fluid acoustic impedance and density-weighted flux",
        "reflection_reference": reflection,
        "reflection_measured_real": float(r_measured.real),
        "reflection_error": float(abs(r_measured - reflection)),
        "pressure_relative_error": error,
        "transmission_error": float(abs(t_measured - transmission)),
        "normal_velocity_jump_relative_to_incident": float(abs(velocity_jump) * 1000 * 1500),
        "measured_energy_sum": float(energy_sum),
        "reference_energy_sum": er + et,
    }


def temporal_check():
    """Advance a production cavity eigenvector; hold spatial error fixed."""
    from wavesimfem.acoustic_time import CoupledNewmark, CoupledRK4

    problem = build(benchmark_config([fluid()], 6, [6], degree=1))
    op = operators(problem)
    result = {}
    for name, integrator in (("explicit", CoupledRK4), ("implicit", CoupledNewmark)):
        mass = (
            sp.diags(np.asarray(op["Mp"].sum(axis=1)).ravel()) if name == "explicit" else op["Mp"]
        )
        eigenvalues, modes = la.eigh(op["Kp"].toarray(), mass.toarray(), subset_by_index=[1, 1])
        omega = np.sqrt(eigenvalues[0])
        mode = modes[:, 0] / np.max(abs(modes[:, 0]))
        end = 0.83 * 2 * np.pi / omega
        runs = []
        for count in (40, 80, 160):
            dt = end / count
            kwargs = {"initial": {"p": mode}}
            if name == "implicit":
                kwargs["dt"] = dt
            solver = integrator(problem.operators, **kwargs)
            try:
                for step in range(count):
                    solver.step(step * dt, dt)
                value = np.concatenate([solver.p.array_r, solver.pdot.array_r / omega])
                exact = np.concatenate([mode * np.cos(omega * end), -mode * np.sin(omega * end)])
                error = relative_error(value, exact)
                integral_error = relative_error(
                    solver.pressure_integral.array_r, mode * np.sin(omega * end) / omega
                )
            finally:
                solver.destroy()
            runs.append(
                {
                    "steps": count,
                    "relative_state_error": error,
                    "pressure_time_integral_error": integral_error,
                }
            )
        order = np.log(
            runs[-2]["relative_state_error"] / runs[-1]["relative_state_error"]
        ) / np.log(2)
        assert order > (3.9 if name == "explicit" else 1.9), (name, order)
        result[name] = {"observed_order": float(order), "runs": runs}
    problem.destroy()
    return {
        "status": "pass",
        "reference": "exact semidiscrete cavity mode; mass matches each integrator",
        **result,
    }


def boundary_check():
    """Independent pressure-release resonance and outgoing-wave reflection."""
    cfg = benchmark_config([fluid()], 8, [8])
    cfg.update(acoustic_boundaries={"top": "pressure_release"}, source_boundary="bottom")
    problem = build(cfg)
    op = operators(problem)
    eigenvalue = la.eigvalsh(op["Kp"].toarray(), op["Mp"].toarray(), subset_by_index=[0, 0])[0]
    frequency = np.sqrt(eigenvalue) / (2 * np.pi)
    frequency_error = abs(frequency / 375 - 1)
    assert not np.any(np.isclose(problem.p_coords[:, 1], 1)), (
        "Pressure-release DOFs were not eliminated"
    )
    assert frequency_error < 2e-6
    problem.destroy()

    cfg = benchmark_config([fluid()], 3, [24])
    cfg.update(acoustic_boundaries={"top": "radiation"}, source_boundary="bottom")
    problem = build(cfg)
    op = operators(problem)
    omega = 2 * np.pi * 400
    y = problem.p_coords[:, 1]
    exact = np.exp(1j * omega / 1500 * y)
    fixed = np.flatnonzero(np.isclose(y, 0))
    operator = op["Kp"] - omega**2 * op["Mp"] - 1j * omega * csr(problem.operators.Dp)
    solution = dirichlet_solve(operator, fixed, exact[fixed])
    waves = np.column_stack([exact, exact.conjugate()])
    outgoing, incoming = np.linalg.lstsq(waves, solution, rcond=None)[0]
    reflection = abs(incoming / outgoing)
    pressure_error = relative_error(solution, exact)
    problem.destroy()
    assert reflection < 1e-5, reflection
    assert pressure_error < 1e-5, pressure_error
    return {
        "status": "pass",
        "pressure_release": {
            "reference_frequency_hz": 375.0,
            "frequency_hz": float(frequency),
            "relative_error": float(frequency_error),
        },
        "normal_radiation": {
            "frequency_hz": 400.0,
            "pressure_relative_error": pressure_error,
            "reflected_pressure_amplitude": float(reflection),
        },
        "limitation": "First-order radiation is exact in the continuum only at normal incidence; this test uses no sponge.",
    }


def incident_fields(coords, components, omega, angle):
    """One-Pa downward wave from y>1 onto solid y<1."""
    ref = fluid_solid_coefficients(1000, 1500, 2000, 2500, 1200, angle)
    x, y = coords[:, 0], coords[:, 1] - 1
    kx = omega / 1500 * np.sin(angle)
    qp = np.lib.scimath.sqrt((omega / 2500) ** 2 - kx**2)
    qs = np.lib.scimath.sqrt((omega / 1200) ** 2 - kx**2)
    qf = omega / 1500 * np.cos(angle)
    phase = np.exp(1j * kx * x)
    if components is None:
        return phase * (np.exp(-1j * qf * y) + ref["reflection"] * np.exp(1j * qf * y))
    vp = ref["velocity_p"] * np.exp(-1j * qp * y)
    vs = ref["velocity_s"] * np.exp(-1j * qs * y)
    vx = kx * 2500 / omega * vp + qs * 1200 / omega * vs
    vy = -qp * 2500 / omega * vp + kx * 1200 / omega * vs
    return 1j / omega * phase * np.where(components == 0, vx, vy)


def boundary_indices(coords, y_bottom, y_top):
    return np.flatnonzero(
        np.isclose(coords[:, 0], 0)
        | np.isclose(coords[:, 0], 1)
        | np.isclose(coords[:, 1], y_bottom)
        | np.isclose(coords[:, 1], y_top)
    )


def legacy_scattering(cfg, omega, angle, shear_reg=1e-8, acoustic_solution=None):
    """Existing vector-displacement approximation on the identical mesh."""
    from dolfinx import fem
    from dolfinx.fem.petsc import assemble_matrix

    from wavesimfem.config import parse_and_resolve_config
    from wavesimfem.solvers import build_problem

    cfg = copy.deepcopy(cfg)
    for region in cfg["layers"]:
        if region.get("material_model") == "fluid":
            region.update(fluid_formulation="regularized_elastic", fluid_shear_reg=shear_reg)
    cfg.update(
        source_kind="traction",
        source_amplitude=None,
        sensor_observable="displacement",
        sensor_region=None,
        source_region=None,
        source_boundary="auto",
        source_y=None,
    )
    problem = build_problem(parse_and_resolve_config(custom_cfg=cfg))
    mats = []
    for form in (
        problem.stiffness(problem.trial_fn, problem.test_fn),
        problem.mass(problem.trial_fn, problem.test_fn),
    ):
        matrix = assemble_matrix(fem.form(form))
        matrix.assemble()
        mats.append(csr(matrix))
        matrix.destroy()
    coords = np.repeat(problem.V.tabulate_dof_coordinates(), 2, axis=0)
    components = np.tile([0, 1], len(coords) // 2)
    solid_values = incident_fields(coords, components, omega, angle)
    p = incident_fields(coords, None, omega, angle)
    y = coords[:, 1] - 1
    qf = omega / 1500 * np.cos(angle)
    kx = omega / 1500 * np.sin(angle)
    ref = fluid_solid_coefficients(1000, 1500, 2000, 2500, 1200, angle)
    dp_y = np.exp(1j * kx * coords[:, 0]) * (
        -1j * qf * np.exp(-1j * qf * y) + 1j * qf * ref["reflection"] * np.exp(1j * qf * y)
    )
    grad_p = np.where(components == 0, 1j * kx * p, dp_y)
    fluid_values = grad_p / (1000 * omega**2)
    reference = np.where(y <= 0, solid_values, fluid_values)
    fixed = boundary_indices(coords, 0, 2)
    values = dirichlet_solve(mats[0] - omega**2 * mats[1], fixed, reference[fixed])
    inside_solid = (y < -0.02) & (coords[:, 0] > 0.05) & (coords[:, 0] < 0.95)
    result = {
        "solid_displacement_relative_error": relative_error(
            values[inside_solid], reference[inside_solid]
        ),
        "fluid_shear_reg": shear_reg,
    }
    if acoustic_solution is not None:
        xy, comp, solution = acoustic_solution
        # Compare the same Cartesian displacement components at the same solid
        # interior nodes, excluding the forced boundaries and interface.
        lookup = {
            (round(x, 12), round(y, 12), int(c)): value
            for (x, y, _), c, value in zip(xy, comp, solution, strict=True)
        }
        acoustic = np.asarray(
            [
                lookup[(round(x, 12), round(y, 12), int(c))]
                for (x, y, _), c in zip(coords[inside_solid], components[inside_solid], strict=True)
            ]
        )
        result["acoustic_solid_displacement_relative_error_same_nodes"] = relative_error(
            acoustic, reference[inside_solid]
        )
        result["legacy_acoustic_relative_difference_same_nodes"] = relative_error(
            values[inside_solid], acoustic
        )
    return result


def fitted_scattering(problem, solution, omega, angle):
    """Extract plane-wave amplitudes from computed interior nodal fields.

    Interface balances and fluxes below are evaluated from these fitted wave
    amplitudes. They are not pointwise finite-element derivative estimates.
    """
    ns = len(problem.u_coords)
    x, y = problem.p_coords[:, 0], problem.p_coords[:, 1] - 1
    kx = omega / 1500 * np.sin(angle)
    qf = omega / 1500 * np.cos(angle)
    inside = (x > 0.05) & (x < 0.95) & (y > 0.05) & (y < 0.95)
    fluid_waves = np.column_stack(
        [np.exp(1j * kx * x - 1j * qf * y), np.exp(1j * kx * x + 1j * qf * y)]
    )
    incident, reflected = np.linalg.lstsq(fluid_waves[inside], solution[ns:][inside], rcond=None)[0]
    x, y = problem.u_coords[:, 0], problem.u_coords[:, 1] - 1
    comp = problem.u_components
    sx_p, sx_s = 2500 / 1500 * np.sin(angle), 1200 / 1500 * np.sin(angle)
    cz_p, cz_s = np.lib.scimath.sqrt(1 - sx_p**2), np.lib.scimath.sqrt(1 - sx_s**2)
    p_wave = np.where(comp == 0, sx_p, -cz_p) * np.exp(1j * kx * x - 1j * omega / 2500 * cz_p * y)
    s_wave = np.where(comp == 0, cz_s, sx_s) * np.exp(1j * kx * x - 1j * omega / 1200 * cz_s * y)
    inside = (x > 0.05) & (x < 0.95) & (y < -0.05) & (y > -0.95)
    vp, vs = (
        np.linalg.lstsq(
            1j / omega * np.column_stack([p_wave, s_wave])[inside],
            solution[:ns][inside],
            rcond=None,
        )[0]
        / incident
    )
    reflected /= incident
    mu, lam = 2000 * 1200**2, 2000 * (2500**2 - 2 * 1200**2)
    admittance = np.cos(angle) / (1000 * 1500)
    normal_jump = -cz_p * vp + sx_s * vs + admittance * (1 - reflected)
    normal_traction = (
        -(lam * sx_p**2 + (lam + 2 * mu) * cz_p**2) / 2500 * vp + 2 * mu * cz_s * sx_s / 1200 * vs
    )
    shear = 2 * mu * sx_p * cz_p / 2500 * vp + mu * (cz_s**2 - sx_s**2) / 1200 * vs
    slip = np.sin(angle) / (1000 * 1500) * (1 + reflected) - sx_p * vp - cz_s * vs
    energy_sum = (
        abs(reflected) ** 2
        + (2000 * 2500 * cz_p.real * abs(vp) ** 2 + 2000 * 1200 * cz_s.real * abs(vs) ** 2)
        / admittance
    )
    reference = fluid_solid_coefficients(1000, 1500, 2000, 2500, 1200, angle)
    return {
        "reflection_error": float(abs(reflected - reference["reflection"])),
        "p_velocity_amplitude_error_relative_to_incident": float(
            abs(vp - reference["velocity_p"]) / admittance
        ),
        "s_velocity_amplitude_error_relative_to_incident": float(
            abs(vs - reference["velocity_s"]) / admittance
        ),
        "normal_velocity_jump_relative_to_incident": float(abs(normal_jump) / admittance),
        "normal_traction_residual_pa_per_pa_incident": float(abs(normal_traction + 1 + reflected)),
        "tangential_traction_pa_per_pa_incident": float(abs(shear)),
        "tangential_slip_m_per_s_per_pa_incident": float(abs(slip)),
        "energy_sum": float(energy_sum),
    }


def scattering_check(angle=0.0, legacy=True):
    cfg = benchmark_config([solid(), fluid()], 16, [24, 24])
    problem = build(cfg)
    stiffness, mass = coupled_matrices(problem)
    omega = 2 * np.pi * 400
    u_exact = incident_fields(problem.u_coords, problem.u_components, omega, angle)
    p_exact = incident_fields(problem.p_coords, None, omega, angle)
    exact = np.concatenate([u_exact, p_exact])
    iu = boundary_indices(problem.u_coords, 0, 2)
    ip = boundary_indices(problem.p_coords, 0, 2) + len(u_exact)
    fixed = np.concatenate([iu, ip])
    solution = dirichlet_solve(stiffness - omega**2 * mass, fixed, exact[fixed])
    eu = relative_error(solution[: len(u_exact)], u_exact)
    ep = relative_error(solution[len(u_exact) :], p_exact)
    assert eu < 1e-4, eu
    assert ep < 1e-4, ep
    ref = fluid_solid_coefficients(1000, 1500, 2000, 2500, 1200, angle)
    fitted = fitted_scattering(problem, solution, omega, angle)
    for quantity in (
        "reflection_error",
        "p_velocity_amplitude_error_relative_to_incident",
        "s_velocity_amplitude_error_relative_to_incident",
        "normal_velocity_jump_relative_to_incident",
        "normal_traction_residual_pa_per_pa_incident",
        "tangential_traction_pa_per_pa_incident",
    ):
        assert fitted[quantity] < 1e-4, (quantity, fitted[quantity])
    assert abs(fitted["energy_sum"] - 1) < 1e-4
    if angle:
        assert fitted["tangential_slip_m_per_s_per_pa_incident"] > 1e-8
    result = {
        "status": "pass",
        "reference": "independent fluid/P/S plane-wave boundary matching",
        "angle_degrees": float(np.rad2deg(angle)),
        "solid_displacement_relative_error": eu,
        "pressure_relative_error": ep,
        "reference_reflection_real": float(ref["reflection"].real),
        "reference_energy_sum": ref["reflected_energy"] + ref["p_energy"] + ref["s_energy"],
        "reference_tangential_slip_m_per_s": float(abs(ref["tangential_slip"])),
    }
    result["fitted_fem_wave_amplitudes"] = fitted
    if legacy:
        result["legacy_sweep"] = [
            legacy_scattering(
                cfg,
                omega,
                angle,
                shear_reg,
                (problem.u_coords, problem.u_components, solution[: len(u_exact)]),
            )
            for shear_reg in (1e-4, 1e-6, 1e-8)
        ]
        result["legacy"] = result["legacy_sweep"][-1]
        if angle == 0:
            assert result["legacy"]["solid_displacement_relative_error"] < 1e-4
            assert result["legacy"]["legacy_acoustic_relative_difference_same_nodes"] < 1e-4
    problem.destroy()
    return result


def modal_speed(problem, reference, length, fixed_bottom=False):
    """Standing guided modes: roller sides, rigid outer fluid surfaces."""
    stiffness, mass = coupled_matrices(problem)
    uxy = problem.u_coords
    fixed = (np.isclose(uxy[:, 0], 0) | np.isclose(uxy[:, 0], length)) & (problem.u_components == 0)
    if fixed_bottom:
        fixed |= np.isclose(uxy[:, 1], np.min(uxy[:, 1]))
    free = np.setdiff1d(np.arange(stiffness.shape[0]), np.flatnonzero(fixed))
    k = np.pi / length
    # ARPACK assumes positive-definite M for a generalized solve, which this
    # raw u/p mass is not. Apply the shift-invert operator explicitly instead.
    ks, ms = stiffness[free][:, free], mass[free][:, free]
    target = (reference * k) ** 2
    factor = sla.splu((ks - 0.98 * target * ms).tocsc())
    transform = sla.LinearOperator(ks.shape, matvec=lambda v: factor.solve(ms @ v), dtype=float)
    eig, vectors = sla.eigs(transform, k=6, which="LM", tol=1e-10)
    eigenvalues = 0.98 * target + 1 / eig
    ix = np.argmin(abs(eigenvalues - target))
    frequency_squared = eigenvalues[ix]
    assert abs(frequency_squared.imag) < 1e-7 * abs(frequency_squared.real)
    speed = np.sqrt(frequency_squared.real) / k
    mode = np.zeros(stiffness.shape[0], dtype=complex)
    mode[free] = vectors[:, ix]
    return float(speed), mode


def scholte_check():
    reference = scholte_speed(1000, 1500, 2000, 2500, 1200)
    results = []
    for nx, ny in ((8, 32), (12, 48)):
        problem = build(benchmark_config([solid(thickness=4), fluid(thickness=4)], nx, [ny, ny]))
        measured, mode = modal_speed(problem, reference, 1, fixed_bottom=True)
        error = abs(measured / reference - 1)
        p = mode[len(problem.u_coords) :]
        y = problem.p_coords[:, 1] - 4
        expected = np.cos(np.pi * problem.p_coords[:, 0]) * np.exp(
            -np.pi * np.sqrt(1 - (reference / 1500) ** 2) * y
        )
        amplitude = np.vdot(expected, p) / np.vdot(expected, expected)
        shape_error = relative_error(p, amplitude * expected)
        results.append(
            {
                "nx": nx,
                "rows_per_region": ny,
                "phase_speed_m_per_s": measured,
                "relative_error": error,
                "pressure_decay_shape_error": shape_error,
            }
        )
        problem.destroy()
    assert results[-1]["relative_error"] < 2e-4
    assert results[-1]["relative_error"] < results[0]["relative_error"]
    assert results[-1]["pressure_decay_shape_error"] < 0.003
    return {
        "status": "pass",
        "reference": "Scholte secular equation and exponential fluid decay",
        "reference_speed_m_per_s": reference,
        "runs": results,
    }


def plate_check():
    results = []
    for length in (1.0, 0.65):
        k = np.pi / length
        reference = fluid_loaded_plate_speed(k, 0.2, 1000, 1500, 2000, 2500, 1200)
        problem = build(
            benchmark_config(
                [fluid("bottom", 3), solid(thickness=0.2), fluid("top", 3)],
                12,
                [42, 6, 42],
                length=length,
            )
        )
        measured, _ = modal_speed(problem, reference, length)
        error = abs(measured / reference - 1)
        assert error < 0.001, error
        results.append(
            {
                "wavenumber_rad_per_m": k,
                "frequency_hz": measured * k / (2 * np.pi),
                "reference_speed_m_per_s": reference,
                "phase_speed_m_per_s": measured,
                "relative_error": error,
            }
        )
        problem.destroy()
    return {
        "status": "pass",
        "reference": "exact elastic state propagator with evanescent fluid halfspaces",
        "branch": "nonleaky flexural A0",
        "runs": results,
    }


CASES = {
    "cavity": cavity_check,
    "temporal": temporal_check,
    "boundaries": boundary_check,
    "fluid-fluid": fluid_fluid_check,
    "normal": scattering_check,
    "oblique": lambda: scattering_check(np.deg2rad(20)),
    "scholte": scholte_check,
    "plate": plate_check,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    parser.add_argument("--output", type=Path, help="Write the measured results to JSON")
    args = parser.parse_args()
    result = {}
    for name, run in CASES.items():
        if args.case in ("all", name):
            result[name] = run()
            print(json.dumps({name: result[name]}, indent=2), flush=True)
    if args.output:
        from wavesimfem.storage import atomic_output

        with atomic_output(args.output) as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
