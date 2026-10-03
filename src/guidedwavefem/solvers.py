"""Finite element guided-wave solvers: implicit Newmark-β and explicit lumped-mass RK4.

Pipeline
--------
1. :func:`build_problem` — structure normalization, conforming mesh, region
   tags, constitutive laws, sponge damping, load and sensors.
2. :func:`solve_implicit` or :func:`solve_explicit` — time integration.
   The explicit solver runs either the single-process array engine
   (CPU/CUDA, :mod:`guidedwavefem.solvers_gpu`) or the distributed PETSc engine.
3. :func:`run_simulation` — orchestration, provenance, and output files.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import os
import platform
import time as pytime
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, io
from dolfinx.mesh import (
    CellType,
    create_rectangle,
    locate_entities,
    locate_entities_boundary,
    meshtags,
)
from mpi4py import MPI
from petsc4py import PETSc

from guidedwavefem.analysis import plot_quick_overview
from guidedwavefem.config import dump_config, parse_and_resolve_config
from guidedwavefem.device import CPU_DEVICE, DeviceInfo, select_device
from guidedwavefem.materials import (
    MeshLayout,
    compute_mesh_layout,
    implicit_time_step,
    preflight_estimate,
    structure_from_config,
)
from guidedwavefem.numerics import PointSensors, operator_bound, rk4_step, time_grid
from guidedwavefem.reporting import format_banner_box, log, print_structure_table, warn
from guidedwavefem.solvers_gpu import (
    build_sensor_interpolation_matrix,
    integrate_explicit_rk4,
    make_backend,
)
from guidedwavefem.sources import describe_source, ricker, source_signal
from guidedwavefem.units import fmt_length, fmt_time

__all__ = [
    "build_problem",
    "mpi_fail_fast",
    "ricker",
    "run_simulation",
    "solve_explicit",
    "solve_implicit",
]

PACKAGE_DIR = Path(__file__).resolve().parent
#: Warn when the damping bound exceeds the frequency bound by this factor
#: (the explicit step is then set by damping rather than by wave propagation).
DAMPING_LIMITED_RATIO = 2.0
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size

#: Minimum number of time samples required by the FFT post-processing.
MIN_POSTPROCESS_SAMPLES = 32
#: Global DOF count above which the implicit solver switches to CG + GAMG.
ITERATIVE_DOF_THRESHOLD = 500_000


def mpi_fail_fast(function: Callable[..., Any]) -> Callable[..., Any]:
    """Abort the MPI job on a rank-local exception so peers never hang at collectives."""

    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except Exception:
            if nprocs > 1:
                import traceback

                traceback.print_exc()
                comm.Abort(1)
            raise

    return wrapped


def _progress(step: int, n_steps: int, t_start: float) -> None:
    elapsed = pytime.perf_counter() - t_start
    eta = elapsed / step * (n_steps - step)
    log(
        f"  [{100.0 * step / n_steps:5.1f}%]  step {step}/{n_steps}  "
        f"elapsed {elapsed:.1f}s  ETA {eta:.1f}s"
    )


# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------
@dataclass
class WaveProblem:
    """Everything needed to integrate one configured guided-wave model."""

    cfg: dict[str, Any]
    summary: dict[str, Any]
    layout: MeshLayout
    domain: Any
    V: Any
    test_fn: Any
    trial_fn: Any
    load_form: Any
    traction_amplitude: Any
    mass: Callable[[Any, Any], Any]
    stiffness: Callable[[Any, Any], Any]
    damping: Callable[[Any, Any], Any]
    damping_visco: Callable[[Any, Any], Any | None]
    any_viscoelastic: bool
    probes: PointSensors
    sensor_x: np.ndarray
    sensor_y: float
    load_y: float
    ndofs_global: int
    n_owned_dofs: int

    def source(self, t: float) -> float:
        """Traction time function ``traction_max · s(t)`` for the configured waveform."""
        cfg = self.cfg
        waveform = source_signal(t, cfg["source_type"], cfg["f_center"], cfg.get("n_cycles"))
        return float(cfg["traction_max"]) * float(waveform)


@dataclass
class SolveResult:
    """Owned (rank-local) sensor histories and integration metadata."""

    times: np.ndarray
    sensor_data_x: np.ndarray
    sensor_data_y: np.ndarray
    dt: float
    n_steps: int
    setup_seconds: float
    loop_seconds: float
    runtime: dict[str, Any] = field(default_factory=dict)


def _create_layered_mesh(summary: dict[str, Any], layout: MeshLayout) -> tuple[Any, Any]:
    """Structured quadrilateral mesh with element rows aligned to every interface."""
    regions = summary["all_regions"]
    y_min, y_max = summary["y_min"], summary["y_max"]
    domain = create_rectangle(
        comm,
        [np.array([0.0, y_min]), np.array([layout.plate_length, y_max])],
        [layout.nx, layout.ny],
        cell_type=CellType.quadrilateral,
    )
    tdim = domain.topology.dim

    # Piecewise-linear stretching maps uniform element rows onto region interfaces.
    thicknesses = [r["thickness"] for r in regions]
    frac_grid = np.concatenate([[0.0], np.cumsum(layout.ny_per_region) / layout.ny])
    interfaces = np.concatenate([[y_min], y_min + np.cumsum(thicknesses)])
    y_frac = np.clip((domain.geometry.x[:, 1] - y_min) / summary["H_domain"], 0.0, 1.0)
    domain.geometry.x[:, 1] = np.interp(y_frac, frac_grid, interfaces)

    tol = min(thicknesses) * 1e-4
    n_local = domain.topology.index_map(tdim).size_local
    indices, values = [], []
    for tag, reg in enumerate(regions, start=1):
        cells = locate_entities(
            domain,
            tdim,
            lambda x, yb=reg["y_bot"], yt=reg["y_top"]: (x[1] >= yb - tol) & (x[1] <= yt + tol),
        )
        cells = cells[cells < n_local]
        indices.append(cells)
        values.append(np.full(len(cells), tag, dtype=np.int32))
    indices_arr = np.concatenate(indices).astype(np.int32)
    order = np.argsort(indices_arr)
    cell_tags = meshtags(domain, tdim, indices_arr[order], np.concatenate(values)[order])
    return domain, cell_tags


def build_problem(cfg: dict[str, Any], summary: dict[str, Any] | None = None) -> WaveProblem:
    """Assemble mesh, function space, variational forms, load, and sensors."""
    summary = summary or structure_from_config(cfg)
    layout = compute_mesh_layout(cfg, summary)
    regions = summary["all_regions"]
    has_top_hs = summary["has_top_halfspace"]
    H_plate, y_max = summary["H_plate"], summary["y_max"]

    domain, cell_tags = _create_layered_mesh(summary, layout)
    dim = domain.topology.dim
    fdim = dim - 1
    dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)

    V = fem.functionspace(domain, ("Q", int(cfg["degree"]), (dim,)))
    u_ = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)
    index_map = V.dofmap.index_map
    bs = V.dofmap.index_map_bs

    # --- Excitation: Gaussian strip traction on the loaded surface/interface ----
    x = ufl.SpatialCoordinate(domain)
    traction_amplitude = fem.Constant(domain, default_scalar_type(0.0))
    sigma_x = cfg["excitation_width"] / 3.0
    profile = ufl.exp(-((x[0] - cfg["source_x"]) ** 2) / (2.0 * sigma_x**2))
    fx, fy = cfg["source_fx"], cfg["source_fy"]
    norm = float(np.hypot(fx, fy))
    traction = traction_amplitude * profile * ufl.as_vector([fx / norm, fy / norm])
    tol_geom = min(r["thickness"] for r in regions) * 1e-4
    if has_top_hs:
        domain.topology.create_connectivity(fdim, dim)
        domain.topology.create_connectivity(dim, fdim)
        facets = locate_entities(domain, fdim, lambda xc: np.isclose(xc[1], H_plate, atol=tol_geom))
        tags = meshtags(domain, fdim, facets, np.full(len(facets), 1, dtype=np.int32))
        load_form = ufl.dot(traction, u_("+")) * ufl.Measure(
            "dS", domain=domain, subdomain_data=tags
        )(1)
        load_y = H_plate
    else:
        facets = locate_entities_boundary(
            domain, fdim, lambda xc: np.isclose(xc[1], y_max, atol=tol_geom)
        )
        tags = meshtags(domain, fdim, facets, np.full(len(facets), 1, dtype=np.int32))
        load_form = ufl.dot(traction, u_) * ufl.Measure("ds", domain=domain, subdomain_data=tags)(1)
        load_y = y_max

    # --- Constitutive laws per region -------------------------------------------
    def epsilon(w: Any) -> Any:
        return ufl.sym(ufl.grad(w))

    def const(value: float) -> Any:
        return fem.Constant(domain, default_scalar_type(value))

    def voigt(c: list[float]) -> Any:
        c11, c12, c16, c22, c26, c66 = (const(v) for v in c)
        return ufl.as_tensor([[c11, c12, c16], [c12, c22, c26], [c16, c26, c66]])

    def sigma_voigt(w: Any, C: Any) -> Any:
        e = epsilon(w)
        s = C * ufl.as_vector([e[0, 0], e[1, 1], 2.0 * e[0, 1]])
        return ufl.as_tensor([[s[0], s[2]], [s[2], s[1]]])

    rho_c: list[Any] = []
    sigma: list[Callable[[Any], Any]] = []
    sigma_visco: list[Callable[[Any], Any] | None] = []
    for reg in regions:
        rho_c.append(const(reg["rho"]))
        if reg["material_model"] in ("isotropic", "fluid"):
            lmbda, mu = const(reg["lmbda"]), const(reg["mu"])
            sigma.append(
                lambda w, lm=lmbda, m=mu: (
                    lm * ufl.tr(epsilon(w)) * ufl.Identity(dim) + 2.0 * m * epsilon(w)
                )
            )
            sigma_visco.append(None)
        else:
            C = voigt(reg["C"])
            sigma.append(lambda w, Cm=C: sigma_voigt(w, Cm))
            if reg["material_model"] == "viscoelastic":
                Cv = voigt(reg["Cv"])
                sigma_visco.append(lambda w, Cm=Cv: sigma_voigt(w, Cm))
            else:
                sigma_visco.append(None)

    # --- 5th-order sponge (ALID) indicator --------------------------------------
    L_abs = cfg["absorb_length"]
    x_right = layout.plate_length - L_abs
    indicator = ufl.conditional(ufl.gt(x[0], x_right), ((x[0] - x_right) / L_abs) ** 5, 0.0)
    indicator += ufl.conditional(ufl.lt(x[0], L_abs), ((L_abs - x[0]) / L_abs) ** 5, 0.0)
    if summary["has_bottom_halfspace"]:
        depth = summary["bottom_halfspace"]["absorb_depth"]
        y_end = summary["y_min"] + depth
        indicator += ufl.conditional(ufl.lt(x[1], y_end), ((y_end - x[1]) / depth) ** 5, 0.0)
    if has_top_hs:
        depth = summary["top_halfspace"]["absorb_depth"]
        y_start = y_max - depth
        indicator += ufl.conditional(ufl.gt(x[1], y_start), ((x[1] - y_start) / depth) ** 5, 0.0)
    alpha = const(cfg["absorb_alpha"])
    beta = const(cfg["absorb_beta"])
    n_regions = len(regions)

    def mass(w1: Any, w2: Any) -> Any:
        return sum(rho_c[i] * ufl.dot(w1, w2) * dx(i + 1) for i in range(n_regions))

    def stiffness(w1: Any, w2: Any) -> Any:
        return sum(ufl.inner(sigma[i](w1), epsilon(w2)) * dx(i + 1) for i in range(n_regions))

    def damping(w1: Any, w2: Any) -> Any:
        return sum(
            (
                alpha * indicator * rho_c[i] * ufl.dot(w1, w2)
                + beta * indicator * ufl.inner(sigma[i](w1), epsilon(w2))
            )
            * dx(i + 1)
            for i in range(n_regions)
        )

    def damping_visco(w_vel: Any, w2: Any) -> Any | None:
        terms = [
            ufl.inner(sigma_visco[i](w_vel), epsilon(w2)) * dx(i + 1)
            for i in range(n_regions)
            if sigma_visco[i] is not None
        ]
        return sum(terms) if terms else None

    # --- Sensors ---------------------------------------------------------------
    n_sensors = int(cfg["n_sensors"])
    sensor_x = cfg["source_x"] + cfg["sensor_offset"] + np.arange(n_sensors) * cfg["sensor_pitch"]
    if cfg.get("sensor_y") is not None:
        sensor_y = float(cfg["sensor_y"])
    else:
        sensor_y = H_plate if has_top_hs else y_max
    probes = PointSensors(
        domain, np.column_stack([sensor_x, np.full(n_sensors, sensor_y), np.zeros(n_sensors)])
    )

    return WaveProblem(
        cfg=cfg,
        summary=summary,
        layout=layout,
        domain=domain,
        V=V,
        test_fn=u_,
        trial_fn=du,
        load_form=load_form,
        traction_amplitude=traction_amplitude,
        mass=mass,
        stiffness=stiffness,
        damping=damping,
        damping_visco=damping_visco,
        any_viscoelastic=any(s is not None for s in sigma_visco),
        probes=probes,
        sensor_x=sensor_x,
        sensor_y=sensor_y,
        load_y=load_y,
        ndofs_global=index_map.size_global * bs,
        n_owned_dofs=index_map.size_local * bs,
    )


def _check_postprocess_samples(cfg: dict[str, Any], n_steps: int) -> None:
    if cfg["run_postprocess"] and n_steps + 1 < MIN_POSTPROCESS_SAMPLES:
        raise ValueError(
            f"Postprocessing requires at least {MIN_POSTPROCESS_SAMPLES} time samples "
            f"(got {n_steps + 1}); reduce dt or disable postprocessing"
        )


def _vtx_writer(stack: contextlib.ExitStack, problem: WaveProblem, function: Any) -> Any | None:
    if not problem.cfg["write_vtx"]:
        return None
    path = os.path.join(problem.cfg["output_dir"], "guided_wave.bp")
    writer = io.VTXWriter(comm, path, [function], engine="BP4")
    stack.callback(writer.close)
    return writer


# ---------------------------------------------------------------------------
# Implicit Newmark-β
# ---------------------------------------------------------------------------
def _configure_ksp(solver: Any, ndofs_global: int) -> str:
    """Choose a PETSc linear solver for the constant effective stiffness matrix."""
    pc = solver.getPC()
    if ndofs_global >= ITERATIVE_DOF_THRESHOLD:
        solver.setType(PETSc.KSP.Type.CG)
        pc.setType(PETSc.PC.Type.GAMG)
        solver.setTolerances(rtol=1e-8, atol=1e-14, max_it=2000)
        return f"CG + GAMG (iterative, {ndofs_global} DOFs)"
    solver.setType(PETSc.KSP.Type.PREONLY)
    pc.setType(PETSc.PC.Type.LU)
    if nprocs == 1:
        return f"LU (serial direct, {ndofs_global} DOFs)"
    try:
        pc.setFactorSolverType("mumps")
        return f"MUMPS LU (parallel direct, {ndofs_global} DOFs)"
    except (PETSc.Error, RuntimeError):
        return _fallback_bjacobi(solver, ndofs_global)


def _fallback_bjacobi(solver: Any, ndofs_global: int) -> str:
    solver.getPC().setType(PETSc.PC.Type.BJACOBI)
    solver.setType(PETSc.KSP.Type.GMRES)
    solver.setTolerances(rtol=1e-10, atol=1e-14, max_it=2000)
    return f"GMRES + BJacobi (parallel, {ndofs_global} DOFs; MUMPS unavailable)"


def solve_implicit(problem: WaveProblem) -> SolveResult:
    """Unconditionally stable Newmark-β with a once-factorized effective matrix."""
    cfg = problem.cfg
    V, u_, du = problem.V, problem.test_fn, problem.trial_fn
    beta_val, gamma_val = cfg["beta_val"], cfg["gamma_val"]
    dt_req = cfg["dt"] or implicit_time_step(
        problem.layout.h_min, problem.summary["c_L_max"], cfg["CFL"]
    )
    dt_val, n_steps = time_grid(problem.layout.duration, dt_req)
    _check_postprocess_samples(cfg, n_steps)
    if rank == 0:
        log(f"  Time step Δt   = {fmt_time(dt_val)}")
        log(f"  Total time T   = {fmt_time(problem.layout.duration)}")
        log(f"  Number of steps: {n_steps}")

    u = fem.Function(V, name="Displacement")
    u_old = fem.Function(V, name="Displacement_old")
    v_old = fem.Function(V, name="Velocity_old")
    a_old = fem.Function(V, name="Acceleration_old")
    beta = fem.Constant(problem.domain, default_scalar_type(beta_val))
    gamma = fem.Constant(problem.domain, default_scalar_type(gamma_val))
    dt = fem.Constant(problem.domain, default_scalar_type(dt_val))

    a_newmark = 1.0 / beta / dt**2 * (u - u_old - dt * v_old) + a_old * (1.0 - 1.0 / (2.0 * beta))
    v_newmark = v_old + dt * ((1.0 - gamma) * a_old + gamma * a_newmark)
    residual = (
        problem.mass(a_newmark, u_)
        + problem.damping(v_newmark, u_)
        + problem.stiffness(u, u_)
        - problem.load_form
    )
    visco = problem.damping_visco(v_newmark, u_)
    if visco is not None:
        residual += visco
    residual_du = ufl.replace(residual, {u: du})

    c1 = 1.0 / (beta_val * dt_val**2)
    c2 = 1.0 / (beta_val * dt_val)
    c3 = 1.0 - 1.0 / (2.0 * beta_val)
    c4 = dt_val * (1.0 - gamma_val)
    c5 = dt_val * gamma_val

    n_owned = len(problem.probes.indices)
    sensor_x = np.zeros((n_steps + 1, n_owned))
    sensor_y = np.zeros((n_steps + 1, n_owned))
    times = np.arange(n_steps + 1) * dt_val
    save_every = max(n_steps // int(cfg["target_snapshots"]), 1)
    report_every = max(n_steps // 20, 1)
    timers = dict.fromkeys(("assemble", "solve", "update", "io"), 0.0)

    with contextlib.ExitStack() as stack:
        t_setup = pytime.perf_counter()
        a_compiled = fem.form(ufl.lhs(residual_du))
        L_compiled = fem.form(ufl.rhs(residual_du))
        A = fem.petsc.assemble_matrix(a_compiled, bcs=[])
        stack.callback(A.destroy)
        A.assemble()
        b = fem.petsc.create_vector(V)
        stack.callback(b.destroy)
        solver = PETSc.KSP().create(comm)
        stack.callback(solver.destroy)
        solver.setOperators(A)
        solver_desc = _configure_ksp(solver, problem.ndofs_global)
        solver.setErrorIfNotConverged(True)
        try:
            solver.setUp()
        except PETSc.Error:
            if nprocs == 1 or problem.ndofs_global >= ITERATIVE_DOF_THRESHOLD:
                raise
            solver_desc = _fallback_bjacobi(solver, problem.ndofs_global)
            solver.setUp()
        setup_seconds = pytime.perf_counter() - t_setup
        if rank == 0:
            log(f"  Solver: {solver_desc}")
            log(f"  Matrix assembled & factorised in {setup_seconds:.2f} s")
            log(f"\n{'═' * 58}\n  Starting IMPLICIT time-stepping  ({n_steps} steps)\n{'═' * 58}")

        vtx = _vtx_writer(stack, problem, u)
        u_vec = u.x.petsc_vec
        t_loop = pytime.perf_counter()
        for i in range(n_steps):
            t_now = times[i + 1]
            problem.traction_amplitude.value = problem.source(t_now)

            t0 = pytime.perf_counter()
            with b.localForm() as b_local:
                b_local.set(0.0)
            fem.petsc.assemble_vector(b, L_compiled)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            t1 = pytime.perf_counter()
            solver.solve(b, u_vec)
            u.x.scatter_forward()
            t2 = pytime.perf_counter()

            a_new = c1 * (u.x.array - u_old.x.array) - c2 * v_old.x.array + c3 * a_old.x.array
            v_old.x.array[:] = v_old.x.array + c4 * a_old.x.array + c5 * a_new
            a_old.x.array[:] = a_new
            u_old.x.array[:] = u.x.array
            t3 = pytime.perf_counter()

            if n_owned:
                values = problem.probes.sample(u)
                sensor_x[i + 1] = values[:, 0]
                sensor_y[i + 1] = values[:, 1]
            if vtx is not None and i % save_every == 0:
                vtx.write(t_now)
            t4 = pytime.perf_counter()
            timers["assemble"] += t1 - t0
            timers["solve"] += t2 - t1
            timers["update"] += t3 - t2
            timers["io"] += t4 - t3
            if rank == 0 and (i + 1) % report_every == 0:
                _progress(i + 1, n_steps, t_loop)
        loop_seconds = pytime.perf_counter() - t_loop

    if rank == 0:
        log("  Loop breakdown: " + ", ".join(f"{k} {v:.2f} s" for k, v in timers.items()))
    return SolveResult(
        times=times,
        sensor_data_x=sensor_x,
        sensor_data_y=sensor_y,
        dt=dt_val,
        n_steps=n_steps,
        setup_seconds=setup_seconds,
        loop_seconds=loop_seconds,
        runtime={
            "integrator": "Newmark",
            "engine": "petsc",
            "device": "cpu",
            "precision": "float64",
            "linear_solver": solver_desc,
        },
    )


# ---------------------------------------------------------------------------
# Explicit lumped-mass RK4
# ---------------------------------------------------------------------------
def solve_explicit(problem: WaveProblem, device: DeviceInfo | None = None) -> SolveResult:
    """Explicit RK4 with row-sum lumped mass and a conservative Gershgorin step.

    ``device`` selects the single-process array engine; it must be ``None`` for
    distributed runs, which use PETSc operators on every rank.
    """
    cfg = problem.cfg
    V, u_, du = problem.V, problem.test_fn, problem.trial_fn
    n_owned_dofs = problem.n_owned_dofs

    with contextlib.ExitStack() as stack:
        t_setup = pytime.perf_counter()
        K_mat = fem.petsc.assemble_matrix(fem.form(problem.stiffness(du, u_)), bcs=[])
        stack.callback(K_mat.destroy)
        K_mat.assemble()

        M_mat = fem.petsc.assemble_matrix(fem.form(problem.mass(du, u_)), bcs=[])
        M_mat.assemble()
        ones = M_mat.createVecRight()
        ones.set(1.0)
        M_lumped = M_mat.createVecLeft()
        M_mat.mult(ones, M_lumped)
        mass_lumped = M_lumped.array_r[:n_owned_dofs].copy()
        for obj in (ones, M_mat, M_lumped):
            obj.destroy()
        if comm.allreduce(bool(np.any(~np.isfinite(mass_lumped) | (mass_lumped <= 0))), op=MPI.LOR):
            raise RuntimeError("Mass lumping produced non-positive entries")
        inv_mass = 1.0 / mass_lumped

        stiffness_bound = operator_bound(K_mat, mass_lumped)
        omega_max = float(np.sqrt(stiffness_bound))
        C_mat = fem.petsc.assemble_matrix(fem.form(problem.damping(du, u_)), bcs=[])
        stack.callback(C_mat.destroy)
        C_mat.assemble()
        if problem.any_viscoelastic:
            Kv_mat = fem.petsc.assemble_matrix(fem.form(problem.damping_visco(du, u_)), bcs=[])
            Kv_mat.assemble()
            C_mat.axpy(1.0, Kv_mat, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
            Kv_mat.destroy()
        damping_bound = operator_bound(C_mat, mass_lumped)

        dt_crit = 1.0 / max(omega_max, damping_bound)
        requested_dt = float(cfg["dt"] or cfg["CFL"] * dt_crit)
        if requested_dt > dt_crit * (1 + 1e-12):
            raise ValueError(
                f"Requested dt={requested_dt:g} exceeds conservative RK4 limit {dt_crit:g}"
            )
        dt_val, n_steps = time_grid(problem.layout.duration, requested_dt)
        _check_postprocess_samples(cfg, n_steps)

        problem.traction_amplitude.value = 1.0
        f_spatial = fem.petsc.create_vector(V)
        stack.callback(f_spatial.destroy)
        fem.petsc.assemble_vector(f_spatial, fem.form(problem.load_form))
        f_spatial.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        load_shape = f_spatial.array_r[:n_owned_dofs].copy()
        setup_seconds = pytime.perf_counter() - t_setup

        if rank == 0:
            log(f"  Frequency upper bound = {omega_max:.2e} rad/s")
            log(f"  Damping spectral bound = {damping_bound:.2e} 1/s")
            log(f"  Conservative step limit = {fmt_time(dt_crit)}")
            log(f"  Δt (CFL={cfg['CFL']}) = {fmt_time(dt_val)}")
            log(f"  Number of steps: {n_steps}")
            log(f"  Total time T   = {fmt_time(problem.layout.duration)}")
            if damping_bound > DAMPING_LIMITED_RATIO * omega_max:
                warn(
                    f"Damping limits the explicit step {damping_bound / omega_max:.0f}x below "
                    "the wave-speed limit. Reduce absorbing_boundaries.beta (stiffness-"
                    "proportional sponge damping) or viscous Cv, or use the implicit solver."
                )

        save_every = max(n_steps // int(cfg["target_snapshots"]), 1)
        report_every = max(n_steps // 20, 1)
        u = fem.Function(V, name="Displacement")
        vtx = _vtx_writer(stack, problem, u)
        runtime = {
            "integrator": "RK4",
            "stiffness_bound": stiffness_bound,
            "damping_bound": damping_bound,
            "dt_limit": dt_crit,
        }

        if device is not None:
            if nprocs != 1:
                raise RuntimeError("The array engine is single-process only")
            sensor_mat = build_sensor_interpolation_matrix(
                V, problem.probes.points, problem.probes.cells, n_owned_dofs
            )
            backend = make_backend(device, cfg["precision"], log if rank == 0 else None)

            def snapshot(_step: int, t_snap: float, u_host: np.ndarray) -> None:
                u.x.array[:n_owned_dofs] = u_host
                u.x.scatter_forward()
                vtx.write(t_snap)

            if rank == 0:
                log(
                    f"\n{'═' * 58}\n  Starting EXPLICIT time-stepping on {device.summary()} "
                    f"({n_steps} steps)\n{'═' * 58}"
                )
            result = integrate_explicit_rk4(
                K_mat.getValuesCSR(),
                C_mat.getValuesCSR(),
                inv_mass,
                load_shape,
                sensor_mat,
                dt=dt_val,
                n_steps=n_steps,
                source=problem.source,
                backend=backend,
                save_every=save_every,
                snapshot_callback=snapshot if vtx is not None else None,
                report_interval=report_every,
                log_func=log if rank == 0 else None,
            )
            runtime.update(engine=backend.name, device=device.device, precision=backend.dtype.name)
            return SolveResult(
                times=result.times,
                sensor_data_x=result.sensor_data_x,
                sensor_data_y=result.sensor_data_y,
                dt=dt_val,
                n_steps=n_steps,
                setup_seconds=setup_seconds,
                loop_seconds=result.loop_seconds,
                runtime=runtime,
            )

        # Distributed PETSc engine (identical RK4 scheme via numerics.rk4_step).
        v_half = fem.Function(V)
        u_arr, vh_arr = u.x.array, v_half.x.array
        u_vec, vh_vec = u.x.petsc_vec, v_half.x.petsc_vec
        r_vec = fem.petsc.create_vector(V)
        stack.callback(r_vec.destroy)
        c_vec = fem.petsc.create_vector(V)
        stack.callback(c_vec.destroy)

        def acceleration(disp: np.ndarray, vel: np.ndarray, t: float) -> np.ndarray:
            u_arr[:n_owned_dofs] = disp
            vh_arr[:n_owned_dofs] = vel
            K_mat.mult(u_vec, r_vec)
            C_mat.mult(vh_vec, c_vec)
            r_vec.axpy(1.0, c_vec)
            return (problem.source(t) * load_shape - r_vec.array_r[:n_owned_dofs]) * inv_mass

        n_owned = len(problem.probes.indices)
        sensor_x = np.zeros((n_steps + 1, n_owned))
        sensor_y = np.zeros((n_steps + 1, n_owned))
        times = np.arange(n_steps + 1) * dt_val
        disp = np.zeros(n_owned_dofs)
        vel = np.zeros(n_owned_dofs)
        if rank == 0:
            log(
                f"\n{'═' * 58}\n  Starting EXPLICIT time-stepping (PETSc, {nprocs} ranks) "
                f"({n_steps} steps)\n{'═' * 58}"
            )
        t_loop = pytime.perf_counter()
        for i in range(n_steps):
            disp, vel = rk4_step(disp, vel, times[i], dt_val, acceleration)
            u_arr[:n_owned_dofs] = disp
            u.x.scatter_forward()
            if n_owned:
                values = problem.probes.sample(u)
                sensor_x[i + 1] = values[:, 0]
                sensor_y[i + 1] = values[:, 1]
            if vtx is not None and i % save_every == 0:
                vtx.write(times[i + 1])
            if (i + 1) % report_every == 0:
                if comm.allreduce(not np.isfinite(disp).all(), op=MPI.LOR):
                    raise FloatingPointError(f"Non-finite state at step {i + 1}/{n_steps}")
                if rank == 0:
                    _progress(i + 1, n_steps, t_loop)
        loop_seconds = pytime.perf_counter() - t_loop

    runtime.update(engine="petsc", device="cpu", precision="float64")
    return SolveResult(
        times=times,
        sensor_data_x=sensor_x,
        sensor_data_y=sensor_y,
        dt=dt_val,
        n_steps=n_steps,
        setup_seconds=setup_seconds,
        loop_seconds=loop_seconds,
        runtime=runtime,
    )


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def _material_fields(region: dict[str, Any]) -> dict[str, Any]:
    model = region["material_model"]
    if model == "isotropic":
        return {"E": region["E"], "nu": region["nu"]}
    if model == "fluid":
        return {"c_fluid": region["c_fluid"], "K": region["K"]}
    fields = {"C_ij": region["C_mat"]}
    if model == "viscoelastic":
        fields["Cv_ij"] = region["Cv_mat"]
    return fields


def _archive_fields(problem: WaveProblem) -> dict[str, Any]:
    """Metadata stored next to the sensor histories in ``sensor_data.npz``."""
    cfg, s = problem.cfg, problem.summary
    plate, regions = s["plate_layers"], s["all_regions"]
    mode = s["structure_mode"]
    top = plate[-1] if plate else s["bottom_halfspace"]
    n_plate = len(plate)
    fields: dict[str, Any] = {
        "sensor_x": problem.sensor_x,
        "sensor_y": problem.sensor_y,
        "sensor_coords": np.column_stack(
            [problem.sensor_x, np.full_like(problem.sensor_x, problem.sensor_y)]
        ),
        "load_y": problem.load_y,
        "f_center": cfg["f_center"],
        "f_centers": np.array([cfg["f_center"]]),
        "plate_length": problem.layout.plate_length,
        "plate_thickness": s["H_plate"],
        "rho": s["rho_plate_top"],
        "c_L": s["c_L_plate_top"],
        "c_T": s["c_T_plate_top"],
        "has_fluid": s["has_fluid"],
        "is_fluid_top": bool(top.get("is_fluid", False)),
        "structure_mode": mode,
        "material_model": (
            f"{mode} ({n_plate} plate layers)"
            if n_plate > 1 or mode != "1_layer"
            else plate[0]["material_model"]
        ),
        "n_layers": n_plate,
        "layers_names": np.array([r["name"] for r in plate], dtype="U"),
        "layers_thicknesses": np.array([r["thickness"] for r in plate]),
        "layers_rhos": np.array([r["rho"] for r in plate]),
        "layers_models": np.array([r["material_model"] for r in plate], dtype="U"),
        "all_regions_names": np.array([r["name"] for r in regions], dtype="U"),
        "all_regions_thicknesses": np.array([r["thickness"] for r in regions]),
        "all_regions_roles": np.array([r["role"] for r in regions], dtype="U"),
        "all_regions_models": np.array([r["material_model"] for r in regions], dtype="U"),
        "has_bottom_halfspace": s["has_bottom_halfspace"],
        "has_top_halfspace": s["has_top_halfspace"],
        "y_min": s["y_min"],
        "y_max": s["y_max"],
        "source_x": cfg["source_x"],
        "sensor_offset": cfg["sensor_offset"],
        "sensor_pitch": cfg["sensor_pitch"],
        "source_type": cfg["source_type"],
        "n_cycles": int(cfg.get("n_cycles") or 0),  # 0 = not applicable (Ricker)
        "disp_f_min": cfg["disp_f_min"],
        "disp_f_max": cfg["disp_f_max"],
        "disp_k_min": cfg["disp_k_min"],
        "disp_k_max": cfg["disp_k_max"],
        "config_source": str(cfg.get("_config_source", "")),
    }
    if s.get("c_fluid_min") is not None:
        fields["c_fluid"] = s["c_fluid_min"]
    if n_plate == 1:
        fields.update(_material_fields(plate[0]))
    elif mode == "halfspace":
        hs = s["bottom_halfspace"]
        if hs.get("c_R") is not None:
            fields["c_R"] = hs["c_R"]
        fields.update(_material_fields(hs))
        fields.update(
            total_depth=s["H_domain"],
            halfspace_depth=s["H_domain"] - hs["absorb_depth"],
            absorb_depth_bottom=hs["absorb_depth"],
        )
    return fields


def _runtime_record(problem: WaveProblem, result: SolveResult) -> dict[str, Any]:
    source_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(PACKAGE_DIR.glob("*.py"))
    }
    from guidedwavefem import __version__

    return {
        "guidedwavefem": __version__,
        "dt": result.dt,
        "steps": result.n_steps,
        "duration": result.n_steps * result.dt,
        "nx": problem.layout.nx,
        "ny_per_region": problem.layout.ny_per_region,
        "ndofs": problem.ndofs_global,
        "mpi_ranks": nprocs,
        "mesh_frequency": problem.layout.mesh_frequency,
        **result.runtime,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "dolfinx": dolfinx.__version__,
        "petsc": list(PETSc.Sys.getVersion()),
        "source_sha256": source_hashes,
    }


def _resolve_device(cfg: dict[str, Any], stype: str) -> DeviceInfo | None:
    """Validate the device request against solver type and MPI size."""
    requested, precision = cfg["device"], cfg["precision"]
    if stype == "implicit":
        if requested not in ("auto", "cpu") and rank == 0:
            warn(f"device '{requested}' is ignored by the implicit solver (PETSc CPU).")
        if precision != "double" and rank == 0:
            warn("precision 'single' is ignored by the implicit solver (float64).")
        return None
    if nprocs > 1:
        if requested not in ("auto", "cpu"):
            raise ValueError(
                f"device '{requested}' requires a single-process run; distributed explicit "
                "runs use the PETSc CPU engine (set device: auto or cpu)."
            )
        if precision != "double" and rank == 0:
            warn("precision 'single' is ignored by the distributed PETSc engine (float64).")
        return None
    return select_device(requested, log_func=log) if requested != "cpu" else CPU_DEVICE


@mpi_fail_fast
def run_simulation(
    args: argparse.Namespace | None = None,
    cfg: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Run one configured simulation and write its outputs.

    Parameters
    ----------
    args
        Parsed CLI namespace (overrides) and/or the ``config`` path.
    cfg
        Configuration dictionary (sectioned or flat). Takes precedence over
        ``args.config``.

    Returns
    -------
    (path to ``sensor_data.npz``, output directory)
    """
    if cfg is None:
        if args is None or not getattr(args, "config", None):
            raise ValueError("A configuration file (-c/--config) or dictionary must be provided.")
        cfg = parse_and_resolve_config(config_path=args.config, args=args)
    else:
        cfg = parse_and_resolve_config(args=args, custom_cfg=cfg)

    stype = cfg["solver_type"]
    out_dir = cfg["output_dir"] or f"results_guided_waves_{stype}"
    cfg["output_dir"] = out_dir
    device = _resolve_device(cfg, stype)
    wall_t0 = pytime.perf_counter()

    summary = structure_from_config(cfg)
    if rank == 0:
        log(
            format_banner_box(
                [
                    f"GuidedWaveFEM solver: {stype.upper()}",
                    f"MPI ranks: {nprocs}",
                    f"Mode: {summary['structure_mode']}",
                    f"Config: {cfg.get('_config_source', 'Defaults')}",
                ]
            )
        )
        print_structure_table(summary, rank=rank, f0=cfg["f_center"])
        if summary["has_fluid"]:
            log(
                "  Fluid model: regularized elasticity (small shear modulus) with continuous "
                "tangential displacement; not a mixed acoustic formulation."
            )
    preflight_estimate(cfg, rank=rank)

    problem = build_problem(cfg, summary)
    layout = problem.layout
    if rank == 0:
        log(f"  Excitation: {describe_source(cfg['source_type'], cfg.get('n_cycles'))}")
        log(f"  Shortest wavelength λ_min = {fmt_length(layout.lambda_min)}")
        log(f"  Target element size h     = {fmt_length(layout.h_target)}")
        log(f"  Mesh: {layout.nx} × {layout.ny} elements, ny_per_region = {layout.ny_per_region}")
        log(f"  DOFs: {problem.ndofs_global} global")
        log(
            f"  Sensors: {len(problem.sensor_x)} at y = {fmt_length(problem.sensor_y)}, "
            f"x = {fmt_length(problem.sensor_x[0])} … {fmt_length(problem.sensor_x[-1])}; "
            f"load at y = {fmt_length(problem.load_y)}, x = {fmt_length(cfg['source_x'])}"
        )
        os.makedirs(out_dir, exist_ok=True)
    comm.Barrier()

    result = solve_implicit(problem) if stype == "implicit" else solve_explicit(problem, device)

    if comm.allreduce(
        not (np.isfinite(result.sensor_data_x).all() and np.isfinite(result.sensor_data_y).all()),
        op=MPI.LOR,
    ):
        raise FloatingPointError("Non-finite sensor data: simulation failed")
    data_x = problem.probes.gather(result.sensor_data_x)
    data_y = problem.probes.gather(result.sensor_data_y)

    saved_npz = os.path.join(out_dir, "sensor_data.npz")
    if rank == 0:
        wall = pytime.perf_counter() - wall_t0
        loop = max(result.loop_seconds, 1e-12)
        log(f"\n{'═' * 58}\n  Performance Summary ({stype.upper()})\n{'═' * 58}")
        log(f"  Total wall time:     {wall:.2f} s")
        log(f"  ├─ Setup / factorise: {result.setup_seconds:.2f} s")
        log(f"  ├─ Time-step loop:    {result.loop_seconds:.2f} s")
        log(f"  └─ Per-step average:  {loop / result.n_steps * 1e3:.3f} ms/step")
        log(f"  Throughput:           {result.n_steps / loop:.0f} steps/s\n{'═' * 58}")

        cfg["sensor_y"] = float(problem.sensor_y)
        cfg["load_y"] = float(problem.load_y)
        cfg["runtime"] = _runtime_record(problem, result)
        saved_cfg = os.path.join(out_dir, "simulation_config.yaml")
        dump_config(cfg, saved_cfg)
        log(f"  Saved simulation configuration: {saved_cfg}")

        archive = {
            "times": result.times,
            "sensor_data_x": data_x,
            "sensor_data_y": data_y,
            **_archive_fields(problem),
        }
        (np.savez_compressed if cfg["compress_output"] else np.savez)(saved_npz, **archive)
        log(f"  Saved simulation sensor data: {saved_npz}")

        if cfg["quick_plot"] and cfg["run_postprocess"]:
            plot_quick_overview(
                result.times,
                problem.sensor_x,
                data_y,
                cfg["f_center"],
                cfg["source_x"],
                os.path.join(out_dir, "quick_results.png"),
                source_type=cfg["source_type"],
                n_cycles=cfg.get("n_cycles"),
            )
    comm.Barrier()
    return saved_npz, out_dir
