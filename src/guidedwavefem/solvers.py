"""Finite element wave propagation solvers: implicit Newmark-β and explicit lumped RK4."""

from __future__ import annotations

import argparse
import functools
import hashlib
import os
import platform
import time as pytime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

import dolfinx.fem.petsc
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

import matplotlib

matplotlib.use("Agg")  # safe for headless & MPI execution
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from guidedwavefem.config import dump_config, parse_and_resolve_config
from guidedwavefem.materials import (
    _format_banner_box,
    _normalize_structure,
    _print_structure_table,
    log,
    preflight_estimate,
)
from guidedwavefem.numerics import PointSensors, operator_bound, rk4_step, time_grid
from guidedwavefem.device import select_device
from guidedwavefem.solvers_gpu import (
    build_sensor_interpolation_matrix,
    solve_explicit_rk4_device,
)

PACKAGE_DIR = Path(__file__).resolve().parent
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size


def mpi_fail_fast(function):
    """Prevent peers hanging at collectives after a rank-local exception."""
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception:
            if nprocs > 1:
                import traceback
                traceback.print_exc()
                comm.Abort(1)
            raise
    return wrapped


def ricker(t: float, f0: float, t0: float) -> float:
    """Compute Ricker wavelet value at time t."""
    tau = t - t0
    a_param = (np.pi * f0 * tau) ** 2
    return float((1.0 - 2.0 * a_param) * np.exp(-a_param))


@mpi_fail_fast
def run_simulation(
    args: argparse.Namespace | None = None,
    cfg: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Execute guided wave finite element simulation using implicit or explicit time integration."""
    if cfg is None:
        if args is None or not getattr(args, "config", None):
            raise ValueError(
                "A configuration file (-c/--config) or config dictionary must be provided to run_simulation."
            )
        cfg = parse_and_resolve_config(config_path=args.config, args=args)
    else:
        cfg = parse_and_resolve_config(args=args, custom_cfg=cfg)

    stype = (getattr(args, "solver", None) if args is not None else None) or cfg.get(
        "solver_type", "explicit"
    )
    stype = str(stype).lower().strip()
    if stype not in ("implicit", "explicit"):
        raise ValueError(f"Unknown solver '{stype}'. Must be 'implicit' or 'explicit'.")

    # Set output directory based on solver
    out_dir = (
        (getattr(args, "output", None) if args is not None else None)
        or cfg.get("output_dir")
        or f"results_guided_waves_{stype}"
    )
    cfg["output_dir"] = out_dir

    sensing_length = float(cfg["sensing_length"])
    absorb_length = float(cfg["absorb_length"])
    f_center = float(cfg["f_center"])
    source_x = float(cfg["source_x"])
    source_fx = float(cfg.get("source_fx", 0.0))
    source_fy = float(cfg.get("source_fy", 1.0))
    traction_max = float(cfg.get("traction_max", 1.0e6))
    excitation_width = float(cfg.get("excitation_width", 2.0e-3))
    elems_per_wavelength = int(cfg.get("elems_per_wavelength", 10))
    min_elems_x = int(cfg.get("min_elems_x", 100))
    min_elems_y = int(cfg.get("min_elems_y", 4))
    degree = int(cfg.get("degree", 2))
    CFL = float(cfg.get("CFL", 0.5))
    time_factor = float(cfg.get("time_factor", 1.2))
    beta_val = float(cfg.get("beta_val", 0.25))
    gamma_val = float(cfg.get("gamma_val", 0.5))
    n_sensors = int(cfg.get("n_sensors", 301))
    sensor_offset = float(cfg.get("sensor_offset", 0.05))
    sensor_pitch = float(cfg.get("sensor_pitch", 0.5e-3))
    write_vtx = bool(cfg.get("write_vtx", False))
    target_snapshots = int(cfg.get("target_snapshots", 200))
    f_min_disp_MHz = float(cfg.get("f_min_disp_MHz", 0.0))
    f_max_disp_MHz = float(cfg.get("f_max_disp_MHz", 2.0))
    k_min_disp = float(cfg.get("k_min_disp", 0.0))
    k_max_disp = float(cfg.get("k_max_disp", 3.0))

    # Adaptive Rayleigh damping defaults
    absorb_alpha = cfg.get("absorb_alpha")
    absorb_beta = cfg.get("absorb_beta")
    if absorb_alpha is None:
        absorb_alpha = 1e7
    else:
        absorb_alpha = float(absorb_alpha)
    if absorb_beta is None:
        absorb_beta = 1e-8
    else:
        absorb_beta = float(absorb_beta)

    wall_t0 = pytime.perf_counter()

    # Total plate length = left absorber + sensing region + right absorber
    plate_length = absorb_length + sensing_length + absorb_length

    # Multilayer / Halfspace structure normalization & wave speed analysis
    mode_arg = getattr(args, "structure_mode", None) if args is not None else None
    struct_summary = _normalize_structure(
        mode=mode_arg or cfg.get("structure_mode", "N_layers"),
        single_layer_input=cfg.get("single_layer"),
        layers_input=cfg.get("layers"),
        halfspace_input=cfg.get("halfspace"),
        bottom_hs_input=cfg.get("bottom_halfspace"),
        top_hs_input=cfg.get("top_halfspace"),
    )
    all_regions = struct_summary["all_regions"]
    norm_plate_layers = struct_summary["plate_layers"]
    n_regions = len(all_regions)
    n_plate_layers = len(norm_plate_layers)
    H_plate = struct_summary["H_plate"]
    y_min = struct_summary["y_min"]
    y_max = struct_summary["y_max"]
    H_domain = struct_summary["H_domain"]
    has_bottom_halfspace = struct_summary["has_bottom_halfspace"]
    has_top_halfspace = struct_summary["has_top_halfspace"]
    active_structure_mode = struct_summary["structure_mode"]

    c_T_min = struct_summary["c_T_min"]
    c_L_max = struct_summary["c_L_max"]
    c_wave_min = struct_summary.get("c_wave_min", c_T_min)
    c_L_top = struct_summary["c_L_plate_top"]
    c_T_top = struct_summary["c_T_plate_top"]
    rho_top = struct_summary["rho_plate_top"]
    has_fluid = struct_summary.get("has_fluid", False)
    c_fluid_min = struct_summary.get("c_fluid_min")
    any_viscoelastic = any(r["material_model"] == "viscoelastic" for r in all_regions)

    if rank == 0:
        cfg_source_str = str(cfg.get("_config_source", "Defaults"))
        banner_lines = [
            f"Guided Waves FEM Solver: {stype.upper()}",
            f"MPI ranks: {nprocs}",
            f"Mode: {active_structure_mode}",
            f"Config: {cfg_source_str}",
        ]
        log(_format_banner_box(banner_lines))
        _print_structure_table(struct_summary, rank=rank, f0=f_center)
        if has_fluid:
            log(
                "  Fluid model: regularized elasticity with continuous tangential displacement; "
                "not a mixed inviscid acoustic formulation."
            )

    # Preflight resource and discretization estimation
    preflight_estimate(cfg, rank=rank)

    # Mesh sizing based on shortest wavelength across all active regions
    mesh_frequency = float(cfg.get("max_frequency") or 3 * f_center)
    lambda_min = c_wave_min / mesh_frequency
    h_target = lambda_min / elems_per_wavelength
    nx = max(int(np.ceil(plate_length / h_target)), min_elems_x)

    ny_per_region = [
        max(
            int(np.ceil(reg["thickness"] / h_target)),
            reg.get("min_elems_y") or min_elems_y,
        )
        for reg in all_regions
    ]
    ny = sum(ny_per_region)

    if rank == 0:
        log(f"  Shortest wavelength λ_min = {lambda_min * 1e3:.2f} mm")
        log(f"  Target element size h     = {h_target * 1e3:.3f} mm")
        log(
            f"  Mesh: {nx} × {ny} elements ({nx * ny} total), "
            f"ny_per_region = {ny_per_region}"
        )

    h_actual = min(
        plate_length / nx,
        min(r["thickness"] / ny_per_region[i] for i, r in enumerate(all_regions)),
    )
    speed_for_T = struct_summary.get("c_R") or c_wave_min
    T_total = float(cfg.get("duration") or sensing_length / speed_for_T * time_factor)

    # Sensor positions along top surface (or interface if top halfspace is present)
    sensor_x_arr = source_x + sensor_offset + np.arange(n_sensors) * sensor_pitch

    top_reg = all_regions[-1]
    is_top_fluid = top_reg.get("is_fluid", False)
    top_fluid_label = "fluid" if is_top_fluid else "solid"

    cfg_sensor_y = cfg.get("sensor_y", None)
    if cfg_sensor_y is not None:
        sensor_y_coord = float(cfg_sensor_y)
    elif has_top_halfspace:
        sensor_y_coord = H_plate
    else:
        sensor_y_coord = y_max

    if not np.isfinite(sensor_y_coord) or not y_min <= sensor_y_coord <= y_max:
        raise ValueError("sensor_y must lie inside the domain")

    if rank == 0:
        if active_structure_mode == "halfspace":
            interface_desc = "halfspace free surface (y = 0.00 mm)"
        elif has_top_halfspace:
            interface_desc = f"plate-superstrate interface (y = {H_plate * 1e3:.2f} mm)"
        else:
            interface_desc = f"top surface of '{top_reg['name']}' [{top_fluid_label}] (y = {sensor_y_coord * 1e3:.2f} mm)"
        log(
            f"  Sensor array: {n_sensors} sensors on {interface_desc}, "
            f"x = [{sensor_x_arr[0] * 1e3:.1f}, {sensor_x_arr[-1] * 1e3:.1f}] mm, "
            f"pitch = {sensor_pitch * 1e3:.1f} mm"
        )

    # Create conforming mesh spanning [0, y_min] to [plate_length, y_max]
    domain = create_rectangle(
        comm,
        [np.array([0.0, y_min]), np.array([plate_length, y_max])],
        [nx, ny],
        cell_type=CellType.quadrilateral,
    )
    dim = domain.topology.dim
    tdim = dim

    # In-place coordinate stretching: strictly align horizontal element lines with all region interfaces
    thicknesses = [r["thickness"] for r in all_regions]
    frac_grid = np.concatenate([[0.0], np.cumsum(ny_per_region) / ny])
    geom_interfaces = np.concatenate([[y_min], y_min + np.cumsum(thicknesses)])

    y_frac = np.clip((domain.geometry.x[:, 1] - y_min) / H_domain, 0.0, 1.0)
    domain.geometry.x[:, 1] = np.interp(y_frac, frac_grid, geom_interfaces)

    # Tag subdomain cells for each region
    tol_layer = min(thicknesses) * 1e-4
    num_local_cells = domain.topology.index_map(tdim).size_local

    all_cell_indices = []
    all_cell_values = []
    for r_idx, reg in enumerate(all_regions):
        tag_val = r_idx + 1
        y_bot = reg["y_bot"]
        y_top = reg["y_top"]
        cells = locate_entities(
            domain,
            tdim,
            lambda x_c, yb=y_bot, yt=y_top: (
                (x_c[1] >= yb - tol_layer) & (x_c[1] <= yt + tol_layer)
            ),
        )
        cells_loc = cells[cells < num_local_cells]
        all_cell_indices.append(cells_loc)
        all_cell_values.append(np.full(len(cells_loc), tag_val, dtype=np.int32))

    all_cell_indices = np.concatenate(all_cell_indices).astype(np.int32)
    all_cell_values = np.concatenate(all_cell_values).astype(np.int32)
    sort_idx = np.argsort(all_cell_indices)
    cell_tags = meshtags(
        domain, tdim, all_cell_indices[sort_idx], all_cell_values[sort_idx]
    )
    dx_layers = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)

    # Function space
    V = fem.functionspace(domain, ("Q", degree, (dim,)))
    u_ = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)

    ndofs_local = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    ndofs_global = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    if rank == 0:
        log(f"  DOFs: {ndofs_global} global ({ndofs_local} local per rank)")

    # Excitation facet tagging
    tol_geom = min(thicknesses) * 1e-4
    fdim = dim - 1

    x = ufl.SpatialCoordinate(domain)
    traction_amplitude = fem.Constant(domain, default_scalar_type(0.0))
    excite_profile = ufl.exp(
        -((x[0] - source_x) ** 2) / (2.0 * (excitation_width / 3.0) ** 2)
    )
    norm_dir = np.sqrt(source_fx**2 + source_fy**2)
    if norm_dir == 0:
        raise ValueError("source_fx and source_fy cannot both be zero.")
    exc_dir = ufl.as_vector([source_fx / norm_dir, source_fy / norm_dir])
    traction_vector = traction_amplitude * excite_profile * exc_dir

    if has_top_halfspace:
        domain.topology.create_connectivity(fdim, tdim)
        domain.topology.create_connectivity(tdim, fdim)
        interface_facets = locate_entities(
            domain, fdim, lambda x_c: np.isclose(x_c[1], H_plate, atol=tol_geom)
        )
        interface_facet_tags = meshtags(
            domain,
            fdim,
            interface_facets,
            np.full(len(interface_facets), 1, dtype=np.int32),
        )
        dS = ufl.Measure("dS", domain=domain, subdomain_data=interface_facet_tags)
        F_ext_form_expr = ufl.dot(traction_vector, u_("+")) * dS(1)
    else:
        top_facets = locate_entities_boundary(
            domain, fdim, lambda x_c: np.isclose(x_c[1], y_max, atol=tol_geom)
        )
        top_facet_tags = meshtags(
            domain,
            fdim,
            top_facets,
            np.full(len(top_facets), 1, dtype=np.int32),
        )
        ds = ufl.Measure("ds", domain=domain, subdomain_data=top_facet_tags)
        F_ext_form_expr = ufl.dot(traction_vector, u_) * ds(1)

    if rank == 0:
        if has_top_halfspace:
            load_desc = f"plate-superstrate interface (y = {H_plate * 1e3:.2f} mm)"
        else:
            load_desc = (
                f"top surface of top layer '{top_reg['name']}' [{top_fluid_label}] "
                f"(y = {y_max * 1e3:.2f} mm)"
            )
        log(
            f"  Excitation source: applied on {load_desc} (x = {source_x * 1e3:.1f} mm)"
        )

    def epsilon(w):
        return ufl.sym(ufl.grad(w))

    def _voigt_matrix(c11, c12, c16, c22, c26, c66):
        return ufl.as_tensor(
            [
                [c11, c12, c16],
                [c12, c22, c26],
                [c16, c26, c66],
            ]
        )

    def _sigma_from_C(w, C_mat):
        e = epsilon(w)
        e_voigt = ufl.as_vector([e[0, 0], e[1, 1], 2.0 * e[0, 1]])
        s_voigt = C_mat * e_voigt
        return ufl.as_tensor([[s_voigt[0], s_voigt[2]], [s_voigt[2], s_voigt[1]]])

    rho_c_list = []
    sigma_list = []
    sigma_visco_list = []

    for reg in all_regions:
        rho_c_list.append(fem.Constant(domain, default_scalar_type(reg["rho"])))
        model = reg["material_model"]
        if model in ("isotropic", "fluid"):
            lmbda_c = fem.Constant(domain, default_scalar_type(reg["lmbda"]))
            mu_c = fem.Constant(domain, default_scalar_type(reg["mu"]))

            def make_iso_sigma(lmbda_val, mu_val):
                return lambda w: (
                    lmbda_val * ufl.tr(epsilon(w)) * ufl.Identity(dim)
                    + 2.0 * mu_val * epsilon(w)
                )

            sigma_list.append(make_iso_sigma(lmbda_c, mu_c))
            sigma_visco_list.append(None)
        else:
            C = reg["C"]
            C_c = _voigt_matrix(
                fem.Constant(domain, default_scalar_type(C[0])),
                fem.Constant(domain, default_scalar_type(C[1])),
                fem.Constant(domain, default_scalar_type(C[2])),
                fem.Constant(domain, default_scalar_type(C[3])),
                fem.Constant(domain, default_scalar_type(C[4])),
                fem.Constant(domain, default_scalar_type(C[5])),
            )

            def make_aniso_sigma(C_tensor):
                return lambda w: _sigma_from_C(w, C_tensor)

            sigma_list.append(make_aniso_sigma(C_c))

            if model == "viscoelastic":
                Cv = reg["Cv"]
                Cv_c = _voigt_matrix(
                    fem.Constant(domain, default_scalar_type(Cv[0])),
                    fem.Constant(domain, default_scalar_type(Cv[1])),
                    fem.Constant(domain, default_scalar_type(Cv[2])),
                    fem.Constant(domain, default_scalar_type(Cv[3])),
                    fem.Constant(domain, default_scalar_type(Cv[4])),
                    fem.Constant(domain, default_scalar_type(Cv[5])),
                )

                def make_visco_sigma(Cv_tensor):
                    return lambda w_vel: _sigma_from_C(w_vel, Cv_tensor)

                sigma_visco_list.append(make_visco_sigma(Cv_c))
            else:
                sigma_visco_list.append(None)

    # 2D Absorbing layers (Rayleigh damping sponge)
    x_right_start = plate_length - absorb_length
    absorb_right = ufl.conditional(
        ufl.gt(x[0], x_right_start),
        ((x[0] - x_right_start) / absorb_length) ** 5,
        0.0,
    )
    absorb_left = ufl.conditional(
        ufl.lt(x[0], absorb_length),
        ((absorb_length - x[0]) / absorb_length) ** 5,
        0.0,
    )
    absorb_indicator = absorb_left + absorb_right

    # Vertical sponges for halfspaces
    if has_bottom_halfspace:
        L_sponge_sub = struct_summary["bottom_halfspace"]["absorb_depth"]
        y_sub_sponge_end = y_min + L_sponge_sub
        absorb_sub = ufl.conditional(
            ufl.lt(x[1], y_sub_sponge_end),
            ((y_sub_sponge_end - x[1]) / L_sponge_sub) ** 5,
            0.0,
        )
        absorb_indicator = absorb_indicator + absorb_sub

    if has_top_halfspace:
        L_sponge_sup = struct_summary["top_halfspace"]["absorb_depth"]
        y_sup_sponge_start = y_max - L_sponge_sup
        absorb_sup = ufl.conditional(
            ufl.gt(x[1], y_sup_sponge_start),
            ((x[1] - y_sup_sponge_start) / L_sponge_sup) ** 5,
            0.0,
        )
        absorb_indicator = absorb_indicator + absorb_sup

    alpha_absorb = fem.Constant(domain, default_scalar_type(absorb_alpha))
    beta_absorb = fem.Constant(domain, default_scalar_type(absorb_beta))

    def mass(w1, w2):
        return sum(
            rho_c_list[idx] * ufl.dot(w1, w2) * dx_layers(idx + 1)
            for idx in range(n_regions)
        )

    def stiffness(w1, w2):
        return sum(
            ufl.inner(sigma_list[idx](w1), epsilon(w2)) * dx_layers(idx + 1)
            for idx in range(n_regions)
        )

    def damping(w1, w2):
        return sum(
            (
                alpha_absorb * absorb_indicator * rho_c_list[idx] * ufl.dot(w1, w2)
                + beta_absorb
                * absorb_indicator
                * ufl.inner(sigma_list[idx](w1), epsilon(w2))
            )
            * dx_layers(idx + 1)
            for idx in range(n_regions)
        )

    def damping_visco(w_vel, w2):
        terms = [
            ufl.inner(sigma_visco_list[idx](w_vel), epsilon(w2)) * dx_layers(idx + 1)
            for idx in range(n_regions)
            if sigma_visco_list[idx] is not None
        ]
        return sum(terms) if len(terms) > 0 else None

    t0_ricker = 1.5 / f_center

    probes = PointSensors(
        domain,
        np.column_stack([
            sensor_x_arr,
            np.full(n_sensors, sensor_y_coord),
            np.zeros(n_sensors),
        ]),
    )
    n_owned = len(probes.indices)

    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    comm.Barrier()

    # ==========================================================================
    # BRANCH A: IMPLICIT NEWMARK-β SOLVER
    # ==========================================================================
    if stype == "implicit":
        dt_val, Nsteps = time_grid(
            T_total, float(cfg.get("dt") or CFL * h_actual / c_L_max)
        )
        if cfg["run_postprocess"] and Nsteps < 31:
            raise ValueError(
                "Postprocessing requires at least 32 time samples; reduce dt or disable it"
            )

        if rank == 0:
            log(f"  Time step Δt   = {dt_val * 1e9:.2f} ns")
            log(f"  Total time T   = {T_total * 1e6:.1f} μs")
            log(f"  Number of steps: {Nsteps}")

        u = fem.Function(V, name="Displacement")
        u_old = fem.Function(V, name="Displacement_old")
        v_old = fem.Function(V, name="Velocity_old")
        a_old = fem.Function(V, name="Acceleration_old")
        v_new = fem.Function(V, name="Velocity_new")
        a_new = fem.Function(V, name="Acceleration_new")

        beta = fem.Constant(domain, default_scalar_type(beta_val))
        gamma = fem.Constant(domain, default_scalar_type(gamma_val))
        dt = fem.Constant(domain, default_scalar_type(dt_val))

        a_newmark = 1.0 / beta / dt**2 * (u - u_old - dt * v_old) + a_old * (
            1.0 - 1.0 / (2.0 * beta)
        )
        v_newmark = v_old + dt * ((1.0 - gamma) * a_old + gamma * a_newmark)

        Residual = (
            mass(a_newmark, u_)
            + damping(v_newmark, u_)
            + stiffness(u, u_)
            - F_ext_form_expr
        )
        dv_term = damping_visco(v_newmark, u_)
        if dv_term is not None:
            Residual += dv_term

        Residual_du = ufl.replace(Residual, {u: du})
        a_form = ufl.lhs(Residual_du)
        L_form = ufl.rhs(Residual_du)

        t_setup_start = pytime.perf_counter()
        a_compiled = fem.form(a_form)
        L_compiled = fem.form(L_form)

        A = fem.petsc.assemble_matrix(a_compiled, bcs=[])
        A.assemble()
        b = fem.petsc.create_vector(V)

        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        use_iterative = ndofs_global >= 500_000

        if use_iterative:
            solver.setType(PETSc.KSP.Type.CG)
            pc = solver.getPC()
            pc.setType(PETSc.PC.Type.GAMG)
            solver.setTolerances(rtol=1e-8, atol=1e-14, max_it=2000)
            solver_desc = f"CG + GAMG (iterative, {ndofs_global} DOFs)"
        else:
            solver.setType(PETSc.KSP.Type.PREONLY)
            pc = solver.getPC()
            if nprocs > 1:
                pc.setType(PETSc.PC.Type.LU)
                try:
                    pc.setFactorSolverType("mumps")
                    solver_desc = f"MUMPS LU (parallel direct, {ndofs_global} DOFs)"
                except (PETSc.Error, RuntimeError):
                    pc.setType(PETSc.PC.Type.BJACOBI)
                    solver.setType(PETSc.KSP.Type.GMRES)
                    solver.setTolerances(rtol=1e-10, atol=1e-14, max_it=2000)
                    solver_desc = f"GMRES + BJacobi (parallel, {ndofs_global} DOFs)"
            else:
                pc.setType(PETSc.PC.Type.LU)
                solver_desc = f"LU (serial direct, {ndofs_global} DOFs)"

        solver.setErrorIfNotConverged(True)
        try:
            solver.setUp()
        except PETSc.Error:
            if nprocs == 1 or use_iterative:
                raise
            pc.setType(PETSc.PC.Type.BJACOBI)
            solver.setType(PETSc.KSP.Type.GMRES)
            solver.setTolerances(rtol=1e-10, atol=1e-14, max_it=2000)
            solver.setUp()
            solver_desc = "GMRES + BJacobi (MUMPS unavailable)"
        t_setup_end = pytime.perf_counter()

        if rank == 0:
            log(f"  Solver: {solver_desc}")
            log(
                f"  Matrix assembled & factorised in {t_setup_end - t_setup_start:.2f} s"
            )

        nm_c1 = 1.0 / (beta_val * dt_val**2)
        nm_c2 = 1.0 / (beta_val * dt_val)
        nm_c3 = 1.0 - 1.0 / (2.0 * beta_val)
        nm_c4 = dt_val * (1.0 - gamma_val)
        nm_c5 = dt_val * gamma_val

        sensor_data_x = np.zeros((Nsteps + 1, n_owned))
        sensor_data_y = np.zeros((Nsteps + 1, n_owned))
        times = np.zeros(Nsteps + 1)

        if write_vtx:
            vtx = io.VTXWriter(comm, f"{out_dir}/guided_wave.bp", [u], engine="BP4")
        save_freq = max(Nsteps // target_snapshots, 1)

        time_assemble = 0.0
        time_solve = 0.0
        time_update = 0.0
        time_io = 0.0
        u_petsc = u.x.petsc_vec
        report_interval = max(Nsteps // 20, 1)

        if rank == 0:
            log(f"\n{'═' * 58}")
            log(f"  Starting IMPLICIT time-stepping  ({Nsteps} steps)")
            log(f"{'═' * 58}")

        t_loop_start = pytime.perf_counter()

        for i in range(Nsteps):
            t_current = (i + 1) * dt_val
            times[i + 1] = t_current

            traction_amplitude.value = traction_max * ricker(
                t_current, f_center, t0_ricker
            )

            t0 = pytime.perf_counter()
            with b.localForm() as b_local:
                b_local.set(0.0)
            fem.petsc.assemble_vector(b, L_compiled)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            t1 = pytime.perf_counter()
            time_assemble += t1 - t0

            solver.solve(b, u_petsc)
            u.x.scatter_forward()
            t2 = pytime.perf_counter()
            time_solve += t2 - t1

            a_new.x.array[:] = (
                nm_c1 * (u.x.array - u_old.x.array)
                - nm_c2 * v_old.x.array
                + nm_c3 * a_old.x.array
            )
            v_new.x.array[:] = (
                v_old.x.array + nm_c4 * a_old.x.array + nm_c5 * a_new.x.array
            )

            u_old.x.array[:] = u.x.array
            v_old.x.array[:] = v_new.x.array
            a_old.x.array[:] = a_new.x.array
            t3 = pytime.perf_counter()
            time_update += t3 - t2

            if n_owned > 0:
                values = probes.sample(u)
                sensor_data_x[i + 1] = values[:, 0]
                sensor_data_y[i + 1] = values[:, 1]

            if write_vtx and i % save_freq == 0:
                t4 = pytime.perf_counter()
                vtx.write(t_current)
                time_io += pytime.perf_counter() - t4

            if rank == 0 and (i + 1) % report_interval == 0:
                pct = 100.0 * (i + 1) / Nsteps
                elapsed = pytime.perf_counter() - t_loop_start
                eta = elapsed / (i + 1) * (Nsteps - i - 1)
                log(
                    f"  [{pct:5.1f}%]  step {i + 1}/{Nsteps}  elapsed {elapsed:.1f}s  ETA {eta:.1f}s"
                )

        t_loop_end = pytime.perf_counter()
        if write_vtx:
            vtx.close()

        A.destroy()
        b.destroy()
        solver.destroy()

    # ==========================================================================
    # BRANCH B: EXPLICIT LUMPED-MASS RK4 SOLVER
    # ==========================================================================
    elif stype == "explicit":
        u = fem.Function(V, name="Displacement")
        v = fem.Function(V, name="Velocity")
        v_half = fem.Function(V, name="v_half")

        t_setup_start = pytime.perf_counter()

        K_form = fem.form(stiffness(du, u_))
        K_mat = fem.petsc.assemble_matrix(K_form, bcs=[])
        K_mat.assemble()

        M_form = fem.form(mass(du, u_))
        M_mat = fem.petsc.assemble_matrix(M_form, bcs=[])
        M_mat.assemble()

        ones = M_mat.createVecRight()
        ones.set(1.0)
        M_lumped = M_mat.createVecLeft()
        M_mat.mult(ones, M_lumped)
        n_owned_dofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
        M_lumped_np = M_lumped.array_r[:n_owned_dofs].copy()

        ones.destroy()
        M_mat.destroy()

        if comm.allreduce(
            bool(np.any(~np.isfinite(M_lumped_np) | (M_lumped_np <= 0))), op=MPI.LOR
        ):
            raise RuntimeError("Mass lumping produced non-positive entries!")
        M_inv_np = 1.0 / M_lumped_np

        omega2_max = operator_bound(K_mat, M_lumped_np)
        omega_max = np.sqrt(omega2_max)
        C_form = fem.form(damping(du, u_))
        C_mat = fem.petsc.assemble_matrix(C_form, bcs=[])
        C_mat.assemble()

        if any_viscoelastic:
            Kv_form = fem.form(damping_visco(du, u_))
            Kv_mat = fem.petsc.assemble_matrix(Kv_form, bcs=[])
            Kv_mat.assemble()

            C_mat.axpy(
                1.0, Kv_mat, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN
            )
            Kv_mat.destroy()
        damping_bound = operator_bound(C_mat, M_lumped_np)

        dt_crit = 1.0 / max(omega_max, damping_bound)
        requested_dt = float(cfg.get("dt") or CFL * dt_crit)
        if requested_dt > dt_crit * (1 + 1e-12):
            raise ValueError(
                f"Requested dt={requested_dt:g} exceeds conservative RK4 limit {dt_crit:g}"
            )
        dt_val, Nsteps = time_grid(T_total, requested_dt)
        if cfg["run_postprocess"] and Nsteps < 31:
            raise ValueError(
                "Postprocessing requires at least 32 time samples; reduce dt or disable it"
            )

        if rank == 0:
            log(f"  Frequency upper bound = {omega_max:.2e} rad/s")
            log(f"  Damping spectral bound = {damping_bound:.2e} 1/s")
            log(f"  Conservative step limit = {dt_crit * 1e9:.2f} ns")
            log(f"  Δt (CFL={CFL}) = {dt_val * 1e9:.2f} ns")
            log(f"  Number of steps: {Nsteps}")
            log(f"  Total time T   = {T_total * 1e6:.1f} μs")

        F_ext_form = fem.form(F_ext_form_expr)
        f_vec = fem.petsc.create_vector(V)
        r_vec = fem.petsc.create_vector(V)

        traction_amplitude.value = 1.0
        f_spatial = fem.petsc.create_vector(V)
        fem.petsc.assemble_vector(f_spatial, F_ext_form)
        f_spatial.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        t_setup_end = pytime.perf_counter()

        target_device = str(cfg.get("device", "auto"))
        target_precision = str(cfg.get("precision", "double"))
        save_freq = max(Nsteps // target_snapshots, 1)
        report_interval = max(Nsteps // 20, 1)

        if comm.size == 1:
            dev_info = select_device(target_device, log_func=log if rank == 0 else None)
            sensor_mat = build_sensor_interpolation_matrix(
                V, probes.points, probes.cells, n_owned_dofs
            )
            K_csr = K_mat.getValuesCSR()
            C_csr = C_mat.getValuesCSR()
            f_spatial_np = f_spatial.array_r[:n_owned_dofs].copy()
            def source_func(t: float) -> float:
                return float(traction_max * ricker(t, f_center, t0_ricker))

            if write_vtx:
                vtx = io.VTXWriter(comm, f"{out_dir}/guided_wave.bp", [u], engine="BP4")

                def snapshot_cb(step_idx: int, t_snap: float, u_snapshot: np.ndarray) -> None:
                    u.x.array[:n_owned_dofs] = u_snapshot
                    u.x.scatter_forward()
                    vtx.write(t_snap)
            else:
                vtx = None
                snapshot_cb = None

            if rank == 0:
                log(f"\n{'═' * 58}")
                log(f"  Starting EXPLICIT time-stepping ({dev_info.summary()}) ({Nsteps} steps)")
                log(f"{'═' * 58}")

            t_loop_start = pytime.perf_counter()
            times, sensor_data_x, sensor_data_y = solve_explicit_rk4_device(
                K_csr=K_csr,
                C_csr=C_csr,
                M_inv_np=M_inv_np,
                f_spatial_np=f_spatial_np,
                sensor_mat_csr=sensor_mat,
                dt=dt_val,
                Nsteps=Nsteps,
                source_func=source_func,
                device_info=dev_info,
                precision=target_precision,
                save_freq=save_freq,
                snapshot_callback=snapshot_cb,
                report_interval=report_interval,
                log_func=log if rank == 0 else lambda msg: None,
            )
            t_loop_end = pytime.perf_counter()

            if vtx is not None:
                vtx.close()

            K_mat.destroy()
            C_mat.destroy()
            M_lumped.destroy()
            f_spatial.destroy()
            f_vec.destroy()
            r_vec.destroy()

        else:
            sensor_data_x = np.zeros((Nsteps + 1, n_owned))
            sensor_data_y = np.zeros((Nsteps + 1, n_owned))
            times = np.zeros(Nsteps + 1)

            if write_vtx:
                vtx = io.VTXWriter(comm, f"{out_dir}/guided_wave.bp", [u], engine="BP4")
            save_freq = max(Nsteps // target_snapshots, 1)

            u_arr = u.x.array
            v_arr = v.x.array
            vh_arr = v_half.x.array

            u_petsc = u.x.petsc_vec
            v_half_petsc = v_half.x.petsc_vec

            report_interval = max(Nsteps // 20, 1)

            def acceleration(displacement, velocity, t):
                u_arr[:n_owned_dofs] = displacement
                vh_arr[:n_owned_dofs] = velocity
                K_mat.mult(u_petsc, r_vec)
                C_mat.mult(v_half_petsc, f_vec)
                r_vec.axpy(1.0, f_vec)
                force = traction_max * ricker(t, f_center, t0_ricker)
                return (
                    force * f_spatial.array_r[:n_owned_dofs]
                    - r_vec.array_r[:n_owned_dofs]
                ) * M_inv_np

            time_update = 0.0
            time_io = 0.0

            if rank == 0:
                log(f"\n{'═' * 58}")
                log(f"  Starting EXPLICIT time-stepping  ({Nsteps} steps)")
                log(f"{'═' * 58}")

            t_loop_start = pytime.perf_counter()

            for i in range(Nsteps):
                t_previous = i * dt_val
                t_current = (i + 1) * dt_val
                times[i + 1] = t_current

                t0 = pytime.perf_counter()
                displacement, velocity = rk4_step(
                    u_arr[:n_owned_dofs].copy(),
                    v_arr[:n_owned_dofs].copy(),
                    t_previous,
                    dt_val,
                    acceleration,
                )
                u_arr[:n_owned_dofs] = displacement
                v_arr[:n_owned_dofs] = velocity
                u.x.scatter_forward()
                t1 = pytime.perf_counter()
                time_update += t1 - t0

                if n_owned > 0:
                    values = probes.sample(u)
                    sensor_data_x[i + 1] = values[:, 0]
                    sensor_data_y[i + 1] = values[:, 1]

                if write_vtx and i % save_freq == 0:
                    t5 = pytime.perf_counter()
                    vtx.write(t_current)
                    time_io += pytime.perf_counter() - t5

                if rank == 0 and (i + 1) % report_interval == 0:
                    pct = 100.0 * (i + 1) / Nsteps
                    elapsed = pytime.perf_counter() - t_loop_start
                    eta = elapsed / (i + 1) * (Nsteps - i - 1)
                    log(
                        f"  [{pct:5.1f}%]  step {i + 1}/{Nsteps}  elapsed {elapsed:.1f}s  ETA {eta:.1f}s"
                    )

            t_loop_end = pytime.perf_counter()
            if write_vtx:
                vtx.close()

            K_mat.destroy()
            C_mat.destroy()
            M_lumped.destroy()
            f_spatial.destroy()
            f_vec.destroy()
            r_vec.destroy()

    # ── Gather sensor data across all MPI ranks ────────────────
    if comm.allreduce(
        not (np.isfinite(sensor_data_x).all() and np.isfinite(sensor_data_y).all()),
        op=MPI.LOR,
    ):
        raise RuntimeError("Nonfinite sensor data: simulation failed")
    sensor_data_x_all = probes.gather(sensor_data_x)
    sensor_data_y_all = probes.gather(sensor_data_y)
    cfg.update(absorb_alpha=absorb_alpha, absorb_beta=absorb_beta)

    package_py_files = sorted(PACKAGE_DIR.glob("*.py"))
    source_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in package_py_files
    }

    cfg["runtime"] = {
        "dt": dt_val,
        "steps": Nsteps,
        "duration": Nsteps * dt_val,
        "nx": nx,
        "ny_per_region": ny_per_region,
        "ndofs": ndofs_global,
        "mpi_ranks": nprocs,
        "mesh_frequency": mesh_frequency,
        "integrator": "RK4" if stype == "explicit" else "Newmark",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "dolfinx": dolfinx.__version__,
        "petsc": list(PETSc.Sys.getVersion()),
        "source_sha256": source_hashes,
    }
    if stype == "explicit":
        cfg["runtime"].update(
            stiffness_bound=omega2_max,
            damping_bound=damping_bound,
            dt_limit=dt_crit,
        )

    # ── Performance summary & data save (rank 0 only) ──────────
    saved_npz = os.path.join(out_dir, "sensor_data.npz")
    if rank == 0:
        wall_total = pytime.perf_counter() - wall_t0
        loop_total = t_loop_end - t_loop_start
        setup_total = t_setup_end - t_setup_start

        log(f"\n{'═' * 58}")
        log(f"  Performance Summary ({stype.upper()})")
        log(f"{'═' * 58}")
        log(f"  Total wall time:     {wall_total:.2f} s")
        log(f"  ├─ Setup / factorise: {setup_total:.2f} s")
        log(f"  ├─ Time-step loop:    {loop_total:.2f} s")
        log(f"  └─ Per-step average:  {loop_total / Nsteps * 1e3:.2f} ms/step")
        log(f"  Throughput:           {Nsteps / loop_total:.0f} steps/s")
        log(f"{'═' * 58}")

        top_lyr_or_hs = (
            norm_plate_layers[-1]
            if norm_plate_layers
            else struct_summary["bottom_halfspace"]
        )
        is_fluid_top = bool(top_lyr_or_hs.get("is_fluid", False))

        save_kwargs = {
            "times": times,
            "sensor_x": sensor_x_arr,
            "sensor_y": sensor_y_coord,
            "sensor_coords": np.column_stack(
                [sensor_x_arr, np.full_like(sensor_x_arr, sensor_y_coord)]
            ),
            "load_y": H_plate if has_top_halfspace else y_max,
            "sensor_data_x": sensor_data_x_all,
            "sensor_data_y": sensor_data_y_all,
            "f_center": f_center,
            "f_centers": np.array([f_center]),
            "plate_length": plate_length,
            "plate_thickness": H_plate,
            "rho": rho_top,
            "c_L": c_L_top,
            "c_T": c_T_top,
            "has_fluid": has_fluid,
            "is_fluid_top": is_fluid_top,
            "structure_mode": active_structure_mode,
            "material_model": (
                f"{active_structure_mode} ({n_plate_layers} plate layers)"
                if n_plate_layers > 1 or active_structure_mode != "1_layer"
                else norm_plate_layers[0]["material_model"]
            ),
            "n_layers": n_plate_layers,
            "layers_names": np.array(
                [layer["name"] for layer in norm_plate_layers], dtype="U"
            ),
            "layers_thicknesses": np.array(
                [layer["thickness"] for layer in norm_plate_layers]
            ),
            "layers_rhos": np.array([layer["rho"] for layer in norm_plate_layers]),
            "layers_models": np.array(
                [layer["material_model"] for layer in norm_plate_layers], dtype="U"
            ),
            "all_regions_names": np.array([r["name"] for r in all_regions], dtype="U"),
            "all_regions_thicknesses": np.array([r["thickness"] for r in all_regions]),
            "all_regions_roles": np.array([r["role"] for r in all_regions], dtype="U"),
            "all_regions_models": np.array(
                [r["material_model"] for r in all_regions], dtype="U"
            ),
            "has_bottom_halfspace": has_bottom_halfspace,
            "has_top_halfspace": has_top_halfspace,
            "y_min": y_min,
            "y_max": y_max,
            "source_x": source_x,
            "sensor_offset": sensor_offset,
            "sensor_pitch": sensor_pitch,
            "f_min_disp_MHz": f_min_disp_MHz,
            "f_max_disp_MHz": f_max_disp_MHz,
            "k_min_disp": k_min_disp,
            "k_max_disp": k_max_disp,
        }
        if c_fluid_min is not None:
            save_kwargs["c_fluid"] = c_fluid_min

        if n_plate_layers == 1:
            if norm_plate_layers[0]["material_model"] == "isotropic":
                save_kwargs.update(
                    E=norm_plate_layers[0]["E"], nu=norm_plate_layers[0]["nu"]
                )
            elif norm_plate_layers[0]["material_model"] == "fluid":
                save_kwargs.update(
                    c_fluid=norm_plate_layers[0]["c_fluid"],
                    K=norm_plate_layers[0]["K"],
                )
            else:
                save_kwargs.update(C_ij=norm_plate_layers[0]["C_mat"])
            if norm_plate_layers[0]["material_model"] == "viscoelastic":
                save_kwargs.update(Cv_ij=norm_plate_layers[0]["Cv_mat"])
        elif active_structure_mode == "halfspace":
            hs = struct_summary["bottom_halfspace"]
            if hs.get("c_R") is not None:
                save_kwargs["c_R"] = hs["c_R"]
            if hs["material_model"] == "isotropic":
                save_kwargs.update(E=hs["E"], nu=hs["nu"])
            elif hs["material_model"] == "fluid":
                save_kwargs.update(
                    c_fluid=hs["c_fluid"],
                    K=hs["K"],
                )
            else:
                save_kwargs.update(C_ij=hs["C_mat"])
            if hs["material_model"] == "viscoelastic":
                save_kwargs.update(Cv_ij=hs["Cv_mat"])
            save_kwargs.update(
                total_depth=H_domain,
                halfspace_depth=H_domain - hs["absorb_depth"],
                absorb_depth_bottom=hs["absorb_depth"],
            )

        cfg["sensor_y"] = float(sensor_y_coord)
        cfg["load_y"] = float(H_plate if has_top_halfspace else y_max)
        saved_cfg_path = os.path.join(out_dir, "simulation_config.yaml")
        dump_config(cfg, saved_cfg_path)
        log(f"  Saved simulation configuration: {saved_cfg_path}")

        save_kwargs["config_source"] = str(cfg.get("_config_source", ""))
        np.savez(saved_npz, **save_kwargs)
        log(f"\n  Saved simulation sensor data: {saved_npz}")

    if rank == 0 and cfg.get("quick_plot", True) and cfg.get("run_postprocess", True):
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        ax = axes[0]
        t_exc = np.linspace(0, 4.0 / f_center, 1000)
        ax.plot(
            t_exc * 1e6, [ricker(ti, f_center, t0_ricker) for ti in t_exc], "b-", lw=1.5
        )
        ax.set_xlabel("Time [μs]")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Excitation: Ricker wavelet, f₀ = {f_center / 1e3:.0f} kHz")
        ax.grid(True, alpha=0.3)

        _fs = 1.0 / dt_val
        _f_lo = f_center / 5.0
        _f_hi = min(f_center * 3.0, _fs / 2.0 * 0.9)
        _sos = butter(4, [_f_lo, _f_hi], btype="bandpass", fs=_fs, output="sos")
        sensor_y_detrend = np.zeros_like(sensor_data_y_all)
        for j in range(n_sensors):
            sensor_y_detrend[:, j] = sosfiltfilt(_sos, sensor_data_y_all[:, j])

        ax = axes[1]
        global_peak = np.max(np.abs(sensor_y_detrend))
        pitch_mm = (sensor_x_arr[1] - sensor_x_arr[0]) * 1e3
        n_show = min(n_sensors, 30)
        step = max(n_sensors // n_show, 1)
        trace_spacing = pitch_mm * step
        global_scale = 0.8 * trace_spacing / max(global_peak, 1e-30)

        for j in range(0, n_sensors, step):
            offset = sensor_x_arr[j] * 1e3
            ax.plot(
                times * 1e6,
                sensor_y_detrend[:, j] * global_scale + offset,
                "b-",
                lw=0.5,
            )
        ax.set_xlabel("Time [μs]")
        ax.set_ylabel("Position [mm]")
        ax.set_title("Out-of-plane displacement (u_y) — Waterfall plot")
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        _dt_us = (times[1] - times[0]) * 1e6
        _dx_mm = (sensor_x_arr[1] - sensor_x_arr[0]) * 1e3
        extent = [
            times[0] * 1e6 - _dt_us / 2.0,
            times[-1] * 1e6 + _dt_us / 2.0,
            sensor_x_arr[0] * 1e3 - _dx_mm / 2.0,
            sensor_x_arr[-1] * 1e3 + _dx_mm / 2.0,
        ]
        vmax = np.max(np.abs(sensor_y_detrend)) * 0.5
        if vmax > 0:
            ax.imshow(
                sensor_y_detrend.T,
                aspect="auto",
                extent=extent,
                origin="lower",
                cmap="seismic",
                vmin=-vmax,
                vmax=vmax,
            )
        ax.set_xlim(times[0] * 1e6, times[-1] * 1e6)
        ax.set_ylim(sensor_x_arr[0] * 1e3, sensor_x_arr[-1] * 1e3)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_xlabel("Time [μs]")
        ax.set_ylabel("Position [mm]")
        ax.set_title("B-scan (out-of-plane displacement)")

        plt.tight_layout()
        quick_plot_path = os.path.join(out_dir, "quick_results.png")
        fig.savefig(quick_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"  Quick visualisation saved: {quick_plot_path}")

    comm.Barrier()
    return saved_npz, out_dir
