"""Time integration, step-limit, sampling, and solver-helper tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
from petsc4py import PETSc

from wavesimfem.numerics import (
    RK4_STABILITY_RADIUS,
    PetscRK4,
    operator_bound,
    rk4_amplification,
    rk4_half_disk_radius,
    rk4_step,
    sample_stride,
)

ALUMINUM = {"thickness": 1e-3, "rho": 2700.0, "E": 70e9, "nu": 0.33, "material_model": "isotropic"}


# ---------------------------------------------------------------------------
# RK4 stability constant
# ---------------------------------------------------------------------------
def test_rk4_half_disk_radius_bounds_constant():
    radius = rk4_half_disk_radius()
    assert radius == pytest.approx(2.6156, abs=5e-4)
    assert radius >= RK4_STABILITY_RADIUS


def test_rk4_stable_on_half_disk():
    """|R(z)| ≤ 1 on a dense polar grid of the closed half-disk of radius R."""
    r = np.linspace(0.0, RK4_STABILITY_RADIUS, 200)[:, None]
    theta = np.linspace(np.pi / 2, np.pi, 361)[None, :]
    z = r * np.exp(1j * theta)
    assert np.abs(rk4_amplification(z)).max() <= 1.0 + 1e-12
    # Just outside the imaginary-axis limit 2√2 RK4 is unstable.
    assert abs(rk4_amplification(1j * 2.9)) > 1.0


# ---------------------------------------------------------------------------
# PETSc RK4 vs functional oracle
# ---------------------------------------------------------------------------
def _petsc_aij(matrix: sp.csr_matrix) -> PETSc.Mat:
    matrix = sp.csr_matrix(matrix)
    mat = PETSc.Mat().createAIJ(
        size=matrix.shape,
        csr=(
            matrix.indptr.astype(PETSc.IntType),
            matrix.indices.astype(PETSc.IntType),
            matrix.data,
        ),
        comm=PETSc.COMM_SELF,
    )
    mat.assemble()
    return mat


def _damped_system(n: int = 20, seed: int = 1):
    rng = np.random.default_rng(seed)
    K = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n), format="csr") * 3.0e2
    C = sp.diags(rng.uniform(0.0, 0.8, n), format="csr")
    inv_mass = rng.uniform(0.5, 2.0, n)
    load = np.zeros(n)
    load[n // 2] = 1.0
    return K, C, inv_mass, load


def _source(t: float) -> float:
    return float(np.sin(30.0 * t) * np.exp(-((t - 0.05) ** 2) / 1e-3))


def test_petsc_rk4_matches_functional_oracle():
    K, C, inv_mass, load = _damped_system()
    dt, n_steps = 2e-3, 120
    u, v = np.zeros(len(load)), np.zeros(len(load))
    for i in range(n_steps):
        u, v = rk4_step(
            u, v, i * dt, dt, lambda uu, vv, t: inv_mass * (_source(t) * load - K @ uu - C @ vv)
        )

    K_p, C_p = _petsc_aij(K), _petsc_aij(C)
    integrator = PetscRK4(K_p, C_p, inv_mass, load, _source)
    try:
        for i in range(n_steps):
            integrator.step(i * dt, dt)
        np.testing.assert_allclose(integrator.u.array_r, u, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(integrator.v.array_r, v, rtol=1e-12, atol=1e-13)
        assert np.abs(u).max() > 0
    finally:
        integrator.destroy()
        K_p.destroy()
        C_p.destroy()


# ---------------------------------------------------------------------------
# Sensor decimation
# ---------------------------------------------------------------------------
def test_sample_stride_rate_and_limits():
    assert sample_stride(1e-8, 1000, 10e6) == 10
    assert sample_stride(1e-8, 1000, 1e9) == 1  # faster than the solver: every step
    # Nyquist guard: 1/(2.5 * 5 MHz * 10 ns) = 8
    assert sample_stride(1e-8, 1000, 1e6, f_max=5e6) == 8
    # Minimum sample count: 1000 steps, 101 samples incl. t = 0 → stride 10
    assert sample_stride(1e-8, 1000, 1e3, min_samples=101) == 10


@pytest.mark.parametrize("dt,rate", [(0.0, 1e6), (1e-8, 0.0), (np.nan, 1e6), (1e-8, -1.0)])
def test_sample_stride_rejects_invalid(dt, rate):
    with pytest.raises(ValueError):
        sample_stride(dt, 100, rate)


@pytest.mark.parametrize("requested_dt", [7.3e-9, 1e-8, 3.3e-9])
def test_sampled_time_grid_is_uniform_and_ends_at_duration(requested_dt):
    from wavesimfem.solvers import _sampled_time_grid

    layout = SimpleNamespace(duration=10e-6, mesh_frequency=3e6)
    cfg = {
        "sensor_sample_rate": 23e6,
        "disp_f_max": 2e6,
        "f_center": 1e6,
        "run_postprocess": False,
    }
    dt, n_steps, stride = _sampled_time_grid(cfg, layout, requested_dt)
    assert n_steps % stride == 0
    assert dt <= requested_dt * (1 + 1e-12)
    assert n_steps * dt == pytest.approx(layout.duration)
    assert 1.0 / (2.0 * stride * dt) >= 1.25 * 3e6 * (1 - 1e-9)  # Nyquist above 3 f0


def test_array_engine_decimated_history_matches_full():
    from wavesimfem.device import CPU_DEVICE
    from wavesimfem.solvers_gpu import integrate_explicit_rk4, make_backend

    K, C, inv_mass, load = _damped_system()
    S = sp.csr_matrix(sp.eye(len(load), format="csr")[:4])
    common = dict(dt=2e-3, n_steps=120, source=_source, backend=make_backend(CPU_DEVICE, "double"))
    full = integrate_explicit_rk4(K, C, inv_mass, load, S, **common)
    sparse = integrate_explicit_rk4(K, C, inv_mass, load, S, sample_every=6, **common)
    assert sparse.times.shape == (21,)
    np.testing.assert_allclose(sparse.times, full.times[::6])
    np.testing.assert_allclose(sparse.sensor_data_y, full.sensor_data_y[::6], rtol=1e-13)
    with pytest.raises(ValueError, match="sample_every"):
        integrate_explicit_rk4(K, C, inv_mass, load, S, sample_every=0, **common)


# ---------------------------------------------------------------------------
# Sparse operator helpers
# ---------------------------------------------------------------------------
def test_drop_zero_entries_preserves_matvec():
    from wavesimfem.solvers import _drop_zero_entries

    rng = np.random.default_rng(3)
    dense = rng.standard_normal((12, 12))
    dense[np.abs(dense) < 0.8] = 0.0
    pattern = sp.csr_matrix(np.ones((12, 12)))  # store every entry, including zeros
    pattern.data[:] = dense.ravel()
    original = _petsc_aij(pattern)
    filtered = _drop_zero_entries(original)
    try:
        assert filtered.getInfo()["nz_used"] == np.count_nonzero(dense)
        x = original.createVecRight()
        x.array[:] = rng.standard_normal(12)
        y0, y1 = original.createVecLeft(), filtered.createVecLeft()
        original.mult(x, y0)
        filtered.mult(x, y1)
        np.testing.assert_allclose(y1.array_r, y0.array_r, rtol=1e-14, atol=1e-14)
        for vec in (x, y0, y1):
            vec.destroy()
    finally:
        original.destroy()
        filtered.destroy()


# ---------------------------------------------------------------------------
# FEM-level checks on a small plate
# ---------------------------------------------------------------------------
def _small_plate_cfg(**overrides):
    from wavesimfem.config import parse_and_resolve_config

    cfg = {
        "solver": {"type": "explicit", "CFL": 0.8},
        "geometry": {"sensing_length": 0.01, "absorb_length": 0.003},
        "excitation": {"f_center": 5e5, "source_x": 0.004},
        "sensors": {"sensor_offset": 0.001, "sensor_pitch": 0.001, "n_sensors": 4},
        "mesh": {"max_frequency": 5e5, "nx": 26, "ny_per_region": [2], "degree": 2},
        "structure": {"mode": "1_layer", "single_layer": dict(ALUMINUM)},
        "postprocess": {"run_postprocess": False},
    }
    for dotted, value in overrides.items():
        section, key = dotted.split("__")
        cfg.setdefault(section, {})[key] = value
    return parse_and_resolve_config(custom_cfg=cfg)


def _assembled(form):
    from dolfinx import fem
    from dolfinx.fem.petsc import assemble_matrix

    matrix = assemble_matrix(fem.form(form), bcs=[])
    matrix.assemble()
    return matrix


@pytest.mark.parametrize("degree,nx,rows", [(1, 43, 3), (2, 26, 2)])
def test_element_bounds_dominate_assembled_bounds(degree, nx, rows):
    """The pre-mesh element bounds never underestimate the solver's Gershgorin bounds."""
    from wavesimfem.materials import element_operator_bounds
    from wavesimfem.solvers import build_problem

    cfg = _small_plate_cfg(mesh__degree=degree, mesh__nx=nx, mesh__ny_per_region=[rows])
    problem = build_problem(cfg)
    u, v = problem.trial_fn, problem.test_fn
    K, M, C = (
        _assembled(f) for f in (problem.stiffness(u, v), problem.mass(u, v), problem.damping(u, v))
    )
    try:
        ones = M.createVecRight()
        ones.set(1.0)
        lumped = M.createVecLeft()
        M.mult(ones, lumped)
        mass = lumped.array_r[: problem.n_owned_dofs].copy()
        b_k = operator_bound(K, mass)
        b_c = operator_bound(C, mass)
        est_k, est_c = element_operator_bounds(cfg, problem.summary, problem.layout)
        assert est_k >= b_k * (1 - 1e-9)
        assert est_c >= b_c * (1 - 1e-9)
        assert est_k <= 4.0 * b_k  # and remain reasonably sharp
        for vec in (ones, lumped):
            vec.destroy()
    finally:
        for mat in (K, M, C):
            mat.destroy()


def test_rigid_body_near_nullspace_is_orthonormal_kernel():
    from wavesimfem.solvers import _rigid_body_near_nullspace, build_problem

    problem = build_problem(_small_plate_cfg())
    near_nullspace, storage = _rigid_body_near_nullspace(problem.V)
    K = _assembled(problem.stiffness(problem.trial_fn, problem.test_fn))
    try:
        modes = near_nullspace.getVecs()
        assert len(modes) == 3 and storage  # storage keeps the mode memory alive
        gram = np.array([[a.dot(b) for b in modes] for a in modes])
        np.testing.assert_allclose(gram, np.eye(3), atol=1e-10)
        y = K.createVecLeft()
        k_norm = K.norm(PETSc.NormType.INFINITY)
        for mode in modes:
            K.mult(mode, y)
            assert y.norm(PETSc.NormType.INFINITY) <= 1e-9 * k_norm * mode.norm(
                PETSc.NormType.INFINITY
            )
        y.destroy()
    finally:
        K.destroy()
        near_nullspace.destroy()


def test_preflight_explicit_step_is_conservative():
    """Preflight explicit Δt (element bounds) ≤ the solver's assembled-bound step."""
    from wavesimfem.materials import explicit_time_step_estimate
    from wavesimfem.solvers import build_problem

    cfg = _small_plate_cfg()
    problem = build_problem(cfg)
    u, v = problem.trial_fn, problem.test_fn
    K, M, C = (
        _assembled(f) for f in (problem.stiffness(u, v), problem.mass(u, v), problem.damping(u, v))
    )
    try:
        mass = M.getRowSum().array_r[: problem.n_owned_dofs].copy()
        dt_solver = (
            cfg["CFL"]
            * RK4_STABILITY_RADIUS
            / max(np.sqrt(operator_bound(K, mass)), operator_bound(C, mass))
        )
        assert explicit_time_step_estimate(cfg, problem.summary, problem.layout) <= dt_solver * (
            1 + 1e-12
        )
    finally:
        for mat in (K, M, C):
            mat.destroy()


def test_explicit_snapshot_times_and_fields_match_across_backends(monkeypatch):
    from wavesimfem import solvers
    from wavesimfem.device import CPU_DEVICE

    problem = solvers.build_problem(
        _small_plate_cfg(solver__duration=2e-6, output__target_snapshots=5)
    )
    recordings = []
    for device in (None, CPU_DEVICE):
        snapshots = []

        def writer(stack, problem, u, snapshots=snapshots):
            return SimpleNamespace(write=lambda t: snapshots.append((t, u.x.array.copy())))

        monkeypatch.setattr(solvers, "_vtx_writer", writer)
        result = solvers.solve_explicit(problem, device=device)
        assert snapshots[0][0] == 0.0
        assert snapshots[-1][0] == pytest.approx(result.n_steps * result.dt)
        np.testing.assert_array_equal(snapshots[0][1], 0)
        recordings.append(snapshots)
    np.testing.assert_array_equal([t for t, _ in recordings[0]], [t for t, _ in recordings[1]])
    np.testing.assert_allclose(
        [u for _, u in recordings[0]], [u for _, u in recordings[1]], rtol=1e-10, atol=1e-20
    )


def _scipy_csr(mat):
    indptr, indices, values = mat.getValuesCSR()
    return sp.csr_matrix((values, indices, indptr), shape=mat.getSize())


@pytest.mark.parametrize("beta", [None, 2e-7])  # wave-speed- and damping-limited steps
def test_rk4_is_stable_just_below_the_rigorous_limit(beta):
    """Free vibration of an assembled FE system at 0.999 \u00d7 R / max(\u221aB_K, B_C) stays bounded."""
    from wavesimfem.solvers import build_problem

    overrides = {} if beta is None else {"absorbing_boundaries__beta": beta}
    problem = build_problem(
        _small_plate_cfg(mesh__degree=1, mesh__nx=43, mesh__ny_per_region=[3], **overrides)
    )
    u, v = problem.trial_fn, problem.test_fn
    mats = [
        _assembled(f) for f in (problem.stiffness(u, v), problem.mass(u, v), problem.damping(u, v))
    ]
    try:
        K, M, C = (_scipy_csr(m) for m in mats)
    finally:
        for mat in mats:
            mat.destroy()
    mass = np.asarray(M.sum(axis=1)).ravel()
    b_k = float(np.max(np.abs(K).sum(axis=1).A1 / mass))
    b_c = float(np.max(np.abs(C).sum(axis=1).A1 / mass))
    assert bool(b_c > np.sqrt(b_k)) is (beta is not None)
    dt = 0.999 * RK4_STABILITY_RADIUS / max(np.sqrt(b_k), b_c)

    rng = np.random.default_rng(0)
    u_n, v_n = rng.standard_normal(len(mass)), np.zeros(len(mass))
    inv_mass = 1.0 / mass

    def energy(uu, vv):
        return 0.5 * vv @ (mass * vv) + 0.5 * uu @ (K @ uu)

    e0 = energy(u_n, v_n)
    for i in range(3000):
        u_n, v_n = rk4_step(u_n, v_n, i * dt, dt, lambda uu, vv, t: -inv_mass * (K @ uu + C @ vv))
    assert np.isfinite(u_n).all()
    assert energy(u_n, v_n) <= 1.01 * e0
