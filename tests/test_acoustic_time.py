"""Independent oscillator, scaling, and parallel checks of acoustic integrators."""

from __future__ import annotations

import numpy as np
import pytest
from petsc4py import PETSc
from scipy.linalg import eigvals, expm

from wavesimfem.acoustic_time import (
    AcousticOperators,
    CoupledArrayRK4,
    CoupledNewmark,
    CoupledRK4,
    coupled_rk4_bounds,
)
from wavesimfem.numerics import RK4_STABILITY_RADIUS


def _matrix(values, comm=PETSc.COMM_SELF):
    values = np.asarray(values, dtype=PETSc.ScalarType)
    mat = PETSc.Mat().createAIJ(size=values.shape, comm=comm)
    mat.setUp()
    lo, hi = mat.getOwnershipRange()
    if hi > lo and values.shape[1]:
        mat.setValues(
            np.arange(lo, hi, dtype=PETSc.IntType),
            np.arange(values.shape[1], dtype=PETSc.IntType),
            values[lo:hi],
        )
    mat.assemble()
    return mat


def _operators(*, damping=0.0, scale=1.0, comm=PETSc.COMM_SELF):
    return AcousticOperators(
        _matrix([[1.0]], comm),
        _matrix([[4.0]], comm),
        _matrix([[damping]], comm),
        _matrix([[1.0 / scale**2]], comm),
        _matrix([[9.0 / scale**2]], comm),
        _matrix([[damping / scale**2]], comm),
        _matrix([[2.0 / scale]], comm),
    )


def _destroy(ops):
    for matrix in (ops.Ms, ops.Ks, ops.Ds, ops.Mp, ops.Kp, ops.Dp, ops.C):
        if matrix is not None:
            matrix.destroy()


def _mode_initial(ops, scale=1.0):
    # Independent analytic roots: lambda**2 - 17 lambda + 36 = 0.
    lam = (17.0 + np.sqrt(145.0)) / 2.0
    pressure = scale * (lam - 4.0) / 2.0
    ns = ops.Ms.getLocalSize()[0]
    np_ = ops.Mp.getLocalSize()[0]
    return {"u": np.ones(ns), "p": np.full(np_, pressure)}, lam, pressure


@pytest.mark.parametrize(
    "method,expected_order", [(CoupledRK4, 4), (CoupledArrayRK4, 4), (CoupledNewmark, 2)]
)
def test_coupled_mode_has_expected_temporal_order(method, expected_order):
    ops = _operators()
    initial, lam, pressure = _mode_initial(ops)
    omega, end = np.sqrt(lam), 0.8
    errors = []
    try:
        for count in (20, 40, 80):
            dt = end / count
            solver = (
                method(ops, dt, initial=initial)
                if method is CoupledNewmark
                else method(ops, initial=initial)
            )
            try:
                np.testing.assert_allclose(solver.a.array_r, -lam * initial["u"], rtol=1e-13)
                np.testing.assert_allclose(solver.pddot.array_r, -lam * initial["p"], rtol=1e-13)
                for step in range(count):
                    solver.step(step * dt, dt)
                errors.append(
                    np.linalg.norm(
                        np.array(
                            [
                                solver.u.array_r[0] - np.cos(omega * end),
                                solver.p.array_r[0] / pressure - np.cos(omega * end),
                                solver.v.array_r[0] / omega + np.sin(omega * end),
                            ]
                        )
                    )
                )
                integral = pressure * np.sin(omega * end) / omega
                assert solver.pressure_integral.array_r[0] == pytest.approx(
                    integral, abs=0.01 if expected_order == 2 else 1e-4
                )
            finally:
                solver.destroy()
        observed = np.log2(np.array(errors[:-1]) / errors[1:])
        assert np.all(observed > expected_order - 0.12), (errors, observed)
    finally:
        _destroy(ops)


@pytest.mark.parametrize("method", [CoupledRK4, CoupledArrayRK4, CoupledNewmark])
def test_pressure_unit_scaling_preserves_coupled_solution(method):
    outputs, limits = [], []
    for scale in (1.0, 1e9):
        ops = _operators(damping=0.3, scale=scale)
        initial, _, _ = _mode_initial(ops, scale)
        dt = 0.01
        solver = (
            method(ops, dt, initial=initial)
            if method is CoupledNewmark
            else method(ops, initial=initial)
        )
        try:
            for step in range(50):
                solver.step(step * dt, dt)
            outputs.append(np.array([solver.u.array_r[0], solver.p.array_r[0] / scale]))
            limits.append(coupled_rk4_bounds(ops)["dt_limit"])
        finally:
            solver.destroy()
            _destroy(ops)
    np.testing.assert_allclose(outputs[1], outputs[0], rtol=1e-10, atol=1e-11)
    assert limits[1] == pytest.approx(limits[0], rel=1e-14)


def test_coupled_damped_certificate_dominates_independent_spectrum():
    ops = _operators(damping=3.0)
    try:
        # Independently invert the displayed two-field mass block.
        mass = np.array([[1.0, 0.0], [-2.0, 1.0]])
        stiffness = np.array([[4.0, 2.0], [0.0, 9.0]])
        damping = np.eye(2) * 3.0
        operator = np.block(
            [
                [np.zeros((2, 2)), np.eye(2)],
                [-np.linalg.solve(mass, stiffness), -np.linalg.solve(mass, damping)],
            ]
        )
        spectrum = eigvals(operator)
        bounds = coupled_rk4_bounds(ops)
        assert np.max(np.real(spectrum)) <= 1e-12
        assert np.max(np.abs(spectrum)) <= bounds["radius"] * (1.0 + 1e-12)
        assert bounds["coupling"] == pytest.approx(2.0)
        assert bounds["damping"] == pytest.approx(3.0)
        assert bounds["dt_limit"] * bounds["radius"] == pytest.approx(RK4_STABILITY_RADIUS)
    finally:
        _destroy(ops)


@pytest.mark.parametrize("method", [CoupledRK4, CoupledArrayRK4, CoupledNewmark])
def test_same_stage_time_dependent_forcing_matches_independent_solution(method):
    ops = _operators(damping=0.4)
    dt, end = 0.002, 0.5

    def force(t, fs, fp):
        fs.array[:] = np.cos(5.0 * t)
        fp.array[:] = 0.5 * np.sin(5.0 * t)

    # Autonomous augmented system includes cos/sin source states. This is an
    # independent matrix-exponential reference, rather than the RK4 update.
    mass = np.array([[1.0, 0.0], [-2.0, 1.0]])
    stiffness = np.array([[4.0, 2.0], [0.0, 9.0]])
    damping = np.eye(2) * 0.4
    base = np.block(
        [
            [np.zeros((2, 2)), np.eye(2)],
            [-np.linalg.solve(mass, stiffness), -np.linalg.solve(mass, damping)],
        ]
    )
    augmented = np.zeros((6, 6))
    augmented[:4, :4] = base
    augmented[2:4, 4:6] = np.linalg.solve(mass, np.diag([1.0, 0.5]))
    augmented[4:6, 4:6] = [[0.0, -5.0], [5.0, 0.0]]
    expected = expm(end * augmented) @ np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    solver = method(ops, dt, force) if method is CoupledNewmark else method(ops, force)
    try:
        assert solver.a.array_r[0] == pytest.approx(1.0)
        assert solver.pddot.array_r[0] == pytest.approx(2.0)
        for step in range(round(end / dt)):
            solver.step(step * dt, dt)
        actual = np.array(
            [solver.u.array_r[0], solver.p.array_r[0], solver.v.array_r[0], solver.pdot.array_r[0]]
        )
        np.testing.assert_allclose(
            actual, expected[:4], atol=3e-5 if method is CoupledNewmark else 2e-9, rtol=0
        )
    finally:
        solver.destroy()
        _destroy(ops)


@pytest.mark.parametrize("method", [CoupledRK4, CoupledArrayRK4, CoupledNewmark])
def test_pure_fluid_uses_zero_size_solid_field(method):
    ops = AcousticOperators(
        _matrix(np.zeros((0, 0))),
        _matrix(np.zeros((0, 0))),
        None,
        _matrix([[2.0]]),
        _matrix([[8.0]]),
        None,
        _matrix(np.zeros((0, 1))),
    )
    initial = {"p": np.array([1.0])}
    dt = 0.01
    solver = (
        method(ops, dt, initial=initial)
        if method is CoupledNewmark
        else method(ops, initial=initial)
    )
    try:
        for step in range(100):
            solver.step(step * dt, dt)
        assert solver.u.getSize() == 0
        assert solver.p.array_r[0] == pytest.approx(np.cos(2.0), abs=8e-5)
    finally:
        solver.destroy()
        _destroy(ops)


@pytest.mark.parametrize("method", [CoupledRK4, CoupledNewmark])
def test_distributed_coupled_mode(method):
    """Also invoked directly under mpiexec; each field may have empty local rows."""
    ops = _operators(comm=PETSc.COMM_WORLD)
    initial, lam, pressure = _mode_initial(ops)
    dt, end = 0.01, 0.3
    solver = (
        method(ops, dt, initial=initial)
        if method is CoupledNewmark
        else method(ops, initial=initial)
    )
    try:
        for step in range(round(end / dt)):
            solver.step(step * dt, dt)
        np.testing.assert_allclose(solver.u.array_r, np.cos(np.sqrt(lam) * end), atol=2e-4, rtol=0)
        np.testing.assert_allclose(
            solver.p.array_r, pressure * np.cos(np.sqrt(lam) * end), atol=2e-3, rtol=0
        )
    finally:
        solver.destroy()
        _destroy(ops)


@pytest.mark.parametrize("method", [CoupledRK4, CoupledNewmark])
def test_distributed_rectangular_coupling_matches_independent_exponential(method):
    """Differently sized distributed fields, consistent mass, damping, and C.T."""
    ns, np_ = 3, 5
    ms = np.eye(ns) * 2.0 + np.ones((ns, ns)) * 0.1
    mp = np.eye(np_) * 0.5 + np.ones((np_, np_)) * 0.03
    ks, kp = np.diag([4.0, 5.0, 6.0]), np.diag([2.0, 3.0, 4.0, 5.0, 6.0])
    ds, dp = 0.3 * ms, 0.2 * mp
    coupling = np.random.default_rng(42).uniform(-1.0, 1.0, (ns, np_))
    ops = AcousticOperators(
        *[_matrix(value, PETSc.COMM_WORLD) for value in (ms, ks, ds, mp, kp, dp, coupling)]
    )
    if method is CoupledRK4:
        ms, mp = np.diag(ms.sum(axis=1)), np.diag(mp.sum(axis=1))
    mass = np.block([[ms, np.zeros((ns, np_))], [-coupling.T, mp]])
    stiffness = np.block([[ks, coupling], [np.zeros((np_, ns)), kp]])
    damping = np.block([[ds, np.zeros((ns, np_))], [np.zeros((np_, ns)), dp]])
    size = ns + np_
    generator = np.block(
        [
            [np.zeros((size, size)), np.eye(size)],
            [-np.linalg.solve(mass, stiffness), -np.linalg.solve(mass, damping)],
        ]
    )
    augmented = np.zeros((2 * size + np_, 2 * size + np_))
    augmented[: 2 * size, : 2 * size] = generator
    augmented[2 * size :, ns:size] = np.eye(np_)
    positions = np.linspace(0.1, 1.0, size)
    velocities = np.full(size, -0.1)
    initial = {}
    for matrix, names, positions_, velocities_ in (
        (ops.Ms, ("u", "v"), positions[:ns], velocities[:ns]),
        (ops.Mp, ("p", "pdot"), positions[ns:], velocities[ns:]),
    ):
        lo, hi = matrix.getOwnershipRange()
        initial[names[0]], initial[names[1]] = positions_[lo:hi], velocities_[lo:hi]
    dt, end = 0.002, 0.1
    expected = expm(end * augmented) @ np.concatenate([positions, velocities, np.zeros(np_)])
    solver = (
        method(ops, dt, initial=initial)
        if method is CoupledNewmark
        else method(ops, initial=initial)
    )
    try:
        for step in range(round(end / dt)):
            solver.step(step * dt, dt)
        for state, matrix, reference in (
            (solver.u, ops.Ms, expected[:ns]),
            (solver.p, ops.Mp, expected[ns:size]),
            (solver.v, ops.Ms, expected[size : size + ns]),
            (solver.pdot, ops.Mp, expected[size + ns : 2 * size]),
            (solver.pressure_integral, ops.Mp, expected[2 * size :]),
        ):
            lo, hi = matrix.getOwnershipRange()
            np.testing.assert_allclose(
                state.array_r,
                reference[lo:hi],
                atol=5e-6 if method is CoupledNewmark else 2e-10,
                rtol=0,
            )
    finally:
        solver.destroy()
        _destroy(ops)


def test_acoustic_explicit_refuses_uncertified_step_and_cuda():
    ops = _operators()
    solver = CoupledRK4(ops)
    try:
        with pytest.raises(ValueError, match="coupled stability"):
            solver.step(0.0, solver.stable_dt() * 1.01)
        with pytest.raises(NotImplementedError, match="Use CoupledArrayRK4"):
            CoupledRK4(ops, device="cuda")
    finally:
        solver.destroy()
        _destroy(ops)


@pytest.mark.parametrize("method", [CoupledRK4, CoupledNewmark])
@pytest.mark.parametrize("invalid", ["conversion", "shape", "nonfinite", "vector", "mapping"])
def test_distributed_initial_validation_is_collective(method, invalid):
    ops = _operators(comm=PETSc.COMM_WORLD)
    comm = ops.Ms.comm.tompi4py()
    initial, _, _ = _mode_initial(ops)
    vector = ops.Mp.createVecLeft()
    vector.set(0.0)
    if comm.rank == 0:
        if invalid == "conversion":
            initial["p"] = ["not a number"]
        elif invalid == "shape":
            initial["p"] = np.zeros((2, 2))
        elif invalid == "nonfinite":
            initial["p"][:] = np.nan
        elif invalid == "vector":
            vector.array[:] = np.nan
            initial["p"] = vector
        else:
            initial = []
    try:
        with pytest.raises(ValueError, match="Invalid acoustic initial state"):
            if method is CoupledNewmark:
                method(ops, 0.01, initial=initial)
            else:
                method(ops, initial=initial)
        # Validation must leave the communicator and caller-owned operators usable.
        assert comm.allreduce(1) == comm.size
        solver = method(ops, 0.01) if method is CoupledNewmark else method(ops)
        solver.destroy()
    finally:
        vector.destroy()
        _destroy(ops)


@pytest.mark.parametrize("method", [CoupledRK4, CoupledArrayRK4, CoupledNewmark])
def test_closed_cavity_constant_pressure_mode_is_preserved(method):
    """The physical zero-frequency pressure mode must survive an exactly singular Kp."""
    ops = AcousticOperators(
        _matrix(np.zeros((0, 0))),
        _matrix(np.zeros((0, 0))),
        None,
        _matrix(np.diag([2.0, 3.0])),
        _matrix([[1.0, -1.0], [-1.0, 1.0]]),
        None,
        _matrix(np.zeros((0, 2))),
    )
    initial = {"p": np.full(2, 3.0), "pdot": np.full(2, 2.0)}
    dt, end = 0.005, 0.5
    solver = (
        method(ops, dt, initial=initial)
        if method is CoupledNewmark
        else method(ops, initial=initial)
    )
    try:
        for step in range(round(end / dt)):
            solver.step(step * dt, dt)
        np.testing.assert_allclose(solver.p.array_r, 3.0 + 2.0 * end, atol=1e-10, rtol=0)
        np.testing.assert_allclose(solver.pdot.array_r, 2.0, atol=1e-9, rtol=0)
        np.testing.assert_allclose(
            solver.pressure_integral.array_r, 3.0 * end + end**2, atol=1e-10, rtol=0
        )
    finally:
        solver.destroy()
        _destroy(ops)


@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("scale", [1.0, 1e9, 1e25])
def test_array_rk4_mass_normalization_preserves_scaled_pressure(precision, scale):
    """Independent modal solution also covers masses below float32's range."""
    ops = _operators(scale=scale)
    initial, lam, pressure = _mode_initial(ops, scale)
    dt, end = 0.002, 0.2
    solver = CoupledArrayRK4(ops, initial=initial, precision=precision)
    try:
        for step in range(round(end / dt)):
            solver.step(step * dt, dt)
        tolerance = 2e-6 if precision == "single" else 2e-10
        np.testing.assert_allclose(
            [solver.u.array_r[0], solver.p.array_r[0] / pressure],
            np.cos(np.sqrt(lam) * end),
            rtol=tolerance,
            atol=tolerance,
        )
        assert solver.pressure_integral.array_r[0] / pressure == pytest.approx(
            np.sin(np.sqrt(lam) * end) / np.sqrt(lam), abs=tolerance
        )
        assert solver.backend.dtype == np.dtype(np.float32 if precision == "single" else np.float64)
    finally:
        solver.destroy()
        _destroy(ops)


@pytest.mark.parametrize("precision", ["double", "single"])
def test_array_rk4_rectangular_forced_system_matches_petsc(precision):
    ms = np.diag([2.0, 3.0])
    mp = np.diag([0.2, 0.5, 0.8])
    ops = AcousticOperators(
        *[
            _matrix(value)
            for value in (
                ms,
                np.diag([4.0, 5.0]),
                0.2 * ms,
                mp,
                np.diag([3.0, 5.0, 6.0]),
                0.4 * mp,
                np.array([[0.2, 0.3, -0.1], [-0.3, 0.1, 0.5]]),
            )
        ]
    )

    def force(t, fs, fp):
        fs.array[:] = [np.sin(t), np.cos(2.0 * t)]
        fp.array[:] = [np.sin(0.3 * t), np.cos(t), np.cos(3.0 * t)]

    initial = {"p": np.array([0.2, -0.1, 0.5]), "v": np.array([0.3, -0.2])}
    array = CoupledArrayRK4(ops, force, initial=initial, precision=precision)
    petsc = CoupledRK4(ops, force, initial=initial)
    try:
        assert array.stable_dt() == petsc.stable_dt()
        for step in range(100):
            array.step(step * 0.003, 0.003)
            petsc.step(step * 0.003, 0.003)
        for name in ("u", "v", "p", "pdot", "a", "pddot", "pressure_integral"):
            np.testing.assert_allclose(
                getattr(array, name).array_r,
                getattr(petsc, name).array_r,
                rtol=2e-5 if precision == "single" else 1e-12,
                atol=2e-6 if precision == "single" else 1e-13,
            )
    finally:
        array.destroy()
        petsc.destroy()
        _destroy(ops)


def test_array_rk4_cuda_request_never_silently_falls_back(monkeypatch):
    from wavesimfem import device

    monkeypatch.setattr(device, "detect_available_devices", lambda: [device.CPU_DEVICE])
    ops = _operators()
    try:
        with pytest.raises(device.DeviceUnavailableError, match="not available"):
            CoupledArrayRK4(ops, device="cuda")
    finally:
        _destroy(ops)


def test_array_rk4_cuda_matches_cpu_when_available():
    from wavesimfem.device import is_cupy_available

    if not is_cupy_available():
        pytest.skip("A CuPy CUDA device is required for the accelerator comparison")
    ops = _operators(scale=1e9)
    initial, _, _ = _mode_initial(ops, 1e9)
    cpu = CoupledArrayRK4(ops, initial=initial, device="cpu")
    gpu = CoupledArrayRK4(ops, initial=initial, device="cuda")
    try:
        assert gpu.device_info.is_gpu
        for step in range(100):
            cpu.step(step * 0.002, 0.002)
            gpu.step(step * 0.002, 0.002)
        for name in ("u", "v", "p", "pdot", "a", "pddot", "pressure_integral"):
            np.testing.assert_allclose(
                getattr(gpu, name).array_r, getattr(cpu, name).array_r, rtol=1e-11, atol=1e-11
            )
    finally:
        cpu.destroy()
        gpu.destroy()
        _destroy(ops)


def _separable_case(ops):
    from wavesimfem.acoustic_time import SeparableForcing

    solid, pressure = ops.Ms.createVecRight(), ops.Mp.createVecRight()
    solid.set(1.0)
    pressure.set(0.5)

    def coefficients(t):
        return np.cos(5.0 * t), np.sin(5.0 * t)

    def callback(t, fs, fp):
        fs.array[:] = np.cos(5.0 * t)
        fp.array[:] = 0.5 * np.sin(5.0 * t)

    forcing = SeparableForcing((solid, None), (None, pressure), coefficients)
    return forcing, callback, (solid, pressure)


@pytest.mark.parametrize("method", [CoupledRK4, CoupledArrayRK4, CoupledNewmark])
def test_separable_forcing_matches_equivalent_callback(method):
    ops = _operators(damping=0.4)
    forcing, callback, vectors = _separable_case(ops)
    dt = 0.002
    build = (
        (lambda f: method(ops, dt, f)) if method is CoupledNewmark else (lambda f: method(ops, f))
    )
    separable, generic = build(forcing), build(callback)
    try:
        for step in range(200):
            separable.step(step * dt, dt)
            generic.step(step * dt, dt)
        for name in ("u", "v", "p", "pdot", "a", "pddot", "pressure_integral"):
            np.testing.assert_allclose(
                getattr(separable, name).array_r,
                getattr(generic, name).array_r,
                rtol=1e-13,
                atol=1e-15,
            )
    finally:
        separable.destroy()
        generic.destroy()
        for vector in vectors:
            vector.destroy()
        _destroy(ops)


@pytest.mark.parametrize("method", [CoupledRK4, CoupledArrayRK4])
def test_rk4_reuses_end_of_step_acceleration(method):
    ops = _operators(damping=0.4)
    calls = []

    def force(t, fs, fp):
        calls.append(t)
        fs.array[:] = np.cos(5.0 * t)

    solver = method(ops, force)
    try:
        assert len(calls) == 1
        for step in range(10):
            solver.step(step * 0.01, 0.01)
        # Four evaluations per step; the first stage reuses a(t_n).
        assert len(calls) == 1 + 4 * 10
        solver.step(0.5, 0.01)  # time jump: the cached acceleration is stale
        assert len(calls) == 1 + 4 * 10 + 5
        if method is CoupledRK4:
            solver.u.array[:] += 1e-3  # external edit invalidates the cache
            solver.step(0.51, 0.01)
            assert len(calls) == 1 + 4 * 10 + 5 + 5
    finally:
        solver.destroy()
        _destroy(ops)


def test_array_rk4_mirrors_download_lazily(monkeypatch):
    ops = _operators()
    initial, _, _ = _mode_initial(ops)
    solver = CoupledArrayRK4(ops, initial=initial)
    downloads = []
    original = solver.backend.to_host
    monkeypatch.setattr(
        solver.backend, "to_host", lambda array: downloads.append(1) or original(array)
    )
    try:
        for step in range(20):
            solver.step(step * 0.01, 0.01)
        assert not downloads
        first = solver.p.array_r.copy()
        assert len(downloads) == 1
        np.testing.assert_array_equal(solver.p.array_r, first)
        assert len(downloads) == 1
        solver.step(0.2, 0.01)
        assert not np.array_equal(solver.p.array_r, first)
        assert len(downloads) == 2
    finally:
        solver.destroy()
        _destroy(ops)


def test_array_rk4_reports_divergence_when_state_is_read():
    ops = _operators()
    solver = CoupledArrayRK4(ops, initial={"u": np.ones(1)})
    try:
        solver._array_state[0][:] = np.nan
        solver._stale.add("u")
        with pytest.raises(FloatingPointError, match="non-finite"):
            _ = solver.u
    finally:
        solver.destroy()
        _destroy(ops)


def test_distributed_coupled_newmark_iterative_fallback(monkeypatch):
    """When _direct_solver returns None, CoupledNewmark keeps its Nest operator
    intact and solves via field-split GMRES (serial and MPI)."""
    from wavesimfem import acoustic_time

    ops = _operators(comm=PETSc.COMM_WORLD)
    initial, lam, pressure = _mode_initial(ops)
    dt, end = 0.01, 0.3
    monkeypatch.setattr(acoustic_time, "_direct_solver", lambda *_a, **_kw: None)
    solver = CoupledNewmark(ops, dt, initial=initial)
    try:
        assert solver._flat_rhs is None
        assert solver._A.getType() == PETSc.Mat.Type.NEST
        for step in range(round(end / dt)):
            solver.step(step * dt, dt)
        np.testing.assert_allclose(solver.u.array_r, np.cos(np.sqrt(lam) * end), atol=2e-4, rtol=0)
        np.testing.assert_allclose(
            solver.p.array_r, pressure * np.cos(np.sqrt(lam) * end), atol=2e-3, rtol=0
        )
    finally:
        solver.destroy()
        _destroy(ops)
