"""Tests for device selection, the sparse sensor operator, and the array RK4 engine."""

from __future__ import annotations

import dolfinx
import numpy as np
import pytest
import scipy.sparse as sp
from mpi4py import MPI

from wavesimfem.device import (
    CPU_DEVICE,
    DeviceInfo,
    DeviceUnavailableError,
    detect_available_devices,
    normalize_device_request,
    select_device,
)
from wavesimfem.numerics import rk4_step
from wavesimfem.solvers_gpu import (
    build_sensor_interpolation_matrix,
    integrate_explicit_rk4,
    make_backend,
    solve_explicit_rk4_device,
)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def test_device_detection_and_selection():
    devices = detect_available_devices()
    assert any(d.backend == "scipy" and d.device == "cpu" for d in devices)

    dev_auto = select_device("auto")
    assert isinstance(dev_auto, DeviceInfo)
    assert dev_auto.summary()

    dev_cpu = select_device("cpu")
    assert dev_cpu.device == "cpu"
    assert not dev_cpu.is_gpu


@pytest.mark.parametrize("bad", ["non_existent_device_xyz", "gpu", "cuda:x", "", 3])
def test_invalid_device_request_raises(bad):
    with pytest.raises(ValueError):
        select_device(bad)


@pytest.mark.parametrize(
    "raw,expected",
    [("AUTO", "auto"), (" cpu ", "cpu"), ("cuda", "cuda:0"), ("cuda:1", "cuda:1")],
)
def test_normalize_device_request(raw, expected):
    assert normalize_device_request(raw) == expected


def test_explicit_unavailable_gpu_request_raises(monkeypatch):
    import wavesimfem.device as device_mod

    monkeypatch.setattr(device_mod, "detect_available_devices", lambda: [CPU_DEVICE])
    with pytest.raises(DeviceUnavailableError):
        device_mod.select_device("cuda:0")
    assert device_mod.select_device("auto").device == "cpu"


# ---------------------------------------------------------------------------
# Sensor interpolation operator
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("degree", [1, 2])
def test_sensor_interpolation_matrix_exactness(degree):
    """Sparse operator equals DOLFINx point evaluation, including sensors that
    share a cell or sit in neighbouring cells (exercises the colouring)."""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_SELF, 6, 6)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", degree, (2,)))
    n_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

    pts = np.array(
        [
            [0.25, 0.25, 0.0],
            [0.26, 0.24, 0.0],  # same cell as previous
            [0.40, 0.25, 0.0],  # neighbouring cell
            [0.50, 0.50, 0.0],  # mesh vertex
            [0.75, 0.85, 0.0],
            [1.00, 0.17, 0.0],  # boundary
        ]
    )
    tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    candidates = dolfinx.geometry.compute_collisions_points(tree, pts)
    colliding = dolfinx.geometry.compute_colliding_cells(mesh, candidates, pts)
    active = [colliding.links(i)[0] for i in range(len(pts))]

    S = build_sensor_interpolation_matrix(V, pts, active, n_owned)
    assert S.shape == (2 * len(pts), n_owned)

    u_test = dolfinx.fem.Function(V)
    rand_vals = np.random.default_rng(42).standard_normal(len(u_test.x.array))
    u_test.x.array[:] = rand_vals
    expected = u_test.eval(pts, active).ravel()
    np.testing.assert_allclose(S @ rand_vals[:n_owned], expected, atol=1e-13, rtol=0)


def test_sensor_interpolation_matrix_empty():
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_SELF, 2, 2)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (2,)))
    S = build_sensor_interpolation_matrix(V, np.zeros((0, 3)), [], 18)
    assert S.shape == (0, 18)


# ---------------------------------------------------------------------------
# RK4 engine
# ---------------------------------------------------------------------------
def _random_damped_system(n: int = 24, seed: int = 0):
    rng = np.random.default_rng(seed)
    lap = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n), format="csr")
    K = (lap * 4.0e2).tocsr()
    C = sp.diags(rng.uniform(0.0, 0.5, n), format="csr")
    inv_mass = rng.uniform(0.5, 2.0, n)
    load = np.zeros(n)
    load[n // 3] = 1.0
    S = sp.csr_matrix(sp.eye(n, format="csr")[:6])  # 3 sensors: rows (x, y) pairs
    return K, C, inv_mass, load, S


def _source(t: float) -> float:
    return float(np.sin(40.0 * t) * np.exp(-((t - 0.05) ** 2) / 1e-3))


def _reference_rk4(K, C, inv_mass, load, S, dt, n_steps):
    n = len(inv_mass)
    u, v = np.zeros(n), np.zeros(n)
    hist = [S @ u]
    for i in range(n_steps):
        u, v = rk4_step(
            u, v, i * dt, dt, lambda uu, vv, t: inv_mass * (_source(t) * load - K @ uu - C @ vv)
        )
        hist.append(S @ u)
    return np.asarray(hist)


@pytest.mark.parametrize("chunk_bytes", [64 * 1024**2, 3 * 6 * 8 + 1])
def test_engine_matches_reference_rk4(chunk_bytes):
    """Array engine reproduces numerics.rk4_step (incl. a tiny sensor ring buffer)."""
    K, C, inv_mass, load, S = _random_damped_system()
    dt, n_steps = 2e-3, 157
    ref = _reference_rk4(K, C, inv_mass, load, S, dt, n_steps)

    result = integrate_explicit_rk4(
        K,
        C,
        inv_mass,
        load,
        S,
        dt=dt,
        n_steps=n_steps,
        source=_source,
        backend=make_backend(CPU_DEVICE, "double"),
        history_chunk_bytes=chunk_bytes,
    )
    assert result.times.shape == (n_steps + 1,)
    assert result.times[-1] == pytest.approx(n_steps * dt)
    np.testing.assert_allclose(result.sensor_data_x, ref[:, 0::2], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(result.sensor_data_y, ref[:, 1::2], rtol=1e-12, atol=1e-15)
    assert np.abs(ref).max() > 0


def test_engine_single_precision_and_snapshots():
    K, C, inv_mass, load, S = _random_damped_system()
    snaps: list[tuple[int, float]] = []
    result = integrate_explicit_rk4(
        K,
        C,
        inv_mass,
        load,
        S,
        dt=2e-3,
        n_steps=20,
        source=_source,
        backend=make_backend(CPU_DEVICE, "single"),
        save_every=5,
        snapshot_callback=lambda step, t, u: snaps.append((step, t)),
    )
    assert result.sensor_data_x.dtype == np.float32
    assert [s for s, _ in snaps] == [0, 5, 10, 15]


def test_engine_detects_instability():
    K, C, inv_mass, load, S = _random_damped_system()
    with pytest.raises(FloatingPointError, match="Non-finite"):
        integrate_explicit_rk4(
            K,
            C,
            inv_mass,
            load,
            S,
            dt=1.0,  # far beyond the RK4 stability limit
            n_steps=400,
            source=lambda t: 1.0,
            backend=make_backend(CPU_DEVICE, "double"),
        )


@pytest.mark.parametrize("bad", [dict(n_steps=0), dict(dt=0.0), dict(dt=float("nan"))])
def test_engine_rejects_invalid_step_parameters(bad):
    K, C, inv_mass, load, S = _random_damped_system()
    kwargs = dict(dt=1e-3, n_steps=5) | bad
    with pytest.raises(ValueError):
        integrate_explicit_rk4(
            K,
            C,
            inv_mass,
            load,
            S,
            source=_source,
            backend=make_backend(CPU_DEVICE),
            **kwargs,
        )


def test_engine_rejects_inconsistent_shapes():
    K, C, inv_mass, load, S = _random_damped_system()
    with pytest.raises(ValueError, match="Inconsistent"):
        integrate_explicit_rk4(
            K,
            C,
            inv_mass[:-1],
            load,
            S,
            dt=1e-3,
            n_steps=2,
            source=_source,
            backend=make_backend(CPU_DEVICE),
        )


def test_legacy_device_wrapper_and_tuple_csr():
    K, C, inv_mass, load, S = _random_damped_system()
    times, sx, sy = solve_explicit_rk4_device(
        K_csr=(K.indptr, K.indices, K.data),
        C_csr=(C.indptr, C.indices, C.data),
        M_inv_np=inv_mass,
        f_spatial_np=load,
        sensor_mat_csr=S,
        dt=2e-3,
        Nsteps=10,
        source_func=_source,
        device_info=CPU_DEVICE,
    )
    assert times.shape == (11,) and sx.shape == (11, 3) and sy.shape == (11, 3)


@pytest.mark.gpu
def test_cupy_backend_matches_reference():
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no CUDA device")
    K, C, inv_mass, load, S = _random_damped_system()
    dt, n_steps = 2e-3, 50
    ref = _reference_rk4(K, C, inv_mass, load, S, dt, n_steps)
    gpu = DeviceInfo(backend="cupy", device="cuda:0", name="CUDA")
    result = integrate_explicit_rk4(
        K,
        C,
        inv_mass,
        load,
        S,
        dt=dt,
        n_steps=n_steps,
        source=_source,
        backend=make_backend(gpu, "double"),
    )
    np.testing.assert_allclose(result.sensor_data_x, ref[:, 0::2], rtol=1e-10, atol=1e-14)
