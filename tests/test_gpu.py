"""Tests for GPU and accelerated device detection, operators, and solvers."""

from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI
import dolfinx

from guidedwavefem.device import (
    DeviceInfo,
    detect_available_devices,
    select_device,
)
from guidedwavefem.solvers_gpu import (
    build_sensor_interpolation_matrix,
)
from guidedwavefem import run_simulation


def test_device_detection_and_selection():
    """Verify hardware scanning and fallback selection mechanisms."""
    devices = detect_available_devices()
    assert len(devices) >= 1
    assert any(d.backend == "scipy" and d.device == "cpu" for d in devices)

    # Auto selection should return a valid device
    dev_auto = select_device("auto")
    assert isinstance(dev_auto, DeviceInfo)
    assert dev_auto.summary()

    # Explicit CPU selection
    dev_cpu = select_device("cpu")
    assert dev_cpu.device == "cpu"
    assert not dev_cpu.is_gpu

    # Explicit invalid device should safely fallback to CPU
    dev_fallback = select_device("non_existent_device_xyz")
    assert dev_fallback.device == "cpu"


@pytest.mark.parametrize("degree", [1, 2])
def test_sensor_interpolation_matrix_exactness(degree):
    """Verify that sparse sensor matrix matches exact DOLFINx basis evaluation to machine precision."""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 6, 6)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", degree, (2,)))
    n_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

    pts = np.array(
        [
            [0.25, 0.25, 0.0],
            [0.50, 0.50, 0.0],
            [0.75, 0.85, 0.0],
        ]
    )
    tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    cell_candidates = dolfinx.geometry.compute_collisions_points(tree, pts)
    cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, pts)
    active = [cells.links(i)[0] for i in range(len(pts))]

    S = build_sensor_interpolation_matrix(V, pts, active, n_owned)
    assert S.shape == (2 * len(pts), n_owned)
    assert S.nnz > 0

    # Evaluate on random displacement field
    u_test = dolfinx.fem.Function(V)
    np.random.seed(42)
    rand_vals = np.random.randn(len(u_test.x.array))
    u_test.x.array[:] = rand_vals

    expected = u_test.eval(pts, active).flatten()
    predicted = S.dot(rand_vals[:n_owned])

    err = np.max(np.abs(expected - predicted))
    assert err < 1e-13, f"Sensor interpolation error too large: {err}"


def test_explicit_simulation_device_cpu(tmp_path):
    """Run full explicit simulation using the accelerated CPU device engine."""
    config = {
        "solver_type": "explicit",
        "device": "cpu",
        "precision": "double",
        "sensing_length": 0.01,
        "absorb_length": 0.003,
        "f_center": 5e5,
        "source_x": 0.004,
        "sensor_offset": 0.001,
        "sensor_pitch": 0.001,
        "n_sensors": 4,
        "duration": 5e-6,
        "max_frequency": 5e5,
        "min_elems_x": 10,
        "min_elems_y": 2,
        "elems_per_wavelength": 2,
        "degree": 2,
        "run_postprocess": False,
        "output_dir": str(tmp_path),
        "structure_mode": "1_layer",
        "single_layer": {
            "thickness": 0.001,
            "rho": 2700.0,
            "E": 70e9,
            "nu": 0.33,
            "material_model": "isotropic",
        },
    }
    path, _ = run_simulation(cfg=config)
    with np.load(path) as data:
        assert np.isfinite(data["sensor_data_y"]).all()
        assert np.max(np.abs(data["sensor_data_y"])) > 0
        assert data["times"][-1] == pytest.approx(5e-6)
        assert data["sensor_data_y"].dtype == np.float64


def test_explicit_simulation_single_precision(tmp_path):
    """Verify single-precision (float32) explicit simulation execution."""
    config = {
        "solver_type": "explicit",
        "device": "cpu",
        "precision": "single",
        "sensing_length": 0.01,
        "absorb_length": 0.003,
        "f_center": 5e5,
        "source_x": 0.004,
        "sensor_offset": 0.001,
        "sensor_pitch": 0.001,
        "n_sensors": 4,
        "duration": 5e-6,
        "max_frequency": 5e5,
        "min_elems_x": 10,
        "min_elems_y": 2,
        "elems_per_wavelength": 2,
        "degree": 2,
        "run_postprocess": False,
        "output_dir": str(tmp_path),
        "structure_mode": "1_layer",
        "single_layer": {
            "thickness": 0.001,
            "rho": 2700.0,
            "E": 70e9,
            "nu": 0.33,
            "material_model": "isotropic",
        },
    }
    path, _ = run_simulation(cfg=config)
    with np.load(path) as data:
        assert np.isfinite(data["sensor_data_y"]).all()
        assert np.max(np.abs(data["sensor_data_y"])) > 0
        assert data["sensor_data_y"].dtype == np.float32


def test_solve_explicit_rk4_torch_cpu():
    """Verify PyTorch sparse solver engine on CPU when PyTorch is installed."""
    pytest.importorskip("torch")
    from guidedwavefem.solvers_gpu import solve_explicit_rk4_torch
    import scipy.sparse as sp

    n = 20
    K = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n), format="csr")
    K = -K
    C = sp.csr_matrix((n, n), dtype=np.float64)
    M_inv = np.ones(n, dtype=np.float64)
    f_spatial = np.zeros(n, dtype=np.float64)
    f_spatial[0] = 1.0
    S = sp.eye(n, format="csr")

    times, sx, sy = solve_explicit_rk4_torch(
        K_csr=(K.indptr, K.indices, K.data),
        C_csr=(C.indptr, C.indices, C.data),
        M_inv_np=M_inv,
        f_spatial_np=f_spatial,
        sensor_mat_csr=S,
        dt=1e-5,
        Nsteps=10,
        source_func=lambda t: 1.0,
        device_name="cpu",
        precision="double",
    )
    assert len(times) == 11
    assert sx.shape == (11, n // 2)
    assert np.isfinite(sx).all()

