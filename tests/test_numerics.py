"""Independent numerical checks; run inside the wavesim environment."""

import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI
from scipy.linalg import expm

from wavesimfem import (
    PointSensors,
    dispersion_spectrum,
    dump_config,
    load_config,
    parse_and_resolve_config,
    rk4_step,
    validate_tensor,
)


def test_damped_rk4_convergence():
    exact = expm(np.array([[0.0, 1.0], [-4.0, -1.0]])) @ [1.0, 0.0]
    errors = []
    for n in (20, 40, 80):
        u, v = np.array([1.0]), np.array([0.0])
        for i in range(n):
            u, v = rk4_step(u, v, i / n, 1 / n, lambda u, v, t: -4 * u - v)
        errors.append(np.linalg.norm(np.array([u[0], v[0]]) - exact))
    assert np.all(np.log2(np.array(errors[:-1]) / errors[1:]) > 3.8)


def test_exact_point_interpolation():
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    space = fem.functionspace(domain, ("Lagrange", 2, (2,)))
    field = fem.Function(space)
    field.interpolate(lambda x: np.vstack([x[0] ** 2 + 2 * x[1], x[0] - x[1] ** 2]))
    field.x.scatter_forward()
    points = np.array([[0.137, 0.291, 0.0], [0.61, 0.43, 0.0], [1.0, 0.17, 0.0]])
    sensors = PointSensors(domain, points)
    values = sensors.sample(field)
    p = points[sensors.indices]
    np.testing.assert_allclose(values[:, 0], p[:, 0] ** 2 + 2 * p[:, 1], atol=1e-12)
    np.testing.assert_allclose(values[:, 1], p[:, 0] - p[:, 1] ** 2, atol=1e-12)
    with pytest.raises(ValueError, match="outside"):
        PointSensors(domain, np.array([[2.0, 0.0, 0.0]]))


@pytest.mark.parametrize(
    "element",
    [("Lagrange", 2, (2,)), ("Lagrange", 2), ("DG", 2, (2,))],
    ids=["vector", "scalar", "dg-vector"],
)
def test_exact_point_evaluation_matrix_matches_eval(element):
    """The cached sensor operator reproduces Function.eval, ghosts included."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 5, 4, cell_type=mesh.CellType.quadrilateral)
    space = fem.functionspace(domain, element)
    field = fem.Function(space)
    rng = np.random.default_rng(7 + domain.comm.rank)
    field.x.array[:] = rng.standard_normal(field.x.array.size)
    field.x.scatter_forward()
    # Points on shared edges and vertices exercise neighbouring-cell colouring.
    points = np.array([[x, y, 0.0] for x in np.linspace(0.0, 1.0, 11) for y in (0.0, 0.5, 0.83)])
    sensors = PointSensors(domain, points)
    expected = field.eval(sensors.points, sensors.cells).reshape(len(sensors.indices), -1)
    np.testing.assert_allclose(sensors.sample(field), expected, rtol=1e-13, atol=1e-13)
    field.x.array[:] *= -2.0
    np.testing.assert_allclose(sensors.sample(field), -2.0 * expected, rtol=1e-13, atol=1e-13)
    assert len(sensors._evaluators) == 1
    if space.num_sub_spaces:
        component = field.sub(1)
        np.testing.assert_allclose(
            sensors.sample(component)[:, 0], -2.0 * expected[:, 1], rtol=1e-13, atol=1e-13
        )
    from wavesimfem.numerics import point_evaluation_matrix

    with pytest.raises(ValueError, match="valid local cell"):
        point_evaluation_matrix(field, np.array([[0.1, 0.1, 0.0]]), np.array([-1]))


def test_point_sensor_operator_is_rank_local(monkeypatch):
    """Only sensor-owning ranks build the operator, so it must not allocate a
    Function (collective neighbourhood setup) and must restore the field."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 6, 6)
    field = fem.Function(fem.functionspace(domain, ("Lagrange", 2, (2,))))
    field.interpolate(lambda x: np.vstack([x[0] ** 2, x[0] * x[1]]))
    before = field.x.array.copy()
    # A single sensor: on several ranks, exactly one owns it.
    sensors = PointSensors(domain, np.array([[0.05, 0.07, 0.0]]))

    def forbidden(*args, **kwargs):
        raise AssertionError("sensor sampling must not allocate a Function")

    monkeypatch.setattr(fem, "Function", forbidden)
    values = sensors.sample(field)
    np.testing.assert_array_equal(field.x.array, before)
    if len(sensors.indices):
        np.testing.assert_allclose(values, [[0.05**2, 0.05 * 0.07]], atol=1e-13)
    assert domain.comm.allreduce(len(sensors.indices)) == 1


def test_forward_spectrum_physical_coordinates():
    t = np.arange(256) * 1e-7
    x = np.arange(64) * 0.0005
    frequency, wave = 625000.0, 2 * np.pi * 8 / (64 * 0.0005)
    signal = np.cos(2 * np.pi * frequency * t[:, None] - wave * x[None, :])
    f, k, power = dispersion_spectrum(signal, t, x)
    fi, ki = np.unravel_index(power.argmax(), power.shape)
    assert abs(f[fi] - frequency) < f[1] - f[0]  # Hz
    assert abs(k[ki] - wave) < k[1] - k[0]  # rad/m
    assert k[-1] == pytest.approx(np.pi / 0.0005)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_point_sensor_gather(dtype):
    """Gatherv reassembles per-rank owned columns in global order (any rank count)."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    points = np.array([[x, 0.37, 0.0] for x in np.linspace(0.05, 0.95, 7)])
    sensors = PointSensors(domain, points)
    n_times = 5
    t = np.arange(n_times)[:, None]
    local = (100 * t + sensors.indices[None, :]).astype(dtype)  # encodes (time, global idx)
    gathered = sensors.gather(local)
    if domain.comm.rank == 0:
        expected = (100 * t + np.arange(len(points))[None, :]).astype(dtype)
        assert gathered.dtype == dtype
        np.testing.assert_array_equal(gathered, expected)
    else:
        assert gathered is None


def test_csr_row_abs_sums_handles_empty_rows():
    from wavesimfem.numerics import csr_row_abs_sums

    indptr = np.array([0, 2, 2, 3, 3])
    values = np.array([-1.0, 2.0, -4.0])
    np.testing.assert_array_equal(csr_row_abs_sums(indptr, values), [3.0, 0.0, 4.0, 0.0])


@pytest.mark.parametrize("duration,dt", [(0.0, 1e-9), (1e-6, 0.0), (np.inf, 1e-9), (1e-6, np.nan)])
def test_time_grid_rejects_invalid(duration, dt):
    from wavesimfem.numerics import time_grid

    with pytest.raises(ValueError):
        time_grid(duration, dt)


def test_time_grid_fits_duration_exactly():
    from wavesimfem.numerics import time_grid

    dt, steps = time_grid(1e-5, 3e-8)
    assert steps * dt == pytest.approx(1e-5, rel=1e-15)
    assert dt <= 3e-8


def test_material_energy():
    with pytest.raises(ValueError, match="positive"):
        validate_tensor([1, 2, 0, 1, 0, 1], "C")
    np.testing.assert_array_equal(validate_tensor([0] * 6, "Cv", positive=False), np.zeros((3, 3)))


def test_boolean_provenance(tmp_path):
    path = str(tmp_path / "config.yaml")
    dump_config({"enabled": True, "disabled": False}, path)
    result = load_config(path)
    assert result["enabled"] is True and result["disabled"] is False


def test_plot_limits_do_not_stretch_fft(tmp_path, monkeypatch):
    import wavesimfem as ws

    t, x = np.arange(256) * 1e-7, np.arange(16) * 0.001
    signal = np.sin(2 * np.pi * 5e5 * t[:, None] - 1000 * x[None, :])
    path = tmp_path / "signal.npz"
    np.savez(
        path,
        times=t,
        sensor_x=x,
        sensor_data_y=signal,
        sensor_data_x=signal,
        f_center=5e5,
        c_L=6000.0,
        c_T=3000.0,
    )
    extents = []

    def inspect_save(fig, *args, **kwargs):
        for ax in fig.axes:
            extents.extend(image.get_extent() for image in ax.images)

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", inspect_save)
    ws.plot_full_analysis(str(path), str(tmp_path), k_max="10 rad/mm")
    assert extents[1][1] < 3.3  # Nyquist ~pi rad/mm, NOT the requested 10


def test_shipped_examples_cover_every_structure_mode():
    """configs/ holds valid examples covering every supported structure mode."""
    from collections import Counter
    from pathlib import Path

    from wavesimfem.materials import STRUCTURE_MODES, structure_from_config

    paths = sorted(
        p
        for p in (Path(__file__).resolve().parents[1] / "configs").glob("*")
        if p.suffix in (".yaml", ".yml", ".json", ".toml")
    )
    modes = Counter()
    for path in paths:
        cfg = parse_and_resolve_config(str(path))
        summary = structure_from_config(cfg)
        assert summary["H_domain"] > 0
        modes[cfg["structure_mode"]] += 1
    assert set(STRUCTURE_MODES).issubset(modes), (
        f"Missing modes: {set(STRUCTURE_MODES) - set(modes)}"
    )


def test_assembled_operator_bound():
    import ufl
    from dolfinx.fem.petsc import assemble_matrix
    from scipy.linalg import eigvalsh

    from wavesimfem.numerics import operator_bound

    domain = mesh.create_unit_square(MPI.COMM_SELF, 3, 3)
    space = fem.functionspace(domain, ("Lagrange", 1))
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    stiffness = assemble_matrix(fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx))
    mass = assemble_matrix(fem.form(u * v * ufl.dx))
    stiffness.assemble()
    mass.assemble()
    kd = stiffness.convert("dense")
    md = mass.convert("dense")
    lumped = md.getDenseArray().sum(axis=1)
    exact = eigvalsh(kd.getDenseArray(), np.diag(lumped))[-1]
    assert operator_bound(stiffness, lumped) >= exact * (1 - 1e-12)
    for matrix in (kd, md, stiffness, mass):
        matrix.destroy()


def test_package_namespace_exports():
    from importlib.metadata import version

    import wavesimfem as ws

    assert ws.__version__ == version("wavesimfem")
    assert set(ws.__all__) <= set(dir(ws))
    assert set(ws.__all__) == {"__version__", *ws._EXPORTS}
    with pytest.raises(AttributeError):
        ws.does_not_exist
    for name in (
        "run_simulation",
        "plot_full_analysis",
        "load_config",
        "dump_config",
        "parse_and_resolve_config",
        "preflight_estimate",
        "rk4_step",
        "operator_bound",
        "time_grid",
        "PointSensors",
        "validate_tensor",
        "dispersion_spectrum",
    ):
        assert hasattr(ws, name) and callable(getattr(ws, name))


@pytest.mark.parametrize(
    "indptr,values,expected",
    [
        ([0], [], []),
        ([0, 0, 0], [], [0.0, 0.0]),
        ([0, 0, 2, 2, 3, 3], [-1.0, 2.5, -4.0], [0.0, 3.5, 0.0, 4.0, 0.0]),
    ],
)
def test_csr_row_abs_sums_edge_cases(indptr, values, expected):
    from wavesimfem.numerics import csr_row_abs_sums

    np.testing.assert_allclose(
        csr_row_abs_sums(np.asarray(indptr), np.asarray(values, dtype=float)),
        np.asarray(expected, dtype=float),
    )


def test_drop_zero_entries_handles_empty_rows():
    from petsc4py import PETSc

    from wavesimfem.solvers import _drop_zero_entries

    indptr = np.array([0, 0, 2, 2, 4, 4], dtype=PETSc.IntType)
    indices = np.array([0, 1, 1, 2], dtype=PETSc.IntType)
    values = np.array([0.0, -2.0, 0.0, 0.0], dtype=PETSc.ScalarType)
    mat = PETSc.Mat().createAIJ(size=(5, 3), csr=(indptr, indices, values), comm=PETSc.COMM_SELF)
    mat.assemble()
    filtered = _drop_zero_entries(mat)
    try:
        out_indptr, out_indices, out_values = filtered.getValuesCSR()
        assert list(out_indptr) == [0, 0, 1, 1, 1, 1]
        assert list(out_indices) == [1]
        np.testing.assert_allclose(out_values, [-2.0])
        np.testing.assert_allclose(
            filtered.convert("dense", out=PETSc.Mat()).getDenseArray(),
            mat.convert("dense", out=PETSc.Mat()).getDenseArray(),
        )
    finally:
        filtered.destroy()
        mat.destroy()


def test_petsc_ksp_is_scale_invariant_for_tiny_rhs():
    from petsc4py import PETSc

    from wavesimfem.solvers import ITERATIVE_DOF_THRESHOLD, _configure_ksp, _fallback_bjacobi

    n = 6
    mat = PETSc.Mat().createAIJ(size=(n, n), comm=PETSc.COMM_SELF)
    for i in range(n):
        mat.setValue(i, i, 4.0)
        if i > 0:
            mat.setValue(i, i - 1, -1.0)
        if i + 1 < n:
            mat.setValue(i, i + 1, -1.0)
    mat.assemble()
    rhs = mat.createVecLeft()
    sol = mat.createVecRight()
    ksp = PETSc.KSP().create(PETSc.COMM_SELF)
    try:
        ksp.setOperators(mat)
        _configure_ksp(ksp, ITERATIVE_DOF_THRESHOLD)
        assert ksp.getTolerances()[1] == 0.0
        _fallback_bjacobi(ksp, ITERATIVE_DOF_THRESHOLD)
        # Tiny RHS below 1e-14 must still solve accurately (atol=0.0).
        rhs.set(1e-18)
        sol.set(0.0)
        ksp.solve(rhs, sol)
        assert ksp.getConvergedReason() > 0
        expected = np.linalg.solve(
            mat.convert("dense", out=PETSc.Mat()).getDenseArray(), np.full(n, 1e-18)
        )
        np.testing.assert_allclose(sol.array_r, expected, rtol=1e-8, atol=0.0)
        # Zero RHS converges cleanly to zero.
        rhs.set(0.0)
        sol.set(1.0)
        ksp.solve(rhs, sol)
        assert ksp.getConvergedReason() > 0
        np.testing.assert_array_equal(sol.array_r, np.zeros(n))
        assert ksp.getTolerances()[1] == 0.0
    finally:
        ksp.destroy()
        rhs.destroy()
        sol.destroy()
        mat.destroy()


def test_history_store_close_cleans_scratch_even_if_mmap_close_fails(tmp_path):
    import types
    from pathlib import Path

    from wavesimfem.storage import HistoryStore

    store = HistoryStore("disk", tmp_path)
    arr = store.zeros((4, 3))
    assert arr.shape == (4, 3)
    scratch_path = Path(store._scratch.name)
    assert scratch_path.exists()

    class BrokenMmap:
        def close(self) -> None:
            raise OSError("close failed")

    store._maps.append(types.SimpleNamespace(_mmap=BrokenMmap()))
    with pytest.raises(OSError, match="close failed"):
        store.close()
    assert not scratch_path.exists()
    assert store._scratch is None


def test_describe_thread_state_does_not_report_single_rank_on_multi_rank(monkeypatch):
    from wavesimfem import parallel

    monkeypatch.setattr(
        parallel,
        "_policy",
        parallel.ThreadPolicy(
            ranks_per_node=1,
            threads_per_rank=None,
            applied=False,
            reason="single rank",
        ),
    )
    assert "single rank" not in parallel.describe_thread_state(4)
    assert "single rank" in parallel.describe_thread_state(1)
