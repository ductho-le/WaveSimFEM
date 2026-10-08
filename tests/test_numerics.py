"""Independent numerical checks; run inside the gwfem environment."""

import numpy as np
import pytest
from dolfinx import fem, mesh
from mpi4py import MPI
from scipy.linalg import expm

from guidedwavefem import (
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
    from guidedwavefem.numerics import csr_row_abs_sums

    indptr = np.array([0, 2, 2, 3, 3])
    values = np.array([-1.0, 2.0, -4.0])
    np.testing.assert_array_equal(csr_row_abs_sums(indptr, values), [3.0, 0.0, 4.0, 0.0])


@pytest.mark.parametrize("duration,dt", [(0.0, 1e-9), (1e-6, 0.0), (np.inf, 1e-9), (1e-6, np.nan)])
def test_time_grid_rejects_invalid(duration, dt):
    from guidedwavefem.numerics import time_grid

    with pytest.raises(ValueError):
        time_grid(duration, dt)


def test_time_grid_fits_duration_exactly():
    from guidedwavefem.numerics import time_grid

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
    import guidedwavefem as gw

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
    gw.plot_full_analysis(str(path), str(tmp_path), k_max="10 rad/mm")
    assert extents[1][1] < 3.3  # Nyquist ~pi rad/mm, NOT the requested 10


def test_shipped_examples_cover_every_structure_mode():
    """configs/ holds valid examples covering every supported structure mode."""
    from collections import Counter
    from pathlib import Path

    from guidedwavefem.materials import STRUCTURE_MODES, structure_from_config

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

    from guidedwavefem.numerics import operator_bound

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

    import guidedwavefem as gw

    assert gw.__version__ == version("guidedwavefem")
    assert set(gw.__all__) <= set(dir(gw))
    assert set(gw.__all__) == {"__version__", *gw._EXPORTS}
    with pytest.raises(AttributeError):
        gw.does_not_exist
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
        assert hasattr(gw, name) and callable(getattr(gw, name))
