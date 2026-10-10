"""Regression coverage for exact mesh controls, input validation, and arrival estimates."""

import numpy as np
import pytest
import scipy.sparse as sp

from wavesimfem.config import load_config, parse_and_resolve_config
from wavesimfem.materials import (
    a0_wavenumber,
    compute_mesh_layout,
    normalize_structure,
    preflight_estimate,
    structure_from_config,
)
from wavesimfem.parallel import limit_threads_per_rank
from wavesimfem.solvers_gpu import _NumpyLikeBackend, integrate_explicit_rk4
from wavesimfem.units import parse_quantity


def plate_config(**updates):
    cfg = dict(
        sensing_length=0.12,
        absorb_length=0.02,
        f_center=1e5,
        structure_mode="1_layer",
        single_layer=dict(thickness=1e-5, rho=2700.0, E=70e9, nu=0.33),
        nx=1683,
        ny_per_region=[4],
        CFL=0.8,
    )
    cfg.update(updates)
    return parse_and_resolve_config(custom_cfg=cfg)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_mesh_counts_are_exact_and_independent_of_polynomial_degree(degree):
    cfg = plate_config(degree=degree, nx=900, ny_per_region=[6], CFL=0.5)
    layout = compute_mesh_layout(cfg, structure_from_config(cfg))
    assert layout.nx == 900
    assert layout.ny_per_region == [6]
    assert layout.h_x == pytest.approx(0.16 / 900)
    assert layout.h_y_per_region == pytest.approx([1e-5 / 6])
    assert cfg["CFL"] == 0.5


@pytest.mark.parametrize(
    "updates",
    [
        dict(f_center=2e5),
        dict(max_frequency=9e5),
        dict(min_frequency=1e3),
        dict(min_wavelength=1e-5),
        dict(single_layer=dict(thickness=2e-5, rho=4000, E=30e9, nu=0.25)),
        dict(
            single_layer=dict(
                thickness=1e-5,
                rho=1600,
                material_model="anisotropic",
                C=[150e9, 8e9, 0, 20e9, 0, 5e9],
            )
        ),
    ],
)
def test_physics_and_band_controls_cannot_refine_the_configured_mesh(updates):
    cfg = plate_config(**updates)
    layout = compute_mesh_layout(cfg, structure_from_config(cfg))
    assert layout.nx == 1683
    assert layout.ny_per_region == [4]


def test_a0_dispersion_arrival_and_mesh_refinement():
    cfg = plate_config()
    summary = structure_from_config(cfg)
    region = summary["all_regions"][0]
    args = (1e-5, region["c_L"], region["c_T"])
    k = a0_wavenumber(1e5, *args)
    assert k == pytest.approx(6354.674976, rel=1e-7)
    wavelength = 2 * np.pi / a0_wavenumber(3e5, *args)
    assert wavelength == pytest.approx(0.000570434849, rel=1e-7)
    layout = compute_mesh_layout(cfg, summary)
    assert (layout.nx, layout.ny_per_region) == (1683, [4])
    assert cfg["sensor_pitch"] < wavelength / 2
    assert cfg["disp_k_max"] > k
    # Independent Kirchhoff–Love group speed at the lower retained edge.
    bending = np.sqrt(70e9 * 1e-5**2 / (12 * 2700 * (1 - 0.33**2)))
    group = 2 * np.sqrt(bending * 2 * np.pi * 2e4)
    assert layout.duration >= 0.12 / group
    fine = compute_mesh_layout(dict(cfg, nx=2 * cfg["nx"], ny_per_region=[8]), summary)
    assert fine.plate_length / fine.nx <= layout.plate_length / layout.nx * 0.501


def test_conservative_controls():
    cfg = plate_config(min_wavelength=1e-4, min_group_velocity=10.0)
    layout = compute_mesh_layout(cfg, structure_from_config(cfg))
    assert (layout.nx, layout.ny_per_region) == (1683, [4])
    assert layout.duration >= 0.0144
    assert cfg["sensor_pitch"] <= 1.25e-5


@pytest.mark.parametrize(
    "text,suffix",
    [
        ("solver:\n  CFL: 0.5\n  CFL: 1.0\n", ".yaml"),
        ("structure:\n  single_layer:\n    E: 70e9\n    E: 7e9\n", ".yaml"),
        ('{"structure":{"single_layer":{"E":70e9,"E":7e9}}}', ".json"),
    ],
)
def test_duplicate_keys(tmp_path, text, suffix):
    path = tmp_path / ("duplicate" + suffix)
    path.write_text(text)
    with pytest.raises(ValueError, match="Duplicate"):
        load_config(str(path))


def test_malformed_yaml_is_configuration_error(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("layers: [")
    with pytest.raises(ValueError, match="Invalid YAML"):
        load_config(str(path))


def test_prefix_case():
    assert parse_quantity("1 mHz", "frequency") == 0.001
    assert parse_quantity("1 MHz", "frequency") == 1e6
    with pytest.raises(ValueError):
        parse_quantity("1 rad/Mm", "wavenumber")


@pytest.mark.parametrize("extra", [dict(K=-1), dict(c_sound=1600), dict(E=-1, nu=2)])
def test_fluid_conflicts(extra):
    with pytest.raises(ValueError):
        normalize_structure(
            mode="1_layer",
            single_layer_input=dict(
                thickness=0.001, rho=1000, material_model="fluid", c=1500, **extra
            ),
        )


@pytest.mark.parametrize(
    "updates",
    [
        dict(stiffness=(np.array([0, 1, 1]), np.array([2]), np.ones(1))),
        dict(sensor_matrix=sp.csr_matrix(np.ones((3, 2)))),
        dict(sensor_matrix=sp.csr_matrix([[np.nan, 0], [0, 1]])),
        dict(inv_mass=np.array([1.0, -1.0])),
        dict(inv_mass=np.ones((2, 1))),
        dict(load_shape=np.ones((2, 1))),
        dict(n_steps=2.5),
        dict(save_every=0),
        dict(report_interval=1.5),
        dict(sample_every=True),
        dict(history_chunk_bytes=0),
    ],
)
def test_rk4_rejects_malformed_inputs(updates):
    kwargs = dict(
        stiffness=sp.eye(2, format="csr"),
        damping=sp.csr_matrix((2, 2)),
        inv_mass=np.ones(2),
        load_shape=np.ones(2),
        sensor_matrix=sp.eye(2, format="csr"),
        dt=0.01,
        n_steps=2,
        source=lambda t: 1.0,
        backend=_NumpyLikeBackend(np.dtype("float64")),
    )
    kwargs.update(updates)
    with pytest.raises(ValueError):
        integrate_explicit_rk4(**kwargs)


@pytest.mark.parametrize("tasks,node,expected", [("1,8", "1", 8), ("2(x3),8,4(x2)", "4", 4)])
def test_slurm_current_node(tasks, node, expected):
    policy = limit_threads_per_rank(
        {"SLURM_STEP_TASKS_PER_NODE": tasks, "SLURM_NODEID": node}, cpus=8
    )
    assert policy.ranks_per_node == expected
    assert policy.applied


def test_preflight_includes_histories():
    cfg = plate_config(
        duration=0.2,
        dt=1e-6,
        sensor_sample_rate=1e6,
        n_sensors=1001,
        sensor_pitch=1e-5,
        solver_type="implicit",
    )
    estimate = preflight_estimate(cfg, rank=1)
    assert estimate["history_mem_mb"] >= 2 * 1001 * 200001 * 8 / 1024**2
    assert estimate["total_mem_est_mb"] > 3 * estimate["history_mem_mb"]


def test_streamed_history_matches_retained():
    kwargs = dict(
        stiffness=sp.eye(2, format="csr"),
        damping=sp.csr_matrix((2, 2)),
        inv_mass=np.ones(2),
        load_shape=np.ones(2),
        sensor_matrix=sp.eye(2, format="csr"),
        dt=0.01,
        n_steps=20,
        sample_every=2,
        history_chunk_bytes=32,
        source=lambda t: 1.0,
        backend=_NumpyLikeBackend(np.dtype("float64")),
    )
    expected = integrate_explicit_rk4(**kwargs)
    chunks, times = [], []

    def collect(t, values):
        times.append(t)
        chunks.append(values)

    result = integrate_explicit_rk4(**kwargs, history_callback=collect, retain_history=False)
    assert result.sensor_data_x.shape == (0, 1)
    np.testing.assert_array_equal(np.concatenate(times), expected.times)
    np.testing.assert_allclose(np.concatenate(chunks)[:, 0::2], expected.sensor_data_x)
    np.testing.assert_allclose(np.concatenate(chunks)[:, 1::2], expected.sensor_data_y)


def test_thin_coating_load_resultant():
    pytest.importorskip("dolfinx")
    import ufl
    from dolfinx import fem
    from mpi4py import MPI
    from scipy.special import erf

    from wavesimfem.solvers import build_problem

    solid = dict(rho=2700.0, E=70e9, nu=0.33)
    cfg = plate_config(
        structure_mode="plate_in_halfspaces",
        single_layer=None,
        layers=[dict(solid, thickness=0.001), dict(solid, thickness=1e-9)],
        bottom_halfspace=dict(solid, thickness=0.001),
        top_halfspace=dict(solid, thickness=1e-9),
        sensing_length=0.01,
        absorb_length=0.002,
        nx=32,
        ny_per_region=[1, 1, 1, 1],
        max_frequency=100.0,
        n_sensors=3,
        sensor_pitch=0.001,
        sensor_offset=0.001,
        source_x=0.006,
        excitation_width=0.001,
        min_frequency=10.0,
    )
    problem = build_problem(cfg)
    problem.traction_amplitude.value = 1.0
    unity = fem.Function(problem.V)
    unity.interpolate(lambda x: np.vstack((np.zeros(x.shape[1]), np.ones(x.shape[1]))))
    resultant = fem.assemble_scalar(fem.form(ufl.action(problem.load_form, unity)))
    resultant = MPI.COMM_WORLD.allreduce(resultant, op=MPI.SUM)
    sigma = cfg["excitation_width"] / 3
    length = problem.layout.plate_length
    expected = (
        sigma
        * np.sqrt(np.pi / 2)
        * (
            erf((length - cfg["source_x"]) / (np.sqrt(2) * sigma))
            + erf(cfg["source_x"] / (np.sqrt(2) * sigma))
        )
    )
    assert resultant == pytest.approx(expected, rel=2e-3)
