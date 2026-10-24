"""Regression coverage for exact mesh controls, input validation, and arrival estimates."""

from __future__ import annotations

import stat
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import yaml

from wavesimfem import storage
from wavesimfem.cli import build_parser, main
from wavesimfem.config import generate_template_config, load_config, parse_and_resolve_config
from wavesimfem.materials import (
    a0_wavenumber,
    canonicalize_mode,
    compute_mesh_layout,
    normalize_structure,
    preflight_estimate,
    sampled_time_grid,
    structure_from_config,
)
from wavesimfem.parallel import limit_threads_per_rank
from wavesimfem.reporting import logger
from wavesimfem.solvers_gpu import _NumpyLikeBackend, integrate_explicit_rk4
from wavesimfem.storage import atomic_output
from wavesimfem.units import parse_quantity

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
PLATE = CONFIG_DIR / "01_plate_aluminum_lamb.yaml"
HALFSPACE = CONFIG_DIR / "03_halfspace_steel_rayleigh.yaml"
FLUID_ACOUSTIC = CONFIG_DIR / "07_fluid_acoustic.yaml"


@pytest.fixture(autouse=True)
def _restore_log_level():
    level = logger.level
    yield
    logger.setLevel(level)


def _variant(tmp_path: Path, source: Path, **sections) -> Path:
    """Copy an example configuration with section-level overrides."""
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    for section, values in sections.items():
        raw.setdefault(section, {}).update(values)
    path = tmp_path / source.name
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


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


def test_atomic_output_new_file_follows_umask(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "_UMASK", 0o022)
    path = tmp_path / "result.txt"
    with atomic_output(path) as stream:
        stream.write("data")
    assert stat.S_IMODE(path.stat().st_mode) == 0o644


def test_atomic_output_keeps_mode_of_replaced_file(tmp_path):
    path = tmp_path / "result.bin"
    path.write_bytes(b"old")
    path.chmod(0o640)
    with atomic_output(path, binary=True) as stream:
        stream.write(b"new")
    assert path.read_bytes() == b"new"
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_publish_results_and_benchmark_keep_existing_mode(tmp_path):
    import importlib.util

    from wavesimfem.solvers import ELASTIC_ARCHIVE_SCHEMA_VERSION, _publish_results

    spec = importlib.util.spec_from_file_location(
        "benchmark", CONFIG_DIR.parent / "tools" / "benchmark.py"
    )
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)

    cfg = parse_and_resolve_config(str(PLATE))
    npz_path = tmp_path / "sensor_data.npz"
    yaml_path = tmp_path / "simulation_config.yaml"
    npz_path.write_bytes(b"old")
    yaml_path.write_text("old: true\n", encoding="utf-8")
    npz_path.chmod(0o640)
    yaml_path.chmod(0o640)
    _publish_results(
        cfg,
        {"schema_version": ELASTIC_ARCHIVE_SCHEMA_VERSION, "times": np.zeros(2)},
        str(tmp_path),
    )
    assert stat.S_IMODE(npz_path.stat().st_mode) == 0o640
    assert stat.S_IMODE(yaml_path.stat().st_mode) == 0o640
    with np.load(npz_path, allow_pickle=False) as data:
        assert int(data["schema_version"]) == 1

    report_path = tmp_path / "benchmark.json"
    report_path.write_text("{}\n", encoding="utf-8")
    report_path.chmod(0o640)
    benchmark.write_report(report_path, {"ok": True})
    assert stat.S_IMODE(report_path.stat().st_mode) == 0o640


@pytest.mark.parametrize(
    "raw,expected",
    [(" halfspace ", "halfspace"), ("\t1 layer\n", "1_layer"), ("  N-Layers", "N_layers")],
)
def test_canonicalize_mode_ignores_surrounding_whitespace(raw, expected):
    assert canonicalize_mode(raw) == expected


@pytest.mark.parametrize("text", ["2 mhz", "2mHZ"])
def test_parse_quantity_rejects_ambiguous_milli_megahertz(text):
    with pytest.raises(ValueError, match="ambiguous"):
        parse_quantity(text, "frequency")


def test_parse_quantity_exact_millihertz_still_allowed():
    assert parse_quantity("5 mHz", "frequency") == pytest.approx(5e-3)


def test_cli_frequency_overrides_accept_units():
    args = build_parser().parse_args(["--min-frequency", "200kHz", "--max-frequency", "2 MHz"])
    assert args.min_frequency == pytest.approx(2e5)
    assert args.max_frequency == pytest.approx(2e6)


def test_cli_frequency_override_rejects_bad_unit(capsys):
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["--max-frequency", "2 rad/mm"])
    assert exc_info.value.code == 2
    assert "unknown frequency unit" in capsys.readouterr().err


def test_sampled_time_grid_keeps_max_frequency_alias_free():
    layout = SimpleNamespace(duration=20e-6, mesh_frequency=8e6)
    cfg = {"sensor_sample_rate": 5e6, "disp_f_max": None, "f_center": 1e6, "run_postprocess": False}
    dt, _n_steps, stride = sampled_time_grid(cfg, layout, 5e-9)
    assert 1.0 / (2.0 * stride * dt) >= 1.25 * layout.mesh_frequency * (1 - 1e-9)


def test_generate_template_refuses_to_overwrite(tmp_path):
    path = tmp_path / "mine.yaml"
    path.write_text("edited: true\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        generate_template_config(str(path))
    assert path.read_text(encoding="utf-8") == "edited: true\n"
    generate_template_config(str(path), overwrite=True)
    assert "solver:" in path.read_text(encoding="utf-8")


def test_cli_generate_config_force(tmp_path, capsys):
    path = tmp_path / "mine.yaml"
    path.write_text("edited: true\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc_info:
        main(["--generate-config", str(path)])
    assert exc_info.value.code == 2
    assert "already exists" in capsys.readouterr().err
    main(["--generate-config", str(path), "--force"])
    assert "solver:" in path.read_text(encoding="utf-8")


def test_lamb_modes_rejected_for_halfspace_before_solve(tmp_path):
    config = _variant(tmp_path, HALFSPACE, postprocess={"lamb_modes": True})
    with pytest.raises(ValueError, match=r"lamb_modes.*single isotropic"):
        parse_and_resolve_config(str(config))


def test_lamb_modes_rejected_for_too_few_receivers(tmp_path):
    config = _variant(tmp_path, PLATE, sensors={"n_sensors": 5})
    with pytest.raises(ValueError, match="at least 8 receivers"):
        parse_and_resolve_config(str(config))


def test_lamb_modes_rejected_for_velocity_observable(tmp_path):
    config = _variant(tmp_path, PLATE, sensors={"observable": "velocity"})
    with pytest.raises(ValueError, match="displacement sensors"):
        parse_and_resolve_config(str(config))


def test_wave_characterisation_validates_component_against_observable(tmp_path):
    bad_motion = _variant(
        tmp_path,
        FLUID_ACOUSTIC,
        postprocess={"wave_characterisation": {"enabled": True, "component": "x"}},
    )
    with pytest.raises(ValueError, match="Pressure archives do not contain"):
        parse_and_resolve_config(str(bad_motion))

    bad_pressure = _variant(
        tmp_path,
        PLATE,
        postprocess={"wave_characterisation": {"enabled": True, "component": "pressure"}},
    )
    with pytest.raises(ValueError, match="does not contain pressure sensors"):
        parse_and_resolve_config(str(bad_pressure))

    auto_pressure = _variant(
        tmp_path,
        FLUID_ACOUSTIC,
        postprocess={"wave_characterisation": {"enabled": True}},
    )
    resolved = parse_and_resolve_config(str(auto_pressure))
    assert resolved["wave_characterisation"]["component"] == "pressure"


def test_lamb_modes_ignored_without_postprocessing(tmp_path):
    config = _variant(
        tmp_path, HALFSPACE, postprocess={"lamb_modes": True, "run_postprocess": False}
    )
    cfg = parse_and_resolve_config(str(config))
    assert cfg["lamb_modes"] and not cfg["run_postprocess"]


def test_cli_rejects_missing_latex_before_simulation(tmp_path, monkeypatch, capsys):
    from wavesimfem import analysis, cli

    monkeypatch.setattr(analysis, "latex_available", lambda: False)

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("simulation started despite unusable plotting setup")

    monkeypatch.setattr(cli, "run_simulation", must_not_run)
    config = _variant(tmp_path, PLATE, output={"usetex": True})
    with pytest.raises(SystemExit) as exc_info:
        main(["-c", str(config), "--output", str(tmp_path / "out")])
    assert exc_info.value.code == 2
    assert "usetex=True requires LaTeX" in capsys.readouterr().err


def test_cli_dry_run_only_warns_about_missing_latex(tmp_path, monkeypatch, capsys):
    from wavesimfem import analysis

    monkeypatch.setattr(analysis, "latex_available", lambda: False)
    config = _variant(tmp_path, PLATE, output={"usetex": True})
    main(["-c", str(config), "--dry-run"])
    captured = capsys.readouterr()
    assert "Dry run complete" in captured.out
    assert "post-processing would fail" in captured.out + captured.err
