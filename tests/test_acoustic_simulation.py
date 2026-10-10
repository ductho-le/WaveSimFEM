"""End-to-end acoustic field, boundary, and archive contracts."""

import numpy as np
import pytest

from wavesimfem import run_simulation
from wavesimfem.acoustics import acoustic_waveform


def small_config(tmp_path, solver="explicit", mixed=False):
    fluid = {"name": "water", "thickness": 0.001, "rho": 1000.0, "c": 1500.0}
    solid = {"name": "aluminium", "thickness": 0.001, "rho": 2700.0, "E": 70e9, "nu": 0.33}
    return {
        "solver_type": solver,
        "device": "cpu",
        "CFL": 0.6,
        "sensing_length": 0.01,
        "absorb_length": 0.003,
        "f_center": 1e5,
        "source_x": 0.004,
        "sensor_offset": 0.001,
        "sensor_pitch": 0.001,
        "n_sensors": 4,
        "duration": 2e-5,
        "nx": 12,
        "ny_per_region": [2, 2] if mixed else [2],
        "degree": 1,
        "run_postprocess": False,
        "output_dir": str(tmp_path),
        "structure_mode": "N_layers",
        "layers": [fluid, solid] if mixed else [fluid],
        "source_kind": "traction" if mixed else "acoustic_normal_acceleration",
        "source_amplitude": None if mixed else 1.0,
        "absorb_alpha": 0.0,
        "absorb_beta": 0.0,
    }


@pytest.mark.parametrize("solver", ["explicit", "implicit"])
@pytest.mark.parametrize(
    "observable", ["pressure", "particle_velocity", "displacement", "velocity"]
)
def test_acoustic_observable_archive(tmp_path, solver, observable):
    cfg = small_config(tmp_path, solver, mixed=True)
    cfg.update(
        sensor_observable=observable,
        sensor_y=0.0005 if observable in ("pressure", "particle_velocity") else 0.0015,
    )
    path, _ = run_simulation(cfg=cfg)
    key = {
        "pressure": "sensor_data_pressure",
        "particle_velocity": "sensor_data_vy",
        "displacement": "sensor_data_y",
        "velocity": "sensor_data_vy",
    }[observable]
    with np.load(path, allow_pickle=False) as data:
        assert int(data["schema_version"]) == 2
        assert str(data["sensor_observable"]) == observable
        assert np.isfinite(data[key]).all()
        assert np.max(np.abs(data[key])) > 0
        assert data[key].shape == (len(data["times"]), 4)
        assert data["times"][-1] == pytest.approx(cfg["duration"])
        if observable != "displacement":
            assert "sensor_data_y" not in data


@pytest.mark.parametrize("solver", ["explicit", "implicit"])
def test_prescribed_pressure_and_release(tmp_path, solver):
    cfg = small_config(tmp_path, solver)
    cfg.update(
        source_kind="acoustic_pressure",
        source_amplitude=100.0,
        source_type="tone_burst",
        n_cycles=2,
        acoustic_boundaries={"bottom": "pressure_release"},
        sensor_y=0.001,
    )
    path, _ = run_simulation(cfg=cfg)
    with np.load(path) as data:
        # FE interpolation on the driven face reproduces its nodal interpolant.
        assert np.isfinite(data["sensor_data_pressure"]).all()
        assert np.max(np.abs(data["sensor_data_pressure"])) > 0
        assert "c_T" not in data


def test_acoustic_waveform_rest_compatibility_and_derivatives():
    cfg = {"f_center": 1.0, "source_type": "ricker"}
    assert acoustic_waveform(0.0, cfg) == (0.0, 0.0, 0.0)
    for t in (0.1, 0.5, 1.2):
        h = 1e-5
        f, d, dd = acoustic_waveform(t, cfg)
        lo, hi = acoustic_waveform(t - h, cfg)[0], acoustic_waveform(t + h, cfg)[0]
        assert d == pytest.approx((hi - lo) / (2 * h), rel=2e-5)
        assert dd == pytest.approx((hi - 2 * f + lo) / h**2, rel=2e-5, abs=1e-7)


def test_physical_pressure_velocity_energy_is_conserved(tmp_path):
    from scipy.linalg import eigh
    from scipy.sparse import csr_matrix

    from wavesimfem.acoustic_time import CoupledNewmark
    from wavesimfem.acoustics import AcousticSampler, build_acoustic_problem
    from wavesimfem.config import parse_and_resolve_config

    cfg = parse_and_resolve_config(custom_cfg=small_config(tmp_path))
    problem = build_acoustic_problem(cfg)
    mass, stiffness = problem.operators.Mp, problem.operators.Kp
    m = csr_matrix(mass.getValuesCSR()[::-1], shape=mass.getSize()).toarray()
    k = csr_matrix(stiffness.getValuesCSR()[::-1], shape=stiffness.getSize()).toarray()
    eigenvalues, modes = eigh(k, m)
    mode = modes[:, np.flatnonzero(eigenvalues > 1)[0]]
    omega = np.sqrt(eigenvalues[np.flatnonzero(eigenvalues > 1)[0]])
    dt = 0.02 / omega
    stepper = CoupledNewmark(problem.operators, dt, initial={"p": mode})
    sampler = AcousticSampler(problem)
    try:
        sampler.update(stepper, 0.0, 0.0)
        pressure, kinetic = sampler.fluid_energy()
        assert pressure == pytest.approx(0.5 * mode @ m @ mode, rel=1e-12)
        assert kinetic == 0
        for i in range(100):
            stepper.step(i * dt, dt)
        sampler.update(stepper, 100 * dt, 0.0)
        final_p, final_k = sampler.fluid_energy()
        assert final_k > 0
        assert final_p + final_k == pytest.approx(pressure, rel=1e-9)
    finally:
        stepper.destroy()
        problem.destroy()


@pytest.mark.parametrize("observable", ["pressure", "particle_velocity"])
def test_acoustic_analysis_uses_physical_units(tmp_path, monkeypatch, observable):
    from wavesimfem.analysis import plot_full_analysis, plot_quick_overview
    from wavesimfem.characterisation import write_characterisation

    times = np.arange(256) * 1e-7
    x = np.arange(8) * 0.001
    signal = np.sin(2 * np.pi * 5e5 * times[:, None] - 1000 * x[None, :])
    fields = (
        {"sensor_data_pressure": signal}
        if observable == "pressure"
        else {"sensor_data_vx": signal, "sensor_data_vy": signal}
    )
    path = tmp_path / "sensor_data.npz"
    np.savez(
        path,
        times=times,
        sensor_x=x,
        f_center=5e5,
        c_L=1500.0,
        is_fluid_top=True,
        c_fluid=1500.0,
        sensor_observable=observable,
        sensor_units="Pa" if observable == "pressure" else "m/s",
        **fields,
    )
    labels = []

    def capture(figure, *args, **kwargs):
        labels.extend(a.get_ylabel() for a in figure.axes)

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", capture)
    plot_full_analysis(str(path), str(tmp_path), formats=["png"])
    plot_quick_overview(
        times, x, signal, 5e5, 0.0, str(tmp_path / "quick.png"), observable=observable
    )
    report = write_characterisation(path, tmp_path, formats=["png"])
    assert report["amplitude_units"] == ("Pa" if observable == "pressure" else "m/s")
    assert "peak_envelope_m" not in report["receivers"][0]
    assert not any("Displacement" in label for label in labels)
    if observable == "pressure":
        assert "particle_motion" not in report["figure_notes"]
        assert any("Pa" in label for label in labels)


@pytest.mark.parametrize("solver", ["explicit", "implicit"])
def test_pure_solid_velocity_uses_no_pressure_unknowns(tmp_path, solver):
    cfg = small_config(tmp_path, solver, mixed=True)
    cfg.update(layers=cfg["layers"][1:], ny_per_region=[2], sensor_observable="velocity")
    path, _ = run_simulation(cfg=cfg)
    with np.load(path) as data:
        assert not bool(data["has_acoustic"])
        assert "sensor_data_y" not in data
        assert str(data["sensor_units"]) == "m/s"
        assert np.max(np.abs(data["sensor_data_vy"])) > 0
        assert np.all(data["fluid_pressure_energy"] == 0)


@pytest.mark.parametrize("solver", ["explicit", "implicit"])
def test_field_aware_solid_path_matches_elastic_solver(tmp_path, solver):
    """Selecting a sensor region only changes acquisition, never the solid physics.

    Guards the shared traction footprint (Gaussian sigma = excitation_width / 3)
    and operators between the elastic and field-aware assembly paths.
    """
    cfg = small_config(tmp_path, solver, mixed=True)
    cfg.update(layers=cfg["layers"][1:], ny_per_region=[2], excitation_width=0.002)
    histories = []
    for name, extra in (("elastic", {}), ("field", {"sensor_region": "aluminium"})):
        path, _ = run_simulation(cfg=dict(cfg, output_dir=str(tmp_path / name), **extra))
        with np.load(path) as data:
            schema = int(data["schema_version"]) if "schema_version" in data else 1
            assert schema == (2 if name == "field" else 1)
            histories.append((data["times"].copy(), data["sensor_data_y"].copy()))
    (t_elastic, elastic), (t_field, field) = histories
    np.testing.assert_allclose(t_field, t_elastic, rtol=1e-12, atol=0)
    assert np.max(np.abs(elastic)) > 0
    assert np.linalg.norm(field - elastic) / np.linalg.norm(elastic) < 1e-6


def test_boundary_source_rejects_a_source_y_it_would_ignore(tmp_path):
    from wavesimfem.acoustics import build_acoustic_problem
    from wavesimfem.config import parse_and_resolve_config

    cfg = small_config(tmp_path, mixed=True)
    cfg.update(source_boundary="top", source_region="aluminium", source_y=0.0015)
    with pytest.raises(ValueError, match="does not apply to source_boundary 'top'"):
        build_acoustic_problem(parse_and_resolve_config(custom_cfg=cfg))


@pytest.mark.parametrize("precision", ["single", "double"])
def test_acoustic_array_precision_and_disk_history(tmp_path, precision):
    cfg = small_config(tmp_path)
    cfg.update(precision=precision, history_storage="disk")
    path, _ = run_simulation(cfg=cfg)
    with np.load(path) as data:
        assert np.isfinite(data["sensor_data_pressure"]).all()
        assert np.max(np.abs(data["sensor_data_pressure"])) > 0
    assert not list(tmp_path.glob(".wavesim-history-*"))


@pytest.mark.parametrize("solver", ["explicit", "implicit"])
def test_public_build_and_solve_acoustic_api(tmp_path, solver):
    from wavesimfem import (
        build_problem,
        parse_and_resolve_config,
        solve_explicit,
        solve_implicit,
    )

    problem = build_problem(parse_and_resolve_config(custom_cfg=small_config(tmp_path, solver)))
    try:
        result = (solve_explicit if solver == "explicit" else solve_implicit)(problem)
        assert np.max(np.abs(result.histories["sensor_data_pressure"])) > 0
        assert result.times[-1] == pytest.approx(problem.layout.duration)
        assert not (tmp_path / "sensor_data.npz").exists()
    finally:
        problem.destroy()


@pytest.mark.parametrize("model", ["anisotropic", "viscoelastic"])
def test_acoustic_coupling_preserves_general_solid_constitutive_models(tmp_path, model):
    cfg = small_config(tmp_path, mixed=True)
    young, nu = 70e9, 0.33
    mu = young / (2 * (1 + nu))
    lam = young * nu / ((1 + nu) * (1 - 2 * nu))
    tensor = [lam + 2 * mu, lam, 0.0, lam + 2 * mu, 0.0, mu]
    cfg["layers"][1] = {
        "name": "solid",
        "thickness": 0.001,
        "rho": 2700.0,
        "material_model": model,
        "C": tensor,
    }
    if model == "viscoelastic":
        cfg["layers"][1]["Cv"] = [value * 1e-8 for value in tensor]
    cfg.update(sensor_observable="pressure", sensor_y=0.0005)
    path, _ = run_simulation(cfg=cfg)
    with np.load(path) as data:
        assert np.isfinite(data["sensor_data_pressure"]).all()
        assert np.max(np.abs(data["sensor_data_pressure"])) > 0


def test_acoustic_device_policy_matches_elastic_solver(tmp_path):
    from wavesimfem.acoustics import build_acoustic_problem, solve_acoustic
    from wavesimfem.config import parse_and_resolve_config
    from wavesimfem.device import CPU_DEVICE

    cfg = small_config(tmp_path, "implicit")
    cfg.update(precision="single", duration=2e-6)
    problem = build_acoustic_problem(parse_and_resolve_config(custom_cfg=cfg))
    try:
        with pytest.raises(ValueError, match="single-process explicit array engine"):
            solve_acoustic(problem, device=CPU_DEVICE)
        # As in the elastic solver, the PETSc path ignores single precision.
        result = solve_acoustic(problem)
        assert result.runtime["backend"] == "petsc"
        assert result.runtime["precision"] == "double"
    finally:
        problem.destroy()


def test_separable_source_reproduces_physical_forcing(tmp_path):
    from wavesimfem.acoustics import _source_coefficients, build_acoustic_problem
    from wavesimfem.config import parse_and_resolve_config

    cfg = small_config(tmp_path)
    cfg.update(source_kind="acoustic_pressure", source_boundary="top", source_amplitude=2.0)
    cfg = parse_and_resolve_config(custom_cfg=cfg)
    problem = build_acoustic_problem(cfg)
    fs, fp = problem.operators.Ms.createVecRight(), problem.operators.Mp.createVecRight()
    try:
        t = 3.0e-6
        g, minus_ddg, minus_dg = _source_coefficients(cfg, t)
        assert (g, -minus_dg, -minus_ddg) == pytest.approx(
            tuple(2.0 * a for a in acoustic_waveform(t, cfg))
        )
        fs.set(0.0)
        fp.set(0.0)
        problem.forcing(t, fs, fp)
        assert np.isfinite(fp.array_r).all() and np.max(np.abs(fp.array_r)) > 0
        # A pure fluid has no solid block and a driven face lifts Mp, Dp, Kp.
        assert problem.forcing.solid == (None, None, None)
        assert problem.forcing.pressure[1] is not None
    finally:
        fs.destroy()
        fp.destroy()
        problem.destroy()
