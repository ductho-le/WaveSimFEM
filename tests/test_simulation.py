import numpy as np
import pytest

from wavesimfem import load_config, run_simulation


@pytest.mark.parametrize("solver", ["implicit", "explicit"])
def test_small_plate_without_plotting(tmp_path, solver):
    config = {
        "solver_type": solver,
        "sensing_length": 0.01,
        "absorb_length": 0.003,
        "f_center": 5e5,
        "source_x": 0.004,
        "sensor_offset": 0.001,
        "sensor_pitch": 0.001,
        "n_sensors": 4,
        "duration": 5e-6,
        "max_frequency": 5e5,
        "nx": 10,
        "ny_per_region": [2],
        "CFL": 0.8,
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
        assert np.max(abs(data["sensor_data_y"])) > 0
        assert data["times"][-1] == pytest.approx(5e-6)
    saved = load_config(str(tmp_path / "simulation_config.yaml"))
    assert saved["runtime"]["nx"] == 10
    assert saved["runtime"]["ny_per_region"] == [2]
    assert saved["absorb_alpha"] == pytest.approx(5e6)  # 1e7 scaled to f0 = 500 kHz
    assert saved["runtime"]["dt"] > 0
    assert not (tmp_path / "quick_results.png").exists()


def test_user_cfl_controls_assembled_step_without_changing_mesh(tmp_path):
    """CFL scales the assembled limit; only endpoint/sample rounding reduces it."""
    cfg = {
        "solver": {"type": "explicit", "device": "cpu", "duration": 5e-6},
        "geometry": {"sensing_length": 0.01, "absorb_length": 0.003},
        "structure": {
            "mode": "1_layer",
            "single_layer": {"thickness": 0.001, "rho": 2700.0, "E": 70e9, "nu": 0.33},
        },
        "excitation": {"f_center": 5e5, "source_x": 0.004},
        "mesh": {"nx": 13, "ny_per_region": [3], "degree": 2},
        "sensors": {"sensor_offset": 0.001, "sensor_pitch": 0.001, "n_sensors": 4},
        "postprocess": {"run_postprocess": False},
    }
    runtimes = []
    for cfl in (0.2, 0.8):
        output = tmp_path / str(cfl)
        cfg["solver"]["CFL"] = cfl
        cfg["output"] = {"output_dir": str(output)}
        path, _ = run_simulation(cfg=cfg)
        runtime = load_config(str(output / "simulation_config.yaml"))["runtime"]
        requested_dt = cfl * runtime["dt_limit"]
        assert runtime["nx"] == 13 and runtime["ny_per_region"] == [3]
        assert runtime["dt"] <= requested_dt * (1 + 1e-12)
        assert runtime["steps"] % runtime["sample_stride"] == 0
        # A whole sample interval less would not reach the requested duration.
        assert (runtime["steps"] - runtime["sample_stride"]) * requested_dt < 5e-6
        assert runtime["steps"] * runtime["dt"] == pytest.approx(5e-6)
        with np.load(path) as data:
            assert data["times"][-1] == pytest.approx(5e-6)
        runtimes.append(runtime)
    assert runtimes[0]["dt_limit"] == pytest.approx(runtimes[1]["dt_limit"], rel=1e-12)
    assert runtimes[0]["ndofs"] == runtimes[1]["ndofs"]
    assert runtimes[1]["dt"] > 3.9 * runtimes[0]["dt"]


def test_tone_burst_simulation_records_source(tmp_path):
    config = {
        "solver_type": "explicit",
        "device": "cpu",
        "sensing_length": 0.01,
        "absorb_length": 0.003,
        "f_center": 5e5,
        "source_type": "tone_burst",
        "n_cycles": 3,
        "source_x": 0.004,
        "sensor_offset": 0.001,
        "sensor_pitch": 0.001,
        "n_sensors": 4,
        "duration": 8e-6,
        "max_frequency": 5e5,
        "nx": 10,
        "ny_per_region": [2],
        "CFL": 0.8,
        "degree": 1,
        "run_postprocess": False,
        "output_dir": str(tmp_path),
        "structure_mode": "1_layer",
        "single_layer": {"thickness": 0.001, "rho": 2700.0, "E": 70e9, "nu": 0.33},
    }
    path, _ = run_simulation(cfg=config)
    with np.load(path) as data:
        assert str(data["source_type"]) == "tone_burst"
        assert int(data["n_cycles"]) == 3
        assert np.isfinite(data["sensor_data_y"]).all()
        assert np.max(abs(data["sensor_data_y"])) > 0
        assert float(data["disp_f_max"]) == pytest.approx(1e6)


@pytest.mark.parametrize("beta,warned", [(None, False), (1e-6, True)])
def test_damping_limited_step_warning(tmp_path, capsys, beta, warned):
    config = {
        "solver_type": "explicit",
        "device": "cpu",
        "sensing_length": 0.01,
        "absorb_length": 0.003,
        "absorb_beta": beta,
        "f_center": 5e5,
        "source_x": 0.004,
        "sensor_offset": 0.001,
        "sensor_pitch": 0.001,
        "n_sensors": 4,
        "duration": 2e-7,
        "max_frequency": 5e5,
        "nx": 10,
        "ny_per_region": [2],
        "CFL": 0.8,
        "degree": 1,
        "run_postprocess": False,
        "output_dir": str(tmp_path),
        "structure_mode": "1_layer",
        "single_layer": {"thickness": 0.001, "rho": 2700.0, "E": 70e9, "nu": 0.33},
    }
    run_simulation(cfg=config)
    assert ("Damping limits the explicit step" in capsys.readouterr().out) is warned


@pytest.mark.parametrize("precision", ["double", "single"])
def test_explicit_simulation_device_and_precision(tmp_path, precision):
    out_dir = tmp_path / f"explicit_{precision}"
    config = {
        "solver_type": "explicit",
        "device": "cpu",
        "precision": precision,
        "sensing_length": 0.01,
        "absorb_length": 0.003,
        "f_center": 5e5,
        "source_x": 0.004,
        "sensor_offset": 0.001,
        "sensor_pitch": 0.001,
        "n_sensors": 4,
        "duration": 5e-6,
        "max_frequency": 5e5,
        "nx": 10,
        "ny_per_region": [2],
        "CFL": 0.8,
        "degree": 1,
        "run_postprocess": False,
        "output_dir": str(out_dir),
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
        assert np.max(abs(data["sensor_data_y"])) > 0
        assert data["times"][-1] == pytest.approx(5e-6)
