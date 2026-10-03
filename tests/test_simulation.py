import numpy as np
import pytest

from guidedwavefem import load_config, run_simulation


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
        assert np.max(abs(data["sensor_data_y"])) > 0
        assert data["times"][-1] == pytest.approx(5e-6)
    saved = load_config(str(tmp_path / "simulation_config.yaml"))
    assert saved["runtime"]["ny_per_region"] == [2]
    assert saved["absorb_alpha"] == pytest.approx(5e6)  # 1e7 scaled to f0 = 500 kHz
    assert saved["runtime"]["dt"] > 0
    assert not (tmp_path / "quick_results.png").exists()


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
        "min_elems_x": 10,
        "min_elems_y": 2,
        "elems_per_wavelength": 2,
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
        "min_elems_x": 10,
        "min_elems_y": 2,
        "elems_per_wavelength": 2,
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
        "min_elems_x": 10,
        "min_elems_y": 2,
        "elems_per_wavelength": 2,
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
