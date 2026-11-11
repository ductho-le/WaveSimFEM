"""Resolution studies refine configured counts without changing the model."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

TOOL = Path(__file__).resolve().parents[1] / "tools" / "compare_resolution.py"
SPEC = importlib.util.spec_from_file_location("compare_resolution", TOOL)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


@pytest.mark.parametrize(
    "solver,dt,control,value",
    [
        ("explicit", None, "CFL", 0.3),
        ("implicit", None, "steps_per_period", 70),
        ("explicit", 1e-8, "dt", 5e-9),
    ],
)
def test_refinement_preserves_user_model_and_step_control(solver, dt, control, value):
    cfg = dict(
        nx=11,
        ny_per_region=[2, 3],
        CFL=0.6,
        steps_per_period=35,
        solver_type=solver,
        dt=dt,
        sensor_pitch=0.001,
    )
    key, variants = comparison.resolution_variants(cfg, 1.5, 0.5)
    assert key == control
    assert variants[0]["nx"] == 17
    assert variants[0]["ny_per_region"] == [3, 5]
    assert variants[0][key] == variants[1][key] == value
    assert variants[1]["nx"] == variants[2]["nx"] == 11
    assert variants[1]["ny_per_region"] == variants[2]["ny_per_region"] == [2, 3]
    assert variants[2] == cfg
    assert all(v["sensor_pitch"] == cfg["sensor_pitch"] for v in variants)


@pytest.mark.parametrize("changed", ["nx", "ny_per_region", "CFL", "duration"])
def test_resume_rejects_changed_mesh_or_time_inputs(changed):
    requested = dict(nx=11, ny_per_region=[2], CFL=0.6, duration=1e-5, sensor_y=None)
    saved = {**requested, "sensor_y": 0.001, "runtime": {"dt": 1e-8}}
    comparison.check_resumed_configuration(saved, requested)
    saved[changed] = None
    with pytest.raises(ValueError, match=changed):
        comparison.check_resumed_configuration(saved, requested)


@pytest.mark.parametrize("solver", ["explicit", "implicit"])
def test_resolution_tool_runs_and_resumes_exact_meshes(tmp_path, solver):
    cfg = dict(
        solver_type=solver,
        device="cpu",
        CFL=0.6,
        steps_per_period=35,
        sensing_length=0.01,
        absorb_length=0.003,
        f_center=5e5,
        source_x=0.004,
        sensor_offset=0.001,
        sensor_pitch=0.001,
        n_sensors=4,
        nx=13,
        ny_per_region=[3],
        max_frequency=5e5,
        run_postprocess=False,
        structure_mode="1_layer",
        single_layer=dict(thickness=0.001, rho=2700.0, E=70e9, nu=0.33),
    )
    config = tmp_path / "input.yaml"
    config.write_text(yaml.safe_dump(cfg))
    output = tmp_path / "comparison"
    command = [
        sys.executable,
        str(TOOL),
        "--config",
        str(config),
        "--output",
        str(output),
        "--duration",
        "5e-6",
        "--refinement-factor",
        "1.5",
    ]
    for extra in ([], ["--resume"]):
        completed = subprocess.run(command + extra, capture_output=True, text=True, timeout=60)
        assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((output / "comparison.json").read_text())
    assert [(r["nx"], r["ny_per_region"]) for r in report["runs"]] == [
        (20, [5]),
        (13, [3]),
        (13, [3]),
    ]
    assert all(r["wall_seconds"] is None for r in report["runs"])
    assert report["receivers"] == 4


def test_relative_metrics_rejects_zero_reference():
    import numpy as np

    zeros = np.zeros((16, 4, 2))
    ones = np.ones((16, 4, 2))
    with pytest.raises(ValueError, match="zero or non-finite amplitude"):
        comparison.relative_metrics(ones, zeros)
