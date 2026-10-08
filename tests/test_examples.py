"""Shortened end-to-end runs of every shipped example (all structure modes)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("0*.yaml"))
# Fixed smoke-test grids preserve the earlier short-run meshes independently of
# the frequency used for temporal accuracy and sensor sampling.
SMOKE_GRIDS = {
    "01_plate_aluminum_lamb": (857, [7]),
    "02_layers_bonded_joint": (2910, [8, 4, 8]),
    "03_halfspace_steel_rayleigh": (376, [49]),
    "04_plate_on_halfspace_soil_bedrock": (100, [34, 8]),
    "05_layers_bone_qus": (379, [6, 16, 6]),
    "06_plate_in_halfspaces_immersed_water": (379, [17, 6, 17]),
}


@pytest.mark.slow
@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: p.stem)
def test_shipped_example_runs_end_to_end(tmp_path, config):
    """Each example runs through the CLI with a coarse mesh and a few periods."""
    cfg = yaml.safe_load(config.read_text())
    f0 = float(cfg["excitation"]["f_center"])
    nx, rows = SMOKE_GRIDS[config.stem]
    cfg["mesh"].update(nx=nx, ny_per_region=rows)
    smoke_config = tmp_path / config.name
    smoke_config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp_path / config.stem
    cmd = [
        sys.executable,
        "-m",
        "guidedwavefem",
        "-c",
        str(smoke_config),
        "-q",
        "--duration",
        repr(4.0 / f0),
        "--max-frequency",
        repr(f0),
        "--no-postprocess",
        "--output",
        str(out),
    ]
    subprocess.run(cmd, check=True, timeout=600)
    with np.load(out / "sensor_data.npz") as data:
        assert np.isfinite(data["sensor_data_y"]).all()
        assert np.abs(data["sensor_data_y"]).max() > 0
        assert data["times"][-1] == pytest.approx(4.0 / f0)
        runtime = yaml.safe_load(str(data["simulation_config_yaml"]))["runtime"]
        assert runtime["nx"] == nx
        assert runtime["ny_per_region"] == rows
