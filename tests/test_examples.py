"""Shortened end-to-end runs of every shipped example (all structure modes)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("0*.yaml"))


@pytest.mark.slow
@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: p.stem)
def test_shipped_example_runs_end_to_end(tmp_path, config):
    """Each example runs through the CLI with a coarse mesh and a few periods."""
    f0 = float(yaml.safe_load(config.read_text())["excitation"]["f_center"])
    out = tmp_path / config.stem
    cmd = [
        sys.executable,
        "-m",
        "guidedwavefem",
        "-c",
        str(config),
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
