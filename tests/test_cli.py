"""CLI interface and argument handling regression tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from guidedwavefem.cli import main


def test_cli_generate_config(tmp_path):
    out_yaml = tmp_path / "custom_template.yaml"
    sys.argv = ["guided-waves", "--generate-config", str(out_yaml)]
    main()
    assert out_yaml.exists()
    content = out_yaml.read_text(encoding="utf-8")
    assert "solver:" in content
    assert "geometry:" in content
    assert "structure:" in content


def test_cli_missing_config_exits():
    sys.argv = ["guided-waves"]
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_cli_dry_run(capsys):
    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "01_single_layer_aluminum.yaml"
    )
    sys.argv = ["guided-waves", "-c", str(cfg_path), "--dry-run"]
    main()
    captured = capsys.readouterr()
    assert "PREFLIGHT NUMERICAL ESTIMATE" in captured.out
    assert "Dry run complete" in captured.out


def test_cli_postprocess(tmp_path):
    # Create minimal mock sensor data
    t = np.arange(256) * 1e-7
    x = np.arange(16) * 0.001
    sig = np.sin(2 * np.pi * 5e5 * t[:, None] - 1000 * x[None, :])
    data_file = tmp_path / "sensor_data.npz"
    np.savez(
        str(data_file),
        times=t,
        sensor_x=x,
        sensor_data_y=sig,
        sensor_data_x=sig,
        f_center=5e5,
        c_L=6000.0,
        c_T=3000.0,
    )

    out_dir = tmp_path / "postprocessed"
    sys.argv = [
        "guided-waves",
        "--postprocess",
        str(data_file),
        "--output",
        str(out_dir),
    ]
    main()
    assert (out_dir / "full_analysis.png").exists()
