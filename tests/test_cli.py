"""CLI interface and argument handling regression tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wavesimfem import __version__
from wavesimfem.cli import main
from wavesimfem.reporting import logger

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG_01 = CONFIG_DIR / "01_plate_aluminum_lamb.yaml"
EXAMPLES = sorted(CONFIG_DIR.glob("*.yaml"))


@pytest.fixture(autouse=True)
def _restore_log_level():
    level = logger.level
    yield
    logger.setLevel(level)


def _write_archive(path: Path) -> Path:
    t = np.arange(256) * 1e-7
    x = np.arange(16) * 0.001
    sig = np.sin(2 * np.pi * 5e5 * t[:, None] - 1000 * x[None, :])
    np.savez(
        path,
        times=t,
        sensor_x=x,
        sensor_data_y=sig,
        sensor_data_x=sig,
        f_center=5e5,
        c_L=6000.0,
        c_T=3000.0,
    )
    return path


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_cli_characterisation_existing_archive(tmp_path):
    import json

    archive = _write_archive(tmp_path / "sensor_data.npz")
    config = tmp_path / "analysis.yaml"
    config.write_text(
        "postprocess:\n  wave_characterisation:\n    enabled: true\n"
        "output:\n  figure_formats: [png]\n  usetex: false\n",
        encoding="utf-8",
    )
    main(["-c", str(config), "--postprocess", str(archive), "--output", str(tmp_path)])
    report = json.loads((tmp_path / "characterisation/metrics.json").read_text())
    assert report["options"]["enabled"] is True
    assert len(report["figures"]) == 4


def test_cli_generate_config(tmp_path):
    out_yaml = tmp_path / "custom_template.yaml"
    main(["--generate-config", str(out_yaml)])
    content = out_yaml.read_text(encoding="utf-8")
    for section in ("solver:", "geometry:", "structure:", "device:", "precision:"):
        assert section in content


def test_generated_template_is_valid(tmp_path):
    from wavesimfem.config import parse_and_resolve_config

    out_yaml = tmp_path / "template.yaml"
    main(["--generate-config", str(out_yaml)])
    cfg = parse_and_resolve_config(str(out_yaml))
    assert cfg["device"] == "auto" and cfg["precision"] == "double"


def test_cli_missing_config_exits():
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 1


def test_cli_dry_run_requires_config(tmp_path, capsys):
    archive = _write_archive(tmp_path / "sensor_data.npz")
    with pytest.raises(SystemExit) as exc_info:
        main(["--postprocess", str(archive), "--dry-run"])
    assert exc_info.value.code == 2
    assert "--dry-run requires" in capsys.readouterr().err


def test_cli_dry_run(capsys):
    main(["-c", str(CONFIG_01), "--dry-run"])
    out = capsys.readouterr().out
    assert "PREFLIGHT NUMERICAL ESTIMATE" in out
    assert "Dry run complete" in out


@pytest.mark.parametrize("example", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_cli_dry_run_every_example(example, capsys):
    main(["-c", str(example), "--dry-run"])
    out = capsys.readouterr().out
    assert "Excitation:" in out and "Dry run complete" in out


def test_cli_quiet_suppresses_info(capsys):
    # Supply the recording window so this checks informational output alone;
    # warnings about automatic higher-mode arrivals must survive quiet mode.
    main(["-c", str(CONFIG_01), "--duration", "1e-4", "--dry-run", "--quiet"])
    assert capsys.readouterr().out == ""


def test_cli_device_and_precision_args(capsys):
    main(["-c", str(CONFIG_01), "--dry-run", "--device", "cpu", "--precision", "single"])
    assert "Dry run complete" in capsys.readouterr().out


def test_cli_invalid_device_rejected(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["-c", str(CONFIG_01), "--dry-run", "--device", "gpu0"])
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "wavesim: error: invalid configuration" in err and "Invalid device" in err
    assert "Traceback" not in err


def test_cli_verbose_reraises_config_errors():
    with pytest.raises(ValueError, match="Invalid device"):
        main(["-c", str(CONFIG_01), "--dry-run", "--device", "gpu0", "-v"])


def test_cli_missing_config_file_is_clean_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["-c", str(tmp_path / "nope.yaml"), "--dry-run"])
    assert exc_info.value.code == 2
    assert "Configuration file not found" in capsys.readouterr().err


def test_cli_postprocess(tmp_path):
    import json

    archive = _write_archive(tmp_path / "sensor_data.npz")
    out_dir = tmp_path / "postprocessed"
    main(["--postprocess", str(archive), "--output", str(out_dir)])
    assert (out_dir / "full_analysis.png").exists()
    report = json.loads((out_dir / "wave_physics/report.json").read_text())
    assert report["profile"]["provenance"] == "legacy"
    assert not report["profile"]["lamb_applicable"]
    assert (out_dir / "wave_physics/dispersion_and_modes.png").exists()


def test_cli_postprocess_unit_string_limits(tmp_path):
    archive = _write_archive(tmp_path / "sensor_data.npz")
    out_dir = tmp_path / "limits"
    main(
        [
            "--postprocess",
            str(archive),
            "--output",
            str(out_dir),
            "--f-max",
            "1MHz",
            "--k-max",
            "3 rad/mm",
        ]
    )
    assert (out_dir / "full_analysis.png").exists()


def test_cli_bad_limit_unit_is_clean_error(tmp_path, capsys):
    archive = _write_archive(tmp_path / "sensor_data.npz")
    with pytest.raises(SystemExit) as exc_info:
        main(["--postprocess", str(archive), "--f-max", "2 rad/mm"])
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "unknown frequency unit" in err and "Traceback" not in err


def test_cli_postprocess_respects_config_formats(tmp_path):
    archive = _write_archive(tmp_path / "sensor_data.npz")
    cfg_path = tmp_path / "post.yaml"
    cfg_path.write_text("output:\n  figure_formats: [png]\n  usetex: false\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    main(["-c", str(cfg_path), "--postprocess", str(archive), "--output", str(out_dir)])
    assert sorted(p.suffix for p in out_dir.glob("full_analysis.*")) == [".png"]
    assert sorted(p.suffix for p in out_dir.glob("wave_physics/dispersion_and_modes.*")) == [".png"]


def test_cli_structure_report_can_be_disabled(tmp_path):
    archive = _write_archive(tmp_path / "sensor_data.npz")
    config = tmp_path / "post.yaml"
    config.write_text(
        "postprocess:\n  structure_analysis: false\n"
        "output:\n  figure_formats: [png]\n  usetex: false\n",
        encoding="utf-8",
    )
    destination = tmp_path / "out"
    main(["-c", str(config), "--postprocess", str(archive), "--output", str(destination)])
    assert (destination / "full_analysis.png").exists()
    assert not (destination / "wave_physics").exists()


def test_cli_postprocess_missing_file(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--postprocess", str(tmp_path / "missing.npz")])
    assert exc_info.value.code == 2
    assert "sensor data file not found" in capsys.readouterr().err


def test_cli_frequency_overrides_dry_run(capsys):
    # Verify that overriding max_frequency below YAML min_frequency cleanly adapts
    main(["-c", str(CONFIG_01), "--max-frequency", "500000", "--dry-run"])
    out = capsys.readouterr().out
    assert "Dry run complete" in out

    # Verify explicit --min-frequency override
    main(
        [
            "-c",
            "configs/03_halfspace_steel_rayleigh.yaml",
            "--min-frequency",
            "100000",
            "--max-frequency",
            "500000",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert "Dry run complete" in out
