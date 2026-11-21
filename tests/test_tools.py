"""Repository tools, packaging guards, and cross-module edge-case regressions."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("clean_junk", REPO / "tools" / "clean_junk.py")
clean_junk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clean_junk)

JUNK = [
    "__pycache__/x.cpython-311.pyc",
    "src/wavesimfem/__pycache__/cli.cpython-311.pyc",
    "stray.pyc",
    ".pytest_cache/v/cache/nodeids",
    "tests/.ruff_cache/0.16.9/abc",
    ".mypy_cache/3.11/x.json",
    "notebooks/.ipynb_checkpoints/a-checkpoint.ipynb",
    "htmlcov/index.html",
    "src/wavesimfem.egg-info/PKG-INFO",
    "build/lib/x.py",
    "dist/wavesimfem-0.3.0.tar.gz",
    ".DS_Store",
    "configs/.DS_Store",
    "configs/._01_plate_aluminum_lamb.yaml",
    "Thumbs.db",
    "src/.cli.py.swp",
    "README.md~",
    ".coverage",
    ".coverage.host.1234",
    "coverage.xml",
]
KEEP = [
    "src/wavesimfem/cli.py",
    "configs/01_plate_aluminum_lamb.yaml",
    "README.md",
    "tools/clean_junk.py",
    "results_case01/sensor_data.npz",
    "results_case01/__pycache__/keep.pyc",  # results are never touched
    ".dev/scratch/.DS_Store",  # protected scratch area
    ".venv/lib/x.pyc",  # virtual environments are never touched
    "src/build/keep.txt",  # build/ is junk only at the repository root
]


def _hooks() -> list[dict]:
    config = yaml.safe_load((REPO / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    return [hook for repo in config["repos"] for hook in repo["hooks"]]


def _forbid_regex() -> re.Pattern[str]:
    (hook,) = [hook for hook in _hooks() if hook["id"] == "forbid-junk"]
    return re.compile(hook["files"])


def test_hook_order_guard_first_cleaner_last():
    ids = [hook["id"] for hook in _hooks()]
    assert ids[0] == "forbid-junk"
    assert ids[-1] == "clean-junk"  # must run after hooks that create caches (e.g. ruff)


def _make_tree(root: Path, paths: list[str]) -> None:
    for rel in paths:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")


@pytest.mark.parametrize("path", JUNK)
def test_forbid_junk_regex_blocks_junk(path):
    assert _forbid_regex().search(path)


@pytest.mark.parametrize(
    "path", ["src/wavesimfem/cli.py", "configs/README.md", "tools/clean_junk.py", "LICENSE"]
)
def test_forbid_junk_regex_allows_sources(path):
    assert not _forbid_regex().search(path)


def test_cleaner_removes_junk_and_keeps_everything_else(tmp_path, capsys):
    _make_tree(tmp_path, JUNK + KEEP)
    assert clean_junk.main(["--root", str(tmp_path)]) == 0
    assert "clean-junk: Removed" in capsys.readouterr().out
    for rel in JUNK:
        assert not (tmp_path / rel).exists(), rel
    for rel in KEEP:
        assert (tmp_path / rel).exists(), rel
    assert not (tmp_path / "src" / "wavesimfem" / "__pycache__").exists()


def test_cleaner_dry_run_deletes_nothing(tmp_path, capsys):
    _make_tree(tmp_path, JUNK)
    assert clean_junk.main(["--root", str(tmp_path), "--dry-run"]) == 0
    assert "Would remove" in capsys.readouterr().out
    for rel in JUNK:
        assert (tmp_path / rel).exists(), rel


def test_cleaner_never_deletes_tracked_files(tmp_path):
    _make_tree(tmp_path, ["tracked/__pycache__/vendored.pyc", "tracked/__pycache__/stray.pyc"])
    found = {
        p.relative_to(tmp_path).as_posix()
        for p in clean_junk.find_junk(tmp_path, frozenset({"tracked/__pycache__/vendored.pyc"}))
    }
    assert found == {"tracked/__pycache__/stray.pyc"}


def test_cleaner_reports_clean_tree(tmp_path, capsys):
    _make_tree(tmp_path, KEEP)
    assert clean_junk.main(["--root", str(tmp_path)]) == 0
    assert "working tree is clean" in capsys.readouterr().out


def test_units_reject_numpy_boolean():
    import numpy as np

    from wavesimfem import units

    with pytest.raises(ValueError, match="got a boolean"):
        units.parse_quantity(np.bool_(True), "frequency")


def test_sources_reject_invalid_frequency_and_cycles():
    from wavesimfem import sources

    with pytest.raises(ValueError, match="f0 must be finite"):
        sources.ricker(0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="n_cycles must be >= 1"):
        sources.source_signal(0.0, "tone_burst", 1e6, n_cycles=0)
    with pytest.raises(ValueError, match="n_cycles must be >= 1"):
        sources.source_signal(0.0, "tone_burst", 1e6, n_cycles=True)


def test_device_canonicalizes_leading_zeros():
    from wavesimfem import device

    assert device.normalize_device_request("cuda:00") == "cuda:0"


def test_parallel_rank_count_rejects_trailing_characters():
    from wavesimfem import parallel

    assert parallel._leading_int("2x4") is None
    assert parallel._leading_int(" 4 ") == 4


def test_history_store_creates_missing_directory(tmp_path):
    import numpy as np

    from wavesimfem import storage

    disk_dir = tmp_path / "nested" / "scratch"
    with storage.HistoryStore("disk", directory=disk_dir) as store:
        arr = store.zeros((4, 2), np.float64)
        arr[0] = 1.0
        assert storage.all_finite_history(np.ones(4))
    assert store._scratch is None


def test_config_wraps_invalid_toml(tmp_path):
    from wavesimfem import config

    bad_toml = tmp_path / "bad.toml"
    bad_toml.write_text("[solver\ntype = 1", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid TOML"):
        config.load_config(str(bad_toml))


def test_auto_sensors_fit_without_lateral_sponge():
    from wavesimfem import config

    cfg = config.parse_and_resolve_config(
        custom_cfg={
            "sensing_length": 0.025,
            "absorb_length": 0.0,
            "single_layer": {
                "name": "Water",
                "thickness": 2e-3,
                "material_model": "fluid",
                "rho": 1000.0,
                "c": 1500.0,
            },
            "source_kind": "acoustic_normal_acceleration",
            "source_boundary": "top",
            "source_amplitude": 1e5,
            "f_center": 250e3,
            "nx": 50,
            "ny_per_region": [4],
            "CFL": 0.7,
        }
    )
    assert (
        cfg["source_x"] + cfg["sensor_offset"] + (cfg["n_sensors"] - 1) * cfg["sensor_pitch"]
        <= 0.025
    )


def test_cli_rejects_conflicting_postprocess_flags():
    from wavesimfem import cli

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--postprocess", "a.npz", "--no-postprocess"])
    assert exc_info.value.code == cli.EXIT_USAGE


def test_characterisation_rejects_boolean_time_window():
    from wavesimfem.characterisation import validate_options

    with pytest.raises(ValueError, match="got a boolean"):
        validate_options({"time_window": [True, 1.0]})


def test_characterisation_rejects_motion_component_for_pressure(tmp_path):
    import numpy as np

    from wavesimfem.characterisation import write_characterisation

    pressure_npz = tmp_path / "pressure.npz"
    np.savez(
        pressure_npz,
        times=np.linspace(0.0, 31e-6, 32),
        sensor_x=np.linspace(0.0, 3e-3, 4),
        sensor_data_pressure=np.zeros((32, 4)),
        sensor_observable="pressure",
        sensor_units="Pa",
    )
    with pytest.raises(ValueError, match="Pressure archives"):
        write_characterisation(pressure_npz, tmp_path / "char_out", options={"component": "x"})


def test_legacy_regions_identify_fluid_and_rayleigh_speed():
    from wavesimfem.structure_analysis import _legacy_regions

    fluid_regs = _legacy_regions({"c_L": 1500.0, "c_T": 0.1, "plate_thickness": 2e-3}, "1_layer")
    assert fluid_regs[0]["material_model"] == "fluid"
    hs_regs = _legacy_regions(
        {"material_model": "isotropic", "c_L": 6000.0, "c_T": 3100.0, "total_depth": 5e-3},
        "halfspace",
    )
    assert hs_regs[0]["c_R_m_s"] is not None and hs_regs[0]["c_R_m_s"] > 0


def test_phase_velocity_recovers_single_traveling_wave():
    import numpy as np

    from wavesimfem.wave_spectra import phase_velocity_spectrum

    t = np.linspace(0.0, 40e-6, 256)
    x = np.linspace(0.0, 40e-3, 81)
    wave = np.cos(2 * np.pi * 200e3 * (t[:, None] - x[None, :] / 2500.0))
    spec = phase_velocity_spectrum(t, x, wave, np.linspace(1500.0, 3500.0, 81))
    assert np.isfinite(spec["ridge_velocity"]).any()


def test_benchmark_rejects_reserved_override():
    import argparse

    bench_spec = importlib.util.spec_from_file_location(
        "benchmark", REPO / "tools" / "benchmark.py"
    )
    benchmark = importlib.util.module_from_spec(bench_spec)
    bench_spec.loader.exec_module(benchmark)

    with pytest.raises(argparse.ArgumentTypeError, match="managed by the benchmark harness"):
        benchmark.parse_override("output_dir=foo")


def test_compare_resolution_extracts_pressure_history(tmp_path):
    import numpy as np

    comp_spec = importlib.util.spec_from_file_location(
        "compare_resolution", REPO / "tools" / "compare_resolution.py"
    )
    compare_resolution = importlib.util.module_from_spec(comp_spec)
    comp_spec.loader.exec_module(compare_resolution)

    p_archive = tmp_path / "p.npz"
    np.savez(
        p_archive,
        times=np.linspace(0.0, 1e-5, 16),
        sensor_x=np.linspace(0.0, 1e-2, 4),
        sensor_data_pressure=np.ones((16, 4)),
        sensor_observable="pressure",
    )
    with np.load(p_archive, allow_pickle=False) as data:
        extracted = compare_resolution._extract_history(data)
    assert extracted.shape == (16, 4, 1)


@pytest.mark.parametrize("config_name", ["07_fluid_acoustic.yaml", "08_fluid_solid_acoustic.yaml"])
def test_structure_preview_creates_pressure_archive(tmp_path, monkeypatch, config_name):
    import sys

    import numpy as np

    prev_spec = importlib.util.spec_from_file_location(
        "preview_structure_reports", REPO / "tools" / "preview_structure_reports.py"
    )
    preview_structure_reports = importlib.util.module_from_spec(prev_spec)
    monkeypatch.setitem(sys.modules, prev_spec.name, preview_structure_reports)
    prev_spec.loader.exec_module(preview_structure_reports)
    archive_path, _ = preview_structure_reports.create_synthetic_archive(
        REPO / "configs" / config_name, tmp_path / config_name[:2]
    )
    with np.load(archive_path, allow_pickle=False) as data:
        assert "sensor_data_pressure" in data.files
        assert int(data["schema_version"]) == 2
        saved_cfg = yaml.safe_load(str(data["simulation_config_yaml"]))
        assert "sensor_sample_rate" in saved_cfg and "sample_rate" not in saved_cfg
        from wavesimfem.config import parse_and_resolve_config

        resolved = parse_and_resolve_config(custom_cfg=saved_cfg, validate_simulation=False)
        assert resolved["sensor_sample_rate"] > 0


def test_benchmark_write_report_uses_atomic_json(tmp_path):
    import json

    bench_spec = importlib.util.spec_from_file_location(
        "benchmark", REPO / "tools" / "benchmark.py"
    )
    benchmark = importlib.util.module_from_spec(bench_spec)
    bench_spec.loader.exec_module(benchmark)

    out = tmp_path / "benchmark.json"
    benchmark.write_report(out, {"b": 2, "a": 1})
    assert json.loads(out.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
