"""Result integrity on failed writes and bounded-memory history recording."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from guidedwavefem.config import dump_config
from guidedwavefem.storage import HistoryStore, all_finite_history, atomic_output


def test_atomic_output_preserves_previous_file_on_write_failure(tmp_path):
    path = tmp_path / "previous.txt"
    path.write_text("complete previous result")
    with pytest.raises(OSError, match="disk full"), atomic_output(path) as stream:
        stream.write("partial new result")
        raise OSError("disk full")
    assert path.read_text() == "complete previous result"
    assert list(tmp_path.iterdir()) == [path]


def test_dump_config_preserves_previous_file_on_serialization_failure(tmp_path):
    path = tmp_path / "config.yaml"
    dump_config({"run": "old"}, str(path))
    previous = path.read_bytes()
    with pytest.raises(yaml.representer.RepresenterError):
        dump_config({"bad": object()}, str(path))
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_output_preserves_previous_file_on_replace_failure(tmp_path, monkeypatch):
    path = tmp_path / "previous.bin"
    path.write_bytes(b"old")

    def fail(*_args):
        raise OSError("cannot replace")

    monkeypatch.setattr("guidedwavefem.storage.os.replace", fail)
    with (
        pytest.raises(OSError, match="cannot replace"),
        atomic_output(path, binary=True) as stream,
    ):
        stream.write(b"new")
    assert path.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("compressed", [False, True])
def test_committed_archive_embeds_matching_configuration(tmp_path, compressed):
    from guidedwavefem.solvers import _publish_results

    cfg = {"compress_output": compressed, "runtime": {"run_id": "new"}}
    path = _publish_results(cfg, {"run_id": "new", "values": np.arange(15)}, str(tmp_path))
    with np.load(path, allow_pickle=False) as data:
        embedded = yaml.safe_load(str(data["simulation_config_yaml"]))
        assert embedded == yaml.safe_load((tmp_path / "simulation_config.yaml").read_text())
        assert embedded["runtime"]["run_id"] == str(data["run_id"])
        np.testing.assert_array_equal(data["values"], np.arange(15))
    assert not list(tmp_path.glob(".gwfem-*"))


def test_failed_archive_write_keeps_previous_archive_and_sidecar(tmp_path, monkeypatch):
    from guidedwavefem.solvers import _publish_results

    cfg = {"compress_output": False, "runtime": {"run_id": "old"}}
    _publish_results(cfg, {"run_id": "old"}, str(tmp_path))
    previous = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    def fail(stream, **_arrays):
        stream.write(b"partial ZIP data")
        raise OSError("disk full")

    monkeypatch.setattr("guidedwavefem.solvers.np.savez", fail)
    cfg["runtime"]["run_id"] = "new"
    with pytest.raises(OSError, match="disk full"):
        _publish_results(cfg, {"run_id": "new"}, str(tmp_path))
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == previous


def test_sidecar_failure_leaves_self_contained_committed_archive(tmp_path, monkeypatch, capsys):
    import guidedwavefem.solvers as solvers

    cfg = {"compress_output": False, "runtime": {"run_id": "old"}}
    solvers._publish_results(cfg, {"run_id": "old"}, str(tmp_path))
    original = solvers.os.replace

    def fail_sidecar(source, destination):
        if Path(destination) == tmp_path / "simulation_config.yaml":
            raise OSError("sidecar unavailable")
        return original(source, destination)

    monkeypatch.setattr(solvers.os, "replace", fail_sidecar)
    cfg["runtime"]["run_id"] = "new"
    path = solvers._publish_results(cfg, {"run_id": "new"}, str(tmp_path))
    with np.load(path, allow_pickle=False) as data:
        assert yaml.safe_load(str(data["simulation_config_yaml"]))["runtime"]["run_id"] == "new"
        assert str(data["run_id"]) == "new"
    assert (
        yaml.safe_load((tmp_path / "simulation_config.yaml").read_text())["runtime"]["run_id"]
        == "old"
    )
    assert "Sensor archive committed" in capsys.readouterr().out
    assert not list(tmp_path.glob(".gwfem-*"))


def test_history_store_cleanup_on_exception_and_chunked_finite_check(tmp_path):
    with (
        pytest.raises(RuntimeError, match="interrupted"),
        HistoryStore("disk", str(tmp_path)) as store,
    ):
        history = store.zeros((11, 3), np.float32)
        assert isinstance(history, np.memmap)
        assert not history.any()
        assert all_finite_history(history, chunk_bytes=12)
        history[-1, -1] = np.inf
        assert not all_finite_history(history, chunk_bytes=12)
        assert store.zeros((11, 0)).shape == (11, 0)
        raise RuntimeError("interrupted")
    assert history._mmap.closed
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("solver", ["explicit", "implicit"])
def test_disk_and_memory_fem_histories_match(tmp_path, solver):
    from guidedwavefem import run_simulation

    cfg = {
        "solver_type": solver,
        "device": "cpu",
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
        "structure_mode": "1_layer",
        "single_layer": {"thickness": 0.001, "rho": 2700.0, "E": 70e9, "nu": 0.33},
    }
    saved = []
    for mode in ("memory", "disk"):
        path, _ = run_simulation(
            cfg={**cfg, "history_storage": mode, "output_dir": str(tmp_path / mode)}
        )
        with np.load(path) as data:
            saved.append({key: data[key] for key in ("times", "sensor_data_x", "sensor_data_y")})
        assert not list((tmp_path / mode).glob(".gwfem-*"))
    for key in saved[0]:
        np.testing.assert_array_equal(saved[0][key], saved[1][key])


@pytest.mark.parametrize("chunk_bytes", [1, 72])
def test_chunked_gather_writes_provided_disk_array(tmp_path, chunk_bytes):
    from mpi4py import MPI

    from guidedwavefem.numerics import PointSensors

    # COMM_SELF with deliberately shuffled ownership exercises column placement
    # independently of the geometry used to assign those columns.
    probes = object.__new__(PointSensors)
    probes.comm = MPI.COMM_SELF
    probes.indices = np.array([2, 0, 1])
    probes.count = 3
    expected = np.arange(33, dtype=float).reshape(11, 3)
    with HistoryStore("disk", str(tmp_path)) as store:
        output = store.zeros(expected.shape)
        gathered = probes.gather(expected[:, probes.indices], out=output, chunk_bytes=chunk_bytes)
        assert gathered is output
        np.testing.assert_array_equal(gathered, expected)
