"""Benchmark reports preserve units, rank attribution, and valid atomic JSON."""

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "benchmark", Path(__file__).resolve().parents[1] / "tools" / "benchmark.py"
)
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def test_peak_rss_units_are_platform_specific():
    assert benchmark.peak_rss_bytes(1234, "Darwin") == 1234
    assert benchmark.peak_rss_bytes(1234, "Linux") == 1234 * 1024
    with pytest.raises(RuntimeError, match="units are not defined"):
        benchmark.peak_rss_bytes(1234, "Unknown")


def test_report_distinguishes_node_and_rank_peaks():
    rows = [
        dict(hostname="node-a", peak_rss_bytes=100, wall_seconds=5),
        dict(hostname="node-b", peak_rss_bytes=150, wall_seconds=6),
        dict(hostname="node-a", peak_rss_bytes=120, wall_seconds=4),
    ]
    summary = benchmark.summarize_ranks(rows)
    assert summary["hosts"] == ["node-a", "node-b"]
    assert summary["mpi_ranks"] == 3
    assert summary["wall_seconds_max"] == 6
    assert summary["peak_rss_bytes_max_rank"] == 150
    assert summary["peak_rss_bytes_by_host_upper_bound"] == {"node-a": 220, "node-b": 150}
    assert summary["peak_rss_bytes_sum_rank_peaks_upper_bound"] == 370


@pytest.mark.parametrize("text", ["duration", "=2", "duration=1 ns", "duration="])
def test_override_requires_json_value(text):
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark.parse_override(text)


def test_override_preserves_numbers_strings_and_booleans():
    assert benchmark.parse_override("duration=2e-6") == ("duration", 2e-6)
    assert benchmark.parse_override('history_storage="disk"') == ("history_storage", "disk")
    assert benchmark.parse_override("compress_output=false") == ("compress_output", False)


def test_failed_serialization_leaves_existing_report_intact(tmp_path):
    path = tmp_path / "benchmark.json"
    benchmark.write_report(path, {"complete": True})
    with pytest.raises(ValueError):
        benchmark.write_report(path, {"invalid": float("nan")})
    assert json.loads(path.read_text()) == {"complete": True}
    assert list(tmp_path.iterdir()) == [path]
