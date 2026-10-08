#!/usr/bin/env python3
"""Run a configured FEM case and record timing, per-rank peak RSS, and provenance.

Use a fresh output directory for each process invocation. RSS is the operating
system process-lifetime high-water mark (host memory only), not a sampled or GPU
allocation measurement. Multi-rank sums are upper bounds, not simultaneous peaks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import socket
import sys
import tempfile
import time
from pathlib import Path


def peak_rss_bytes(value: float | None = None, system: str | None = None) -> int:
    """Normalize getrusage's bytes on macOS and KiB on supported Linux hosts."""
    if value is None:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    system = system or platform.system()
    if system not in {"Darwin", "Linux"}:
        raise RuntimeError(f"Peak RSS units are not defined for this platform: {system}")
    return int(value * (1 if system == "Darwin" else 1024))


def parse_override(text: str) -> tuple[str, object]:
    """Parse a flat config key and JSON value; config validation owns the schema."""
    key, separator, value = text.partition("=")
    if not separator or not key.strip():
        raise argparse.ArgumentTypeError("Expected KEY=JSON_VALUE, for example duration=1e-5")
    try:
        return key.strip(), json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON override {text!r}: {exc.msg}") from exc


def summarize_ranks(ranks: list[dict]) -> dict:
    """Keep per-node peaks meaningful when individual rank peaks occur at different times."""
    by_host: dict[str, int] = {}
    for row in ranks:
        by_host[row["hostname"]] = by_host.get(row["hostname"], 0) + row["peak_rss_bytes"]
    return {
        "mpi_ranks": len(ranks),
        "hosts": sorted(by_host),
        "wall_seconds_max": max(row["wall_seconds"] for row in ranks),
        "peak_rss_bytes_max_rank": max(row["peak_rss_bytes"] for row in ranks),
        "peak_rss_bytes_sum_rank_peaks_upper_bound": sum(by_host.values()),
        "peak_rss_bytes_by_host_upper_bound": by_host,
    }


def write_report(path: Path, report: dict) -> None:
    """Publish a complete JSON report, leaving an existing report intact on failure."""
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", "-c", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True, help="New benchmark directory")
    result.add_argument(
        "--set",
        dest="overrides",
        type=parse_override,
        action="append",
        default=[],
        metavar="KEY=JSON",
    )
    result.add_argument(
        "--minimum-hosts", type=int, default=1, help="Fail if fewer hosts participate"
    )
    return result


def _run(args, MPI):
    # These imports occur after the thread policy is applied in main().
    import numpy as np
    import yaml

    from guidedwavefem import load_config, run_simulation, set_verbosity
    from guidedwavefem.config import flatten_config
    from guidedwavefem.parallel import THREAD_ENV_VARS

    comm = MPI.COMM_WORLD
    hosts = comm.allgather(socket.gethostname())
    if args.minimum_hosts < 1:
        raise ValueError("--minimum-hosts must be positive")
    if len(set(hosts)) < args.minimum_hosts:
        raise ValueError(
            f"Expected at least {args.minimum_hosts} hosts, got {len(set(hosts))}: {sorted(set(hosts))}"
        )
    output = args.output.resolve()
    cfg = None
    config_sha = None
    if comm.rank == 0:
        cfg = flatten_config(load_config(str(args.config)))
        cfg.update(dict(args.overrides))
        # Benchmark cost excludes plotting and VTX snapshots; output archives are included.
        cfg.update(output_dir=str(output / "simulation"), run_postprocess=False, write_vtx=False)
        config_sha = hashlib.sha256(args.config.read_bytes()).hexdigest()
        output.mkdir(parents=True, exist_ok=False)
    cfg = comm.bcast(cfg, root=0)
    set_verbosity("WARNING")
    baseline = peak_rss_bytes()
    comm.Barrier()
    started = time.perf_counter()
    archive, _simulation_dir = run_simulation(cfg=cfg)
    comm.Barrier()
    wall = time.perf_counter() - started
    local = {
        "rank": comm.rank,
        "hostname": hosts[comm.rank],
        "pid": os.getpid(),
        "wall_seconds": wall,
        "peak_rss_bytes": peak_rss_bytes(),
        "baseline_peak_rss_bytes": baseline,
        "thread_environment": {key: os.environ.get(key) for key in THREAD_ENV_VARS},
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
    }
    ranks = comm.gather(local, root=0)
    if comm.rank == 0:
        with np.load(archive, allow_pickle=False) as data:
            saved = yaml.safe_load(str(data["simulation_config_yaml"]))
        report = {
            "schema_version": 1,
            "configuration_file": str(args.config.resolve()),
            "configuration_file_sha256": config_sha,
            "overrides": dict(args.overrides),
            "effective_configuration": saved,
            "archive": str(Path(archive).resolve()),
            "archive_bytes": Path(archive).stat().st_size,
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "mpi_library": MPI.Get_library_version().strip(),
            "measurement": {
                "includes": "FEM setup, solve, MPI gathers, and output archive; no plots or VTX snapshots",
                "peak_rss": "OS process-lifetime host-memory high-water mark; excludes GPU memory",
                "rank_peak_sums": "Upper bounds: rank peaks need not occur at the same time",
                "cache_state": "Uncontrolled: repeat fresh processes and report cold/warm runs separately",
            },
            "summary": summarize_ranks(ranks),
            "ranks": ranks,
        }
        write_report(output / "benchmark.json", report)
        print(json.dumps({"benchmark": str(output / "benchmark.json"), **report["summary"]}))
    comm.Barrier()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    from guidedwavefem.parallel import limit_threads_per_rank

    limit_threads_per_rank()
    from mpi4py import MPI

    from guidedwavefem.solvers import mpi_fail_fast

    return mpi_fail_fast(_run)(args, MPI)


if __name__ == "__main__":
    raise SystemExit(main())
