"""Unsuccessful rank-local exits must terminate peers instead of hanging."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from mpi4py import MPI

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit, RuntimeError])
def test_mpi_abort_flushes_complete_diagnostic(monkeypatch, error_type):
    from wavesimfem import solvers

    class BufferedStderr:
        def __init__(self):
            self.pending = []
            self.forwarded = []

        def write(self, text):
            self.pending.append(text)
            return len(text)

        def flush(self):
            self.forwarded.extend(self.pending)
            self.pending.clear()

    class AbortRequested(Exception):
        pass

    stderr = BufferedStderr()

    class AbortComm:
        def Abort(self, code):
            assert code == 1
            # The launcher must receive a complete diagnostic before termination,
            # including the exception line rather than just the first frame.
            assert any(
                "Traceback (most recent call last):" in text
                and f"{error_type.__name__}: injected rank-local failure" in text
                for text in stderr.forwarded
            )
            raise AbortRequested

    monkeypatch.setattr(solvers, "nprocs", 2)
    monkeypatch.setattr(solvers, "comm", AbortComm())
    monkeypatch.setattr(sys, "stderr", stderr)

    @solvers.mpi_fail_fast
    def fail():
        raise error_type("injected rank-local failure")

    with pytest.raises(AbortRequested):
        fail()


@pytest.mark.parametrize(
    "case",
    [
        "config",
        "interrupt",
        "exit",
        "exception",
        "help",
        "version",
        "gather_output",
        "gather_columns",
        "gather_times",
        "gather_dtype",
        "gather_budget",
    ],
)
def test_mpi_shutdown_on_rank_local_failure(case):
    if MPI.COMM_WORLD.size > 1:
        pytest.skip("Subprocess MPI tests must run from a single-rank pytest process")
    launcher = shutil.which("mpiexec") or str(Path(sys.executable).with_name("mpiexec"))
    if not Path(launcher).is_file():
        pytest.skip("MPI launcher unavailable")
    proc = subprocess.Popen(
        [
            launcher,
            "-n",
            "2",
            sys.executable,
            str(Path(__file__).with_name("mpi_failure_case.py")),
            case,
            str(ROOT / "configs" / "01_plate_aluminum_lamb.yaml"),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
        pytest.fail(f"MPI job hung after {case}:\n{stdout}\n{stderr}")
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    if case.startswith("gather_"):
        assert proc.returncode == 0, stdout + stderr
        assert "collective validation completed" in stdout
    elif case in ("help", "version"):
        assert proc.returncode == 0, stdout + stderr
        assert "wavesim" in stdout
    else:
        assert proc.returncode != 0, stdout + stderr
        assert "injected" in stderr, stdout + stderr
