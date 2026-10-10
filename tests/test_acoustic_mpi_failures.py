"""The acoustic public solver must report rank-local divergence collectively."""

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


def _rank_local_nonfinite_worker(solver_type: str) -> None:
    import numpy as np

    sys.path.insert(0, str(ROOT / "src"))
    from wavesimfem.acoustic_time import CoupledNewmark, CoupledRK4
    from wavesimfem.acoustics import build_acoustic_problem, solve_acoustic
    from wavesimfem.config import parse_and_resolve_config

    comm = MPI.COMM_WORLD
    assert comm.size == 2
    cfg = parse_and_resolve_config(
        custom_cfg={
            "solver_type": solver_type,
            "device": "cpu",
            "CFL": 0.5,
            "dt": 1e-8,
            "duration": 2e-8,
            "sensing_length": 0.01,
            "absorb_length": 0.0,
            "absorb_alpha": 0.0,
            "absorb_beta": 0.0,
            "f_center": 1e5,
            "source_x": 0.002,
            "source_kind": "acoustic_normal_acceleration",
            "source_amplitude": 1.0,
            "sensor_offset": 0.001,
            "sensor_pitch": 0.001,
            "n_sensors": 3,
            "nx": 4,
            "ny_per_region": [1],
            "degree": 1,
            "structure_mode": "N_layers",
            "layers": [{"name": "water", "thickness": 0.001, "rho": 1000.0, "c": 1500.0}],
            "run_postprocess": False,
            "write_vtx": False,
        }
    )
    problem = build_acoustic_problem(cfg)
    integrator = CoupledRK4 if solver_type == "explicit" else CoupledNewmark
    original_step = integrator.step

    def inject_nonfinite(self, t, dt):
        original_step(self, t, dt)
        if comm.rank == 0:
            assert self.p.getLocalSize() > 0
            self.p.array[0] = np.nan

    integrator.step = inject_nonfinite
    caught = False
    try:
        # No mpi_fail_fast wrapper: the public numerical API itself must let
        # every caller catch the error, then continue using the communicator.
        try:
            solve_acoustic(problem, solver_type=solver_type)
        except FloatingPointError as exc:
            assert "Non-finite coupled acoustic state at step 1" in str(exc)
            caught = True
        assert all(comm.allgather(caught)), "Not every MPI rank received the divergence error"
        if comm.rank == 0:
            print("acoustic divergence caught collectively", flush=True)
    finally:
        integrator.step = original_step
        problem.destroy()


@pytest.mark.parametrize("solver_type", ["explicit", "implicit"])
def test_acoustic_public_solver_handles_rank_local_nonfinite_state(solver_type):
    if MPI.COMM_WORLD.size > 1:
        pytest.skip("Subprocess MPI tests must run from a single-rank pytest process")
    launcher = shutil.which("mpiexec") or str(Path(sys.executable).with_name("mpiexec"))
    if not Path(launcher).is_file():
        pytest.skip("MPI launcher unavailable")
    proc = subprocess.Popen(
        [launcher, "-n", "2", sys.executable, str(Path(__file__)), "--mpi-worker", solver_type],
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
        pytest.fail(f"MPI acoustic solver hung after rank-local divergence:\n{stdout}\n{stderr}")
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    assert proc.returncode == 0, stdout + stderr
    assert "acoustic divergence caught collectively" in stdout, stdout + stderr


if __name__ == "__main__":
    assert sys.argv[1] == "--mpi-worker"
    _rank_local_nonfinite_worker(sys.argv[2])
