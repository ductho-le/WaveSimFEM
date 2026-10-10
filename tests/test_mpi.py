import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("solver", ["explicit", "implicit"])
@pytest.mark.parametrize("history_storage", ["memory", "disk"])
def test_serial_mpi_embedded_plate(tmp_path, solver, history_storage):
    launcher = shutil.which("mpiexec") or str(Path(sys.executable).with_name("mpiexec"))
    if not Path(launcher).is_file():
        pytest.skip("MPI launcher unavailable")
    driver = str(Path(__file__).with_name("mpi_case.py"))
    for ranks in (1, 2, 4):
        cmd = [
            launcher,
            "-n",
            str(ranks),
            sys.executable,
            driver,
            str(tmp_path / str(ranks)),
            solver,
            history_storage,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=90)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            stdout = getattr(exc, "stdout", "") or ""
            stderr = getattr(exc, "stderr", "") or ""
            print(
                f"\n[MPI FAILURE] ranks={ranks}, solver={solver}\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{stdout}\n"
                f"STDERR:\n{stderr}\n",
                file=sys.stderr,
            )
            raise
    for ranks in (2, 4):
        with (
            np.load(tmp_path / "1" / "sensor_data.npz") as a,
            np.load(tmp_path / str(ranks) / "sensor_data.npz") as b,
        ):
            np.testing.assert_allclose(a["times"], b["times"], rtol=1e-12, atol=0)
            for key in ("sensor_data_x", "sensor_data_y"):
                assert np.linalg.norm(a[key] - b[key]) / np.linalg.norm(a[key]) < 1e-8
    assert not list(tmp_path.glob("*/.wavesim-history-*"))
