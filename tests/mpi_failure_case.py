"""Fault injection driver used by MPI subprocess regressions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wavesimfem import cli

case = sys.argv[1]
if case == "config":
    original = cli.parse_and_resolve_config

    def read_config(*args, **kwargs):
        if cli.rank == 0:
            raise OSError("injected rank-local configuration read failure")
        return original(*args, **kwargs)

    cli.parse_and_resolve_config = read_config
    cli.main(["-c", sys.argv[2], "--dry-run", "-q"])
elif case in ("help", "version"):
    cli.main([f"--{case}"])
elif case.startswith("gather_"):
    import numpy as np

    from wavesimfem.numerics import PointSensors

    probes = object.__new__(PointSensors)
    probes.comm = cli.comm
    probes.indices = np.array([cli.rank])
    probes.count = cli.comm.size
    local = np.zeros((4, 1))
    output = None
    chunk_bytes = 64
    if cli.rank == 0:
        if case == "gather_output":
            output = np.zeros((3, cli.comm.size))
        elif case == "gather_columns":
            local = np.zeros((4, 2))
        elif case == "gather_times":
            local = np.zeros((3, 1))
        elif case == "gather_dtype":
            local = local.astype(np.float32)
        elif case == "gather_budget":
            chunk_bytes = 16
    # Deliberately call the helper without mpi_fail_fast: validation itself must
    # complete collectively, even when only one rank has an invalid argument.
    caught = False
    try:
        probes.gather(local, out=output, chunk_bytes=chunk_bytes)
    except ValueError:
        caught = True
    assert all(cli.comm.allgather(caught))
    if cli.rank == 0:
        print("collective validation completed")
else:

    @cli.mpi_fail_fast
    def fail():
        if cli.rank == 0:
            if case == "interrupt":
                raise KeyboardInterrupt("injected rank-local interruption")
            if case == "exit":
                raise SystemExit("injected unsuccessful exit")
            raise RuntimeError("injected rank-local exception")
        cli.comm.Barrier()

    fail()
