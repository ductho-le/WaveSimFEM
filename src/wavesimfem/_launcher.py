"""Console entry point that fixes the thread policy before heavy imports.

BLAS/OpenMP thread pools are sized when NumPy and PETSc load, so the
per-rank thread cap from :mod:`wavesimfem.parallel` must be applied
before :mod:`wavesimfem.cli` (which imports them) is imported.
"""

from __future__ import annotations

import sys

_FEM_RUNTIME_MODULES = frozenset({"dolfinx", "mpi4py", "petsc4py", "ufl"})


def main() -> None:
    """Apply the MPI thread policy, then run the ``wavesim`` CLI."""
    from wavesimfem.parallel import apply_thread_policy

    apply_thread_policy()

    try:
        from wavesimfem.cli import main as cli_main
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".", 1)[0]
        if missing in _FEM_RUNTIME_MODULES:
            print(
                f"wavesim: error: missing FEM runtime dependency '{missing}'.\n"
                "DOLFINx, PETSc, petsc4py, and mpi4py must be installed from conda-forge:\n"
                "    conda env create -f environment.yml && conda activate wavesim\n"
                "    pip install -e .",
                file=sys.stderr,
            )
            sys.exit(1)
        raise

    cli_main()
