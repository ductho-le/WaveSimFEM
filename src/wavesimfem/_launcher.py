"""Console entry point that fixes the thread policy before heavy imports.

BLAS/OpenMP thread pools are sized when NumPy and PETSc load, so the
per-rank thread cap from :mod:`wavesimfem.parallel` must be applied
before :mod:`wavesimfem.cli` (which imports them) is imported.
"""

from __future__ import annotations


def main() -> None:
    """Apply the MPI thread policy, then run the ``wavesim`` CLI."""
    from wavesimfem.parallel import apply_thread_policy

    apply_thread_policy()

    from wavesimfem.cli import main as cli_main

    cli_main()
