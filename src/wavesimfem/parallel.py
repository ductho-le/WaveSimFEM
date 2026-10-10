"""Thread-placement policy for MPI runs.

WaveSimFEM parallelises with MPI: one rank per core. Threaded BLAS/OpenMP
libraries loaded by NumPy, SciPy, and PETSc default to one thread *per core*
in *every* rank, so ``mpiexec -n 4`` on an 11-core machine can run 40+ threads
that contend for the same cores (measured: ~1000 % CPU for 4 ranks with no
speed-up). :func:`limit_threads_per_rank` caps each rank at
``max(1, cpus // ranks_per_node)`` threads.

The policy must be applied **before** NumPy/PETSc are imported, because the
thread pools are sized when the shared libraries load. It is therefore kept
free of third-party imports and invoked by :mod:`wavesimfem._launcher`.

The policy never overrides a user's choice: if any of :data:`THREAD_ENV_VARS`
is already set, nothing is changed. It is a no-op for single-rank runs.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass

#: Environment variables that size the common BLAS/OpenMP thread pools.
THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# Ranks on *this* node, as exported by the common launchers (first match wins).
_LOCAL_SIZE_VARS: tuple[str, ...] = (
    "OMPI_COMM_WORLD_LOCAL_SIZE",  # Open MPI
    "MPI_LOCALNRANKS",  # MPICH / Hydra, Intel MPI
    "MV2_COMM_WORLD_LOCAL_SIZE",  # MVAPICH2
    "SLURM_STEP_TASKS_PER_NODE",  # srun step, e.g. "4(x2)" or "4,3"
    "SLURM_TASKS_PER_NODE",
)
# Total ranks; used only when no per-node count is available (assumes one node).
_WORLD_SIZE_VARS: tuple[str, ...] = (
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "PMIX_SIZE",
    "MV2_COMM_WORLD_SIZE",
    "SLURM_STEP_NUM_TASKS",
)

_LEADING_INT = re.compile(r"\s*(\d+)")


@dataclass(frozen=True)
class ThreadPolicy:
    """Outcome of :func:`limit_threads_per_rank` (reported in the run banner)."""

    ranks_per_node: int
    threads_per_rank: int | None
    applied: bool
    reason: str

    def describe(self) -> str:
        """One-line, human-readable summary."""
        if self.applied:
            return (
                f"{self.threads_per_rank} BLAS/OpenMP thread(s) per rank "
                f"({self.ranks_per_node} rank(s) on this node)"
            )
        return self.reason


def _leading_int(value: str | None) -> int | None:
    if not value:
        return None
    match = _LEADING_INT.match(value)
    return int(match.group(1)) if match else None


def ranks_per_node(environ: Mapping[str, str] | None = None) -> int:
    """Number of MPI ranks sharing this node, inferred from launcher variables.

    Returns ``1`` when the process was not started by a recognised launcher.
    """
    env = os.environ if environ is None else environ
    for name in (*_LOCAL_SIZE_VARS, *_WORLD_SIZE_VARS):
        if name in ("SLURM_STEP_TASKS_PER_NODE", "SLURM_TASKS_PER_NODE"):
            count = _slurm_local_tasks(env.get(name), env.get("SLURM_NODEID", "0"))
        else:
            count = _leading_int(env.get(name))
        if count is not None and count > 0:
            return count
    return 1


def _slurm_local_tasks(value: str | None, node_id: str) -> int | None:
    """Select a node without materializing potentially large repetition lists."""
    if not value or not node_id.isdecimal():
        return None
    node = int(node_id)
    selected = None
    for entry in value.split(","):
        match = re.fullmatch(r"\s*(\d+)(?:\(x(\d+)\))?\s*", entry)
        if match is None:
            return None
        count, repeat = int(match[1]), int(match[2] or 1)
        if count < 1 or repeat < 1:
            return None
        if 0 <= node < repeat:
            selected = count
        node -= repeat
    return selected


def available_cpus() -> int:
    """CPUs this process may run on (affinity-aware where the OS supports it)."""
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            return max(1, len(affinity(0)))
        except OSError:  # pragma: no cover - exotic platforms
            pass
    return max(1, os.cpu_count() or 1)


def limit_threads_per_rank(
    environ: MutableMapping[str, str] | None = None, *, cpus: int | None = None
) -> ThreadPolicy:
    """Cap BLAS/OpenMP threads per rank to avoid core oversubscription.

    Sets every variable in :data:`THREAD_ENV_VARS` to
    ``max(1, cpus // ranks_per_node)`` when the run has more than one rank on
    this node and the user has not set any of them. Returns the decision.
    """
    env = os.environ if environ is None else environ
    ranks = ranks_per_node(env)
    user_set = [name for name in THREAD_ENV_VARS if env.get(name)]
    if user_set:
        return ThreadPolicy(ranks, None, False, f"thread count left to user ({user_set[0]} set)")
    if ranks <= 1:
        return ThreadPolicy(ranks, None, False, "single rank; library thread defaults kept")
    threads = max(1, (cpus if cpus is not None else available_cpus()) // ranks)
    for name in THREAD_ENV_VARS:
        env[name] = str(threads)
    return ThreadPolicy(ranks, threads, True, "applied")


_policy: ThreadPolicy | None = None


def apply_thread_policy() -> ThreadPolicy:
    """Apply :func:`limit_threads_per_rank` to this process once and remember it."""
    global _policy
    if _policy is None:
        _policy = limit_threads_per_rank()
    return _policy


def current_thread_policy() -> ThreadPolicy | None:
    """The policy applied by :func:`apply_thread_policy`, or ``None`` if never applied."""
    return _policy


def describe_thread_state(n_ranks: int, environ: Mapping[str, str] | None = None) -> str:
    """Banner text for the BLAS/OpenMP thread configuration of this run."""
    if _policy is not None:
        return _policy.describe()
    env = os.environ if environ is None else environ
    user_set = [f"{name}={env[name]}" for name in THREAD_ENV_VARS if env.get(name)]
    if user_set:
        return f"set by user ({', '.join(user_set[:2])})"
    if n_ranks > 1:
        return "library defaults (may oversubscribe cores; set OMP_NUM_THREADS)"
    return "library defaults"
