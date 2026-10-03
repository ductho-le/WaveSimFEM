"""Thread-placement policy tests (pure Python; no MPI launch required)."""

from __future__ import annotations

import subprocess
import sys

import pytest

from guidedwavefem.parallel import (
    THREAD_ENV_VARS,
    describe_thread_state,
    limit_threads_per_rank,
    ranks_per_node,
)


@pytest.mark.parametrize(
    "env,expected",
    [
        ({}, 1),
        ({"OMPI_COMM_WORLD_LOCAL_SIZE": "4", "OMPI_COMM_WORLD_SIZE": "8"}, 4),
        ({"MPI_LOCALNRANKS": "6", "PMI_SIZE": "12"}, 6),
        ({"PMI_SIZE": "3"}, 3),
        ({"SLURM_STEP_TASKS_PER_NODE": "16(x4)"}, 16),
        ({"SLURM_STEP_TASKS_PER_NODE": "8,7"}, 8),
        ({"MPI_LOCALNRANKS": "garbage", "PMI_SIZE": "2"}, 2),
        ({"SLURM_NTASKS": "64"}, 1),  # allocation size alone is not a launch
    ],
)
def test_ranks_per_node(env, expected):
    assert ranks_per_node(env) == expected


def test_policy_divides_cpus_between_local_ranks():
    env = {"MPI_LOCALNRANKS": "4"}
    policy = limit_threads_per_rank(env, cpus=11)
    assert policy.applied and policy.threads_per_rank == 2 and policy.ranks_per_node == 4
    assert all(env[name] == "2" for name in THREAD_ENV_VARS)
    assert "2 BLAS/OpenMP thread(s) per rank" in policy.describe()


def test_policy_never_drops_below_one_thread():
    env = {"OMPI_COMM_WORLD_LOCAL_SIZE": "32"}
    assert limit_threads_per_rank(env, cpus=8).threads_per_rank == 1


def test_policy_respects_user_setting():
    env = {"MPI_LOCALNRANKS": "4", "OMP_NUM_THREADS": "3"}
    policy = limit_threads_per_rank(env, cpus=16)
    assert not policy.applied and "OMP_NUM_THREADS" in policy.reason
    assert env == {"MPI_LOCALNRANKS": "4", "OMP_NUM_THREADS": "3"}


def test_policy_is_noop_for_single_rank():
    env: dict[str, str] = {}
    policy = limit_threads_per_rank(env, cpus=16)
    assert not policy.applied and env == {}


def test_parallel_module_is_dependency_free():
    """The policy must load before NumPy/PETSc, so importing it must not pull them in."""
    code = (
        "import sys, guidedwavefem.parallel, guidedwavefem._launcher; "
        "heavy = {'numpy', 'petsc4py', 'mpi4py', 'dolfinx'} & set(sys.modules); "
        "sys.exit(1 if heavy else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_describe_thread_state_without_policy():
    assert describe_thread_state(1, {}) == "library defaults"
    assert "oversubscribe" in describe_thread_state(4, {})
    assert "OMP_NUM_THREADS=2" in describe_thread_state(4, {"OMP_NUM_THREADS": "2"})
