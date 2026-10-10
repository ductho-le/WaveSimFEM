#!/bin/sh
# Four MPI ranks, one numerical-library thread per rank, using the active environment.
# Usage (repository root): sh tools/run_cpu4.sh -c configs/template.yaml
# All arguments are forwarded to wavesim; CPU/double are enforced last.
set -eu

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec mpiexec -n 4 python -m wavesimfem "$@" --device cpu --precision double
