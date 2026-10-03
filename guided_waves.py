"""Backward-compatibility shim for GuidedWaveFEM.

All core functionality has been refactored into the modular `guidedwavefem` package:
  - `guidedwavefem.config`    : Configuration loading, parsing, validation, and serialization
  - `guidedwavefem.materials` : Material models, structure normalization, and preflight estimation
  - `guidedwavefem.numerics`  : RK4 time stepping, operator bounds, exact point sensors, dispersion spectra
  - `guidedwavefem.solvers`   : Implicit Newmark-β and explicit lumped RK4 solvers
  - `guidedwavefem.analysis`  : 2D FFT signal processing and publication figures
  - `guidedwavefem.cli`       : Command-line interface and orchestration

This file maintains 100% backwards compatibility for `python guided_waves.py` CLI usage
and `from guided_waves import ...` imports.
"""

from __future__ import annotations

import sys
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from guidedwavefem import (  # noqa: E402
    PointSensors,
    __version__,
    compute_rayleigh_speed,
    dispersion_spectrum,
    dump_config,
    flatten_config,
    generate_template_config,
    load_config,
    operator_bound,
    parse_and_resolve_config,
    plot_full_analysis,
    preflight_estimate,
    resolve_and_apply_config,
    rk4_step,
    run_simulation,
    time_grid,
    validate_tensor,
)
from guidedwavefem.cli import main  # noqa: E402
from guidedwavefem.materials import (  # noqa: E402
    _canonicalize_mode,
    _format_banner_box,
    _normalize_region,
    _normalize_structure,
    _print_structure_table,
    log,
)
from guidedwavefem.solvers import comm, mpi_fail_fast, nprocs, rank, ricker  # noqa: E402

__all__ = [
    "PointSensors",
    "__version__",
    "_canonicalize_mode",
    "_format_banner_box",
    "_normalize_region",
    "_normalize_structure",
    "_print_structure_table",
    "comm",
    "compute_rayleigh_speed",
    "dispersion_spectrum",
    "dump_config",
    "flatten_config",
    "generate_template_config",
    "load_config",
    "log",
    "main",
    "mpi_fail_fast",
    "nprocs",
    "operator_bound",
    "parse_and_resolve_config",
    "plot_full_analysis",
    "preflight_estimate",
    "rank",
    "resolve_and_apply_config",
    "ricker",
    "rk4_step",
    "run_simulation",
    "time_grid",
    "validate_tensor",
]

if __name__ == "__main__":
    main()
