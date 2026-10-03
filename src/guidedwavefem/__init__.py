"""GuidedWaveFEM: Unified Guided & Surface Wave Finite Element Framework."""

from __future__ import annotations

__version__ = "0.2.0"

from guidedwavefem.analysis import plot_full_analysis
from guidedwavefem.config import (
    dump_config,
    flatten_config,
    generate_template_config,
    load_config,
    parse_and_resolve_config,
    resolve_and_apply_config,
)
from guidedwavefem.materials import compute_rayleigh_speed, preflight_estimate
from guidedwavefem.numerics import (
    PointSensors,
    dispersion_spectrum,
    operator_bound,
    rk4_step,
    time_grid,
    validate_tensor,
)
from guidedwavefem.solvers import run_simulation

__all__ = [
    "__version__",
    "PointSensors",
    "compute_rayleigh_speed",
    "dispersion_spectrum",
    "dump_config",
    "flatten_config",
    "generate_template_config",
    "load_config",
    "operator_bound",
    "parse_and_resolve_config",
    "plot_full_analysis",
    "preflight_estimate",
    "resolve_and_apply_config",
    "rk4_step",
    "run_simulation",
    "time_grid",
    "validate_tensor",
]
