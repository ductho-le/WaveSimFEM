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
from guidedwavefem.device import DeviceInfo, detect_available_devices, select_device
from guidedwavefem.numerics import (
    PointSensors,
    dispersion_spectrum,
    operator_bound,
    rk4_step,
    time_grid,
    validate_tensor,
)
from guidedwavefem.solvers import run_simulation
from guidedwavefem.solvers_gpu import (
    build_sensor_interpolation_matrix,
    solve_explicit_rk4_device,
)

__all__ = [
    "__version__",
    "DeviceInfo",
    "PointSensors",
    "build_sensor_interpolation_matrix",
    "compute_rayleigh_speed",
    "detect_available_devices",
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
    "select_device",
    "solve_explicit_rk4_device",
    "time_grid",
    "validate_tensor",
]
