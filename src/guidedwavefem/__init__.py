"""GuidedWaveFEM: Unified Guided & Surface Wave Finite Element Framework.

Public symbols are resolved lazily so that ``import guidedwavefem`` (and
lightweight modules such as :mod:`guidedwavefem.config`) do not pull in
DOLFINx, PETSc, or MPI until a FEM feature is actually used.
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

try:
    __version__ = version("guidedwavefem")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0.0.0+unknown"

# Public name -> defining submodule
_EXPORTS: dict[str, str] = {
    # analysis
    "plot_full_analysis": "analysis",
    "plot_quick_overview": "analysis",
    "bandpass_filter": "analysis",
    # config
    "dump_config": "config",
    "flatten_config": "config",
    "generate_template_config": "config",
    "load_config": "config",
    "parse_and_resolve_config": "config",
    "resolve_and_apply_config": "config",
    # device
    "DeviceInfo": "device",
    "DeviceUnavailableError": "device",
    "detect_available_devices": "device",
    "select_device": "device",
    # materials
    "compute_rayleigh_speed": "materials",
    "preflight_estimate": "materials",
    "structure_from_config": "materials",
    # numerics
    "PointSensors": "numerics",
    "dispersion_spectrum": "numerics",
    "operator_bound": "numerics",
    "rk4_step": "numerics",
    "time_grid": "numerics",
    "validate_tensor": "numerics",
    # reporting
    "set_verbosity": "reporting",
    # sources
    "SOURCE_TYPES": "sources",
    "ricker": "sources",
    "source_signal": "sources",
    "tone_burst": "sources",
    # units
    "parse_quantity": "units",
    # solvers
    "run_simulation": "solvers",
    "build_problem": "solvers",
    "solve_explicit": "solvers",
    "solve_implicit": "solvers",
    # solvers_gpu
    "build_sensor_interpolation_matrix": "solvers_gpu",
    "integrate_explicit_rk4": "solvers_gpu",
    "solve_explicit_rk4_device": "solvers_gpu",
}

__all__ = [
    "SOURCE_TYPES",
    "DeviceInfo",
    "DeviceUnavailableError",
    "PointSensors",
    "__version__",
    "bandpass_filter",
    "build_problem",
    "build_sensor_interpolation_matrix",
    "compute_rayleigh_speed",
    "detect_available_devices",
    "dispersion_spectrum",
    "dump_config",
    "flatten_config",
    "generate_template_config",
    "integrate_explicit_rk4",
    "load_config",
    "operator_bound",
    "parse_and_resolve_config",
    "parse_quantity",
    "plot_full_analysis",
    "plot_quick_overview",
    "preflight_estimate",
    "resolve_and_apply_config",
    "ricker",
    "rk4_step",
    "run_simulation",
    "select_device",
    "set_verbosity",
    "solve_explicit",
    "solve_explicit_rk4_device",
    "solve_implicit",
    "source_signal",
    "structure_from_config",
    "time_grid",
    "tone_burst",
    "validate_tensor",
]


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:  # pragma: no cover - static analysers only
    from guidedwavefem.analysis import bandpass_filter, plot_full_analysis, plot_quick_overview
    from guidedwavefem.config import (
        dump_config,
        flatten_config,
        generate_template_config,
        load_config,
        parse_and_resolve_config,
        resolve_and_apply_config,
    )
    from guidedwavefem.device import (
        DeviceInfo,
        DeviceUnavailableError,
        detect_available_devices,
        select_device,
    )
    from guidedwavefem.materials import (
        compute_rayleigh_speed,
        preflight_estimate,
        structure_from_config,
    )
    from guidedwavefem.numerics import (
        PointSensors,
        dispersion_spectrum,
        operator_bound,
        rk4_step,
        time_grid,
        validate_tensor,
    )
    from guidedwavefem.reporting import set_verbosity
    from guidedwavefem.solvers import build_problem, run_simulation, solve_explicit, solve_implicit
    from guidedwavefem.solvers_gpu import (
        build_sensor_interpolation_matrix,
        integrate_explicit_rk4,
        solve_explicit_rk4_device,
    )
    from guidedwavefem.sources import SOURCE_TYPES, ricker, source_signal, tone_burst
    from guidedwavefem.units import parse_quantity
