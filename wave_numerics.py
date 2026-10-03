"""Backward-compatibility shim for wave_numerics.

Re-exports numerical routines from `guidedwavefem.numerics`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from guidedwavefem.numerics import (  # noqa: E402
    PointSensors,
    dispersion_spectrum,
    operator_bound,
    rk4_step,
    time_grid,
    validate_tensor,
)

__all__ = [
    "PointSensors",
    "dispersion_spectrum",
    "operator_bound",
    "rk4_step",
    "time_grid",
    "validate_tensor",
]
