"""Scale-aware physical units for console output, figures, and user input.

WaveSimFEM works in SI internally and covers problems from seismic (Hz, m, s)
to ultrasonic (MHz, mm, μs) scales. This module picks a readable display unit
for a representative magnitude and parses user-facing quantities such as
``"2 MHz"`` or ``"3 rad/mm"``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class Unit:
    """A display unit; ``unit(si_value)`` returns ``si_value / size``."""

    label: str  # plain-text label for logs, e.g. "μs"
    tex: str  # label valid in both Matplotlib mathtext and LaTeX, e.g. r"$\mu$s"
    size: float  # SI value of one display unit (rad/mm -> 1e3 rad/m)

    def __call__(self, value: float | np.ndarray) -> float | np.ndarray:
        """Convert an SI value (or array) to this unit."""
        return value / self.size


_TIME = (
    Unit("s", "s", 1.0),
    Unit("ms", "ms", 1e-3),
    Unit("μs", r"$\mu$s", 1e-6),
    Unit("ns", "ns", 1e-9),
)
_LENGTH = (
    Unit("km", "km", 1e3),
    Unit("m", "m", 1.0),
    Unit("mm", "mm", 1e-3),
    Unit("μm", r"$\mu$m", 1e-6),
)
_DISPLACEMENT = (
    Unit("m", "m", 1.0),
    Unit("mm", "mm", 1e-3),
    Unit("μm", r"$\mu$m", 1e-6),
    Unit("nm", "nm", 1e-9),
    Unit("pm", "pm", 1e-12),
)
_FREQUENCY = (
    Unit("GHz", "GHz", 1e9),
    Unit("MHz", "MHz", 1e6),
    Unit("kHz", "kHz", 1e3),
    Unit("Hz", "Hz", 1.0),
)


def _pick(magnitude: float, table: tuple[Unit, ...]) -> Unit:
    """Largest unit not exceeding ``|magnitude|`` (smallest unit for tiny values)."""
    magnitude = abs(float(magnitude))
    if not np.isfinite(magnitude) or magnitude == 0.0:
        return next(u for u in table if u.size == 1.0)
    for unit in table:
        if magnitude >= unit.size * (1.0 - 1e-12):
            return unit
    return table[-1]


def time_unit(seconds: float) -> Unit:
    """Display unit for a representative time span."""
    return _pick(seconds, _TIME)


def length_unit(meters: float) -> Unit:
    """Display unit for a representative length."""
    return _pick(meters, _LENGTH)


def displacement_unit(meters: float) -> Unit:
    """Display unit for a representative displacement amplitude."""
    return _pick(meters, _DISPLACEMENT)


def observable_unit(magnitude: float, observable: str = "displacement") -> Unit:
    """Amplitude display units for explicitly identified physical observables."""
    if observable == "pressure":
        return _pick(
            magnitude,
            (
                Unit("MPa", "MPa", 1e6),
                Unit("kPa", "kPa", 1e3),
                Unit("Pa", "Pa", 1.0),
                Unit("mPa", "mPa", 1e-3),
            ),
        )
    if observable in ("velocity", "particle_velocity"):
        return _pick(
            magnitude,
            (
                Unit("m/s", "m/s", 1.0),
                Unit("mm/s", "mm/s", 1e-3),
                Unit("μm/s", r"$\mu$m/s", 1e-6),
                Unit("nm/s", "nm/s", 1e-9),
            ),
        )
    if observable != "displacement":
        raise ValueError(f"Unknown physical observable {observable!r}")
    return displacement_unit(magnitude)


def frequency_unit(hertz: float) -> Unit:
    """Display unit for a representative frequency."""
    return _pick(hertz, _FREQUENCY)


def wavenumber_unit(length: Unit) -> Unit:
    """Angular wavenumber unit ``rad/<length>`` paired with a length unit."""
    return Unit(f"rad/{length.label}", f"rad/{length.tex}", 1.0 / length.size)


def _format(value: float, unit: Unit, digits: int) -> str:
    return f"{unit(value):.{digits}g} {unit.label}"


def fmt_time(seconds: float, digits: int = 4) -> str:
    """``0.00012`` -> ``'120 μs'``."""
    return _format(seconds, time_unit(seconds), digits)


def fmt_length(meters: float, digits: int = 4) -> str:
    """``0.002`` -> ``'2 mm'``."""
    return _format(meters, length_unit(meters), digits)


def fmt_frequency(hertz: float, digits: int = 4) -> str:
    """``5e5`` -> ``'500 kHz'``."""
    return _format(hertz, frequency_unit(hertz), digits)


# ---------------------------------------------------------------------------
# Quantity parsing
# ---------------------------------------------------------------------------
QuantityKind = Literal["frequency", "wavenumber"]

_PARSE_UNITS: dict[str, dict[str, float]] = {
    "frequency": {"Hz": 1.0, "mHz": 1e-3, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9},
    "wavenumber": {"rad/m": 1.0, "rad/cm": 1e2, "rad/mm": 1e3, "rad/km": 1e-3},
}
_QUANTITY = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([A-Za-z/]*)\s*$")


def parse_quantity(value: object, kind: QuantityKind) -> float:
    """Parse a number (SI: Hz, rad/m) or a string with a unit (``"2 MHz"``, ``"3 rad/mm"``).

    Raises
    ------
    ValueError
        For booleans, non-finite values, malformed strings, or unknown units.
    """
    units = _PARSE_UNITS[kind]
    if isinstance(value, bool):
        raise ValueError(f"expected a {kind}, got a boolean")
    if isinstance(value, (int, float, np.integer, np.floating)):
        out = float(value)
    elif isinstance(value, str):
        match = _QUANTITY.match(value)
        if not match:
            raise ValueError(f"cannot parse {kind} {value!r}; e.g. '{_example(kind)}'")
        number, unit = match.groups()
        # Accept case variants of the unit name, but never fold SI prefixes.
        unit_key = unit[:-2] + "Hz" if kind == "frequency" and unit[-2:].lower() == "hz" else unit
        if unit_key and unit_key not in units:
            choices = ", ".join(_display_units(kind))
            raise ValueError(f"unknown {kind} unit {unit!r} in {value!r}; use one of: {choices}")
        out = float(number) * (units[unit_key] if unit_key else 1.0)
    else:
        raise ValueError(f"expected a {kind} number or string, got {type(value).__name__}")
    if not np.isfinite(out):
        raise ValueError(f"{kind} must be finite, got {value!r}")
    return out


def _display_units(kind: QuantityKind) -> list[str]:
    return list(_PARSE_UNITS[kind])


def _example(kind: QuantityKind) -> str:
    return "2 MHz" if kind == "frequency" else "3 rad/mm"
