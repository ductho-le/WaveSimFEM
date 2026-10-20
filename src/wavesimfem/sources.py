"""Excitation time functions.

Two waveforms are provided:

``ricker``
    Broadband Ricker (Mexican-hat) wavelet; best for 2D-FFT dispersion extraction
    because it excites a wide band around ``f0`` in a single short pulse.
``tone_burst``
    Hann-windowed ``N``-cycle sine burst, the NDE/SHM standard for narrowband,
    mode-selective excitation (relative -6 dB bandwidth ≈ 2/N).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

#: Supported ``excitation.source_type`` values.
SOURCE_TYPES = ("ricker", "tone_burst")
#: Default cycle count of the Hann tone burst.
DEFAULT_N_CYCLES = 5


def _scalar_or_array(value: np.ndarray) -> np.ndarray | float:
    return float(value) if np.ndim(value) == 0 else value


def _check_f0(f0: float) -> float:
    val = float(f0)
    if not np.isfinite(val) or val <= 0.0:
        raise ValueError(f"f0 must be finite and > 0, got {f0!r}")
    return val


def _resolve_n_cycles(n_cycles: int | None) -> int:
    if n_cycles is None:
        return DEFAULT_N_CYCLES
    if (
        isinstance(n_cycles, (bool, np.bool_))
        or not isinstance(n_cycles, (int, float, np.integer, np.floating))
        or not np.isfinite(n_cycles)
        or n_cycles < 1
        or int(n_cycles) != n_cycles
    ):
        raise ValueError(f"n_cycles must be >= 1 and an integer, got {n_cycles!r}")
    return int(n_cycles)


def ricker(t: ArrayLike, f0: float, t0: float) -> np.ndarray | float:
    """Ricker (Mexican-hat) wavelet ``(1 - 2a) exp(-a)``, ``a = (π f0 (t - t0))²``.

    Accepts scalars or arrays; returns a ``float`` for scalar input.
    """
    f0_val = _check_f0(f0)
    a = (np.pi * f0_val * (np.asarray(t, dtype=float) - t0)) ** 2
    return _scalar_or_array((1.0 - 2.0 * a) * np.exp(-a))


def ricker_delay(f0: float) -> float:
    """Default wavelet delay ``1.5 / f0`` so the pulse starts near zero amplitude."""
    return 1.5 / _check_f0(f0)


def tone_burst(t: ArrayLike, f0: float, n_cycles: int = DEFAULT_N_CYCLES) -> np.ndarray | float:
    """Hann-windowed tone burst ``½(1 - cos(2π f0 t / N)) · sin(2π f0 t)`` on ``[0, N/f0]``.

    Zero outside the window, so it starts and ends with zero value and slope.
    Accepts scalars or arrays; returns a ``float`` for scalar input.
    """
    f0_val = _check_f0(f0)
    n = _resolve_n_cycles(n_cycles)
    t_arr = np.asarray(t, dtype=float)
    phase = 2.0 * np.pi * f0_val * t_arr
    window = 0.5 * (1.0 - np.cos(phase / n))
    inside = (t_arr >= 0.0) & (t_arr <= n / f0_val)
    return _scalar_or_array(np.where(inside, window * np.sin(phase), 0.0))


def _check_type(source_type: str) -> None:
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"Unknown source_type '{source_type}'. Choose from: {SOURCE_TYPES}")


def source_signal(
    t: ArrayLike, source_type: str, f0: float, n_cycles: int | None = None
) -> np.ndarray | float:
    """Unit-amplitude excitation of the given type evaluated at ``t``."""
    _check_type(source_type)
    if source_type == "tone_burst":
        return tone_burst(t, f0, _resolve_n_cycles(n_cycles))
    return ricker(t, f0, ricker_delay(f0))


def source_delay(source_type: str, f0: float, n_cycles: int | None = None) -> float:
    """Time of the pulse centre (used for arrival-time guides)."""
    _check_type(source_type)
    f0_val = _check_f0(f0)
    if source_type == "tone_burst":
        return 0.5 * _resolve_n_cycles(n_cycles) / f0_val
    return ricker_delay(f0_val)


def source_duration(source_type: str, f0: float, n_cycles: int | None = None) -> float:
    """Effective pulse length: ``N/f0`` (tone burst) or ``2 · 1.5/f0`` (Ricker)."""
    return 2.0 * source_delay(source_type, f0, n_cycles)


def describe_source(source_type: str, n_cycles: int | None = None) -> str:
    """Human-readable label, e.g. ``'5-cycle Hann tone burst'``."""
    _check_type(source_type)
    if source_type == "tone_burst":
        return f"{_resolve_n_cycles(n_cycles)}-cycle Hann tone burst"
    return "Ricker wavelet"
