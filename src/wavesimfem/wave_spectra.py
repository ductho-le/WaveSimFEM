"""Measured array dispersion and surface polarization without FEM dependencies.

The velocity image uses an amplitude-weighted form of the phase-shift stack
described by Park et al. (1998, SEG Expanded Abstracts, pp. 1377--1380).
It is a spatial coherence image, not modal energy or a mode identification.
The amplitude weighting avoids assigning unit weight to nearly silent traces.
Welch cross spectra use the ``conj(Ux) * Uy`` convention documented by SciPy.

References
----------
https://www.masw.com/DCImagingScheme.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.signal import csd, find_peaks, welch


def _validate_records(
    times: np.ndarray, *records: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray], float]:
    times = np.asarray(times, dtype=float)
    arrays = [np.asarray(record, dtype=float) for record in records]
    if times.ndim != 1 or len(times) < 32:
        raise ValueError("Spectral analysis requires at least 32 time samples")
    spacing = np.diff(times)
    if (
        not np.isfinite(times).all()
        or np.any(spacing <= 0)
        or not np.allclose(spacing, spacing[0], rtol=1e-7, atol=0)
    ):
        raise ValueError("Spectral analysis requires finite, uniform, increasing times")
    for array in arrays:
        if array.ndim != 2 or array.shape[0] != len(times) or array.shape[1] < 3:
            raise ValueError("Histories must have shape (time, receiver), with >=3 receivers")
        if array.shape != arrays[0].shape or not np.isfinite(array).all():
            raise ValueError("Displacement histories must have matching shapes and finite values")
    # Reference subtraction exactly removes a constant record before averaging.
    arrays = [array - array[:1] for array in arrays]
    arrays = [array - array.mean(axis=0) for array in arrays]
    scale = max(float(np.max(np.abs(array))) for array in arrays)
    if scale > 0:
        arrays = [array / scale for array in arrays]
    return times, arrays, float(spacing[0])


def _band_indices(frequencies: np.ndarray, band: Sequence[float] | np.ndarray | None) -> np.ndarray:
    selected = frequencies > 0
    if band is not None:
        bounds = np.asarray(band, dtype=float)
        if (
            bounds.shape != (2,)
            or not np.isfinite(bounds).all()
            or bounds[0] < 0
            or bounds[1] <= bounds[0]
        ):
            raise ValueError("frequency_band must contain two finite increasing nonnegative bounds")
        selected &= (frequencies >= bounds[0]) & (frequencies <= bounds[1])
    indices = np.flatnonzero(selected)
    if not len(indices):
        raise ValueError("frequency_band contains no positive recorded frequency bins")
    return indices


def _spectral_support(power: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an explicitly uncalibrated spectral-contrast proxy and support.

    No quiet/noise-only record is available, so this is deliberately not called
    SNR. The lower-quartile full-band power is a conservative baseline proxy;
    relative power and spatial/component coherence provide separate checks.
    """
    peak = float(np.max(power))
    if peak <= 0:
        return np.full_like(power, -300.0), np.zeros_like(power), np.zeros_like(power, bool)
    normalized = power / peak
    floor = max(float(np.quantile(normalized[1:], 0.25)), 1e-15)
    relative_db = 10 * np.log10(np.maximum(normalized, 1e-30))
    contrast_db = 10 * np.log10(np.maximum(normalized / floor, 1e-30))
    return relative_db, contrast_db, relative_db >= -30


def phase_velocity_spectrum(
    times: np.ndarray,
    positions: np.ndarray,
    values: np.ndarray,
    phase_velocities: Sequence[float] | np.ndarray,
    *,
    frequency_band: Sequence[float] | np.ndarray | None = None,
    direction: int = 1,
    max_frequencies: int = 320,
) -> dict[str, Any]:
    """Return a measured frequency--velocity image and qualified peak samples.

    Input histories have shape ``(time, receiver)``. Coordinates are in seconds
    and metres, trial speeds in m/s; ``direction=1`` selects +x, ``-1`` selects
    -x. The positive-frequency NumPy FFT of ``cos(w*t-k*x)`` has phase ``-k*x``,
    hence the compensating phase is **positive** for +x propagation.

    ``coherence[f,c] = abs(sum(U[f,x]*exp(i*direction*w*x/c)))**2 /
    sum(abs(U[f,x]))**2`` lies in [0,1] without normalizing each frequency to
    its strongest trial. This is an amplitude-weighted phase-coherence stack,
    not a statistical coherence estimator. The full image is returned together
    with a mask excluding unsupported power, wavelengths below 2*max(dx), and
    wavelengths above the aperture. Irregular receiver spacing is supported;
    max(dx) supplies a conservative sampling screen, not an exact alias theorem.

    A ridge sample needs >=4 effective receivers, coherence >=0.65, an interior
    isolated peak, and >=2 wavelengths across the amplitude-weighted aperture.
    Comparable competing peaks are rejected. These deterministic checks are
    not mode labels, a continuous branch tracker, or confidence intervals.
    ``velocity_resolution`` is the first-order aperture scale c²/(f*L_eff).
    Low spectral contrast (<6 dB) raises the coherence requirement to 0.9;
    it does not exclude genuinely broadband coherent signals by itself.

    Computation allocates at most a velocity-block-by-receiver steering matrix.
    If frequencies are reduced, each interval retains its strongest physical FFT
    bin; no interpolated bins or zero-padding are represented as extra resolution.
    """
    times, (values,), dt = _validate_records(times, values)
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(phase_velocities, dtype=float)
    if (
        positions.ndim != 1
        or len(positions) != values.shape[1]
        or not np.isfinite(positions).all()
        or np.any(np.diff(positions) <= 0)
    ):
        raise ValueError("Receiver positions must be finite, increasing and match the histories")
    if (
        velocities.ndim != 1
        or not 8 <= len(velocities) <= 2048
        or not np.isfinite(velocities).all()
        or np.any(velocities <= 0)
        or np.any(np.diff(velocities) <= 0)
    ):
        raise ValueError("Provide 8--2048 finite, positive, increasing trial phase velocities")
    if isinstance(direction, bool) or direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    if (
        isinstance(max_frequencies, bool)
        or not isinstance(max_frequencies, int)
        or not 2 <= max_frequencies <= 2048
    ):
        raise ValueError("max_frequencies must be an integer from 2 to 2048")
    nt, nx = values.shape
    offsets = positions - positions[0]
    aperture = float(offsets[-1])
    maximum_gap = float(np.max(np.diff(positions)))
    transform = np.fft.rfft(values * np.hanning(nt)[:, None], axis=0)
    all_frequencies = np.fft.rfftfreq(nt, dt)
    full_power = np.mean(np.abs(transform) ** 2, axis=1)
    relative_db, contrast_db, frequency_supported = _spectral_support(full_power)
    selected = _band_indices(all_frequencies, frequency_band)
    # Real-valued Nyquist has no independent quadrature phase for direction.
    selected = selected[all_frequencies[selected] < 0.5 / dt * (1 - 1e-12)]
    if not len(selected):
        raise ValueError("frequency_band contains no bins below the temporal Nyquist limit")
    original_count = len(selected)
    if original_count > max_frequencies:
        selected = np.array(
            [
                group[np.argmax(full_power[group])]
                for group in np.array_split(selected, max_frequencies)
            ]
        )
    frequencies = all_frequencies[selected]
    transform = transform[selected]
    amplitudes = np.abs(transform)
    amplitude_sum = amplitudes.sum(axis=1)
    power_sum = np.sum(amplitudes**2, axis=1)
    effective_receivers = np.divide(
        amplitude_sum**2, power_sum, out=np.zeros(len(selected)), where=power_sum > 0
    )
    weights = np.divide(
        amplitudes,
        amplitude_sum[:, None],
        out=np.zeros_like(amplitudes),
        where=amplitude_sum[:, None] > 0,
    )
    centroid = weights @ offsets
    effective_aperture = np.minimum(
        aperture, np.sqrt(12 * np.sum(weights * (offsets - centroid[:, None]) ** 2, axis=1))
    )
    coherence = np.zeros((len(selected), len(velocities)))
    for row, frequency in enumerate(frequencies):
        if amplitude_sum[row] <= 0:
            continue
        for first in range(0, len(velocities), 128):
            last = min(first + 128, len(velocities))
            steering = np.exp(
                2j * np.pi * direction * frequency * offsets[None, :] / velocities[first:last, None]
            )
            coherence[row, first:last] = np.abs(steering @ transform[row] / amplitude_sum[row]) ** 2
    np.clip(coherence, 0, 1, out=coherence)
    alias_min = 2 * frequencies * maximum_gap
    aperture_max = frequencies * aperture
    support = (
        frequency_supported[selected, None]
        & (velocities[None, :] > alias_min[:, None])
        & (velocities[None, :] < aperture_max[:, None])
        & (effective_receivers[:, None] >= 4 - 1e-8)
    )
    candidate = np.full(len(selected), np.nan)
    ridge = np.full(len(selected), np.nan)
    resolution = np.full(len(selected), np.nan)
    peak_coherence = np.zeros(len(selected))
    reasons = []
    for row, frequency in enumerate(frequencies):
        spatial = (velocities > alias_min[row]) & (velocities < aperture_max[row])
        allowed = np.flatnonzero(spatial)
        reason = "accepted"
        if not frequency_supported[selected[row]]:
            reason = "weak spectral support"
        elif effective_receivers[row] < 4 - 1e-8:
            reason = "fewer than four effective receivers"
        elif len(allowed) < 3:
            reason = "outside spatial sampling and aperture limits"
        else:
            index = int(allowed[np.argmax(coherence[row, allowed])])
            candidate[row] = velocities[index]
            peak_coherence[row] = coherence[row, index]
            resolution[row] = velocities[index] ** 2 / (frequency * effective_aperture[row])
            # Peaks in the complete physically admissible interval, including
            # its boundaries, prevent a clipped second branch being hidden.
            local = coherence[row, allowed]
            local_index = int(np.argmax(local))
            peaks, _ = find_peaks(local, prominence=0.08)
            peaks = list(peaks)
            if local[0] > local[1] and (
                local_index == 0 or local[0] - float(np.min(local[: local_index + 1])) >= 0.08
            ):
                peaks.append(0)
            if local[-1] > local[-2] and (
                local_index == len(local) - 1
                or local[-1] - float(np.min(local[local_index:])) >= 0.08
            ):
                peaks.append(len(local) - 1)
            strong = [p for p in peaks if local[p] >= 0.5 * peak_coherence[row]]
            if len(strong) > 1:
                reason = "competing velocity peaks"
            elif index in (allowed[0], allowed[-1]):
                reason = "peak at scan or sampling boundary"
            elif peak_coherence[row] < (0.65 if contrast_db[selected[row]] >= 6 else 0.9):
                reason = "low spatial coherence"
            elif resolution[row] / velocities[index] > 0.5:
                reason = "fewer than two wavelengths across effective aperture"
            else:
                ridge[row] = candidate[row]
        reasons.append(reason)
    valid = np.isfinite(ridge)
    diagnostics = {
        "method": "amplitude-weighted phase-coherence stack",
        "normalization": "absolute squared coherent sum / squared sum of amplitudes; range 0 to 1",
        "direction": "+x" if direction == 1 else "-x",
        "receiver_count": nx,
        "aperture_m": aperture,
        "maximum_receiver_gap_m": maximum_gap,
        "temporal_nyquist_hz": 0.5 / dt,
        "frequency_bin_hz": 1 / (nt * dt),
        "hann_equivalent_noise_bandwidth_hz": float(
            nt * np.sum(np.hanning(nt) ** 2) / np.sum(np.hanning(nt)) ** 2 / (nt * dt)
        ),
        "spatial_resolution_rad_per_m": float(2 * np.pi / aperture),
        "requested_band_bin_count": original_count,
        "evaluated_frequency_count": len(selected),
        "frequency_reduction": "strongest physical FFT bin in each contiguous group",
        "accepted_ridge_count": int(np.count_nonzero(valid)),
        "ridge_reason_counts": dict(Counter(reasons)),
        "support_rule": "power >= -30 dB of peak; wavelength >2*max(dx) and <aperture; >=4 effective receivers",
        "ridge_rule": "isolated interior peak; coherence >=0.65 (>=0.9 when spectral contrast <6 dB); no second peak >=50% of first; >=2 wavelengths across effective aperture",
        "spectral_contrast_caveat": "Lower-quartile spectral power is a baseline proxy, not a noise-only measurement or calibrated SNR.",
        "sampling_caveat": "The sampling mask screens trial wavelengths; already aliased input can still produce apparent in-band peaks. Receiver-spacing convergence is required to exclude this.",
        "interpretation": "Qualified dominant peak samples only; no automatic wave-family assignment, modal energy, statistical confidence or continuous branch tracking.",
    }
    return {
        "frequencies": frequencies,
        "phase_velocities": velocities,
        "coherence": coherence,
        "support_mask": support,
        "frequency_supported": frequency_supported[selected],
        "relative_power_db": relative_db[selected],
        "spectral_contrast_db": contrast_db[selected],
        "alias_min_velocity": alias_min,
        "aperture_max_velocity": aperture_max,
        "effective_receivers": effective_receivers,
        "effective_aperture": effective_aperture,
        "candidate_velocity": candidate,
        "ridge_velocity": ridge,
        "ridge_valid": valid,
        "ridge_reason": np.asarray(reasons),
        "peak_coherence": peak_coherence,
        "velocity_resolution": resolution,
        "diagnostics": diagnostics,
    }


def polarization_spectrum(
    times: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    *,
    frequency_band: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure surface V/H displacement ratio and phase of vertical vs horizontal.

    Welch spectra (Hann, 50% overlap, >=3 segments) are averaged across receivers
    *before* calculating magnitude-squared coherence. This measures common
    component polarization across the selected array, not mode-specific
    ellipticity. Multiple wave families can reduce coherence or bias the ratio.

    Both components must exceed 1e-6 of the peak combined spectral power, have
    combined frequency support, and have coherence >=0.8 (>=0.9 when spectral
    contrast is below 6 dB). Rejected ratios and
    phases are NaN, including purely horizontal or purely vertical records.
    Phase is ``arg(conj(Ux)*Uy)`` in degrees, in the supplied global coordinates.
    """
    times, (ux, uy), dt = _validate_records(times, ux, uy)
    nt, nx = ux.shape
    segment = max(16, nt // 3)
    overlap = segment // 2
    options = dict(fs=1 / dt, window="hann", nperseg=segment, noverlap=overlap, axis=0)
    frequencies, pxx = welch(ux, **options)
    _, pyy = welch(uy, **options)
    _, pxy = csd(ux, uy, **options)
    pxx, pyy, pxy = [a.mean(axis=1) for a in (pxx, pyy, pxy)]
    total = pxx + pyy
    relative_db, contrast_db, supported = _spectral_support(total)
    component_floor = max(float(np.max(total)) * 1e-6, np.finfo(float).tiny)
    component_support = (pxx > component_floor) & (pyy > component_floor)
    denominator = pxx * pyy
    coherence = np.divide(
        np.abs(pxy) ** 2, denominator, out=np.zeros_like(pxx), where=denominator > 0
    )
    np.clip(coherence, 0, 1, out=coherence)
    threshold = np.where(contrast_db >= 6, 0.8, 0.9)
    valid = supported & component_support & (coherence >= threshold) & (frequencies > 0)
    ratio = np.full(len(frequencies), np.nan)
    phase = np.full(len(frequencies), np.nan)
    ratio[valid] = np.sqrt(pyy[valid] / pxx[valid])
    phase[valid] = np.angle(pxy[valid], deg=True)
    selected = _band_indices(frequencies, frequency_band)
    selected = selected[frequencies[selected] < 0.5 / dt * (1 - 1e-12)]
    if not len(selected):
        raise ValueError("frequency_band contains no bins below the temporal Nyquist limit")
    diagnostics = {
        "method": "Welch auto/cross spectra averaged across receivers before coherence calculation",
        "receiver_count": nx,
        "segment_samples": segment,
        "overlap_samples": overlap,
        "segment_count": 1 + (nt - segment) // (segment - overlap),
        "frequency_bin_hz": 1 / (segment * dt),
        "hann_equivalent_noise_bandwidth_hz": 1.5 / (segment * dt),
        "accepted_frequency_count": int(np.count_nonzero(valid[selected])),
        "phase_convention": "arg(conj(Ux)*Uy), vertical relative to horizontal, global coordinates",
        "support_rule": "coherence >=0.8 (>=0.9 when spectral contrast <6 dB); both powers >1e-6 of combined peak; combined power >=-30 dB of peak",
        "interpretation": "Surface displacement polarization of the receiver ensemble, not mode-specific ellipticity or energy partition. Mixed modes and component cancellation can bias it.",
        "spectral_contrast_caveat": "Spectral contrast is a baseline proxy, not calibrated SNR.",
    }
    return {
        "frequencies": frequencies[selected],
        "vertical_horizontal_ratio": ratio[selected],
        "cross_phase_deg": phase[selected],
        "coherence": coherence[selected],
        "valid": valid[selected],
        "component_support": component_support[selected],
        "relative_power_db": relative_db[selected],
        "spectral_contrast_db": contrast_db[selected],
        "diagnostics": diagnostics,
    }
