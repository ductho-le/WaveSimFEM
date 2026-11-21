"""Receiver-based packet measurements and reproducible wave-characterisation reports.

Measurements describe the selected signal window, not an automatically identified
mode. Attenuation includes interference, leakage, and numerical effects. No
material inversion or statistical confidence interval is implied by the fits.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt
from scipy.stats import linregress

from wavesimfem.storage import atomic_output
from wavesimfem.units import parse_quantity


def validate_options(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Validate report settings without importing plotting or FEM dependencies."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("wave_characterisation must be a mapping or null")
    defaults: dict[str, Any] = dict(
        enabled=False,
        component=None,
        frequency_band=None,
        time_window=None,
        min_relative_amplitude=0.01,
    )
    unknown = set(value) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown wave_characterisation settings: {sorted(unknown)}")
    options = defaults | dict(value)
    if not isinstance(options["enabled"], (bool, np.bool_)):
        raise ValueError("wave_characterisation.enabled must be boolean")
    options["enabled"] = bool(options["enabled"])
    if options["component"] not in (None, "x", "y", "pressure"):
        raise ValueError("wave_characterisation.component must be x, y, pressure or null")
    for key in ("frequency_band", "time_window"):
        pair = options[key]
        if pair is None:
            continue
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"{key} must contain two increasing bounds")
        try:
            parsed = []
            for v in pair:
                if isinstance(v, (bool, np.bool_)):
                    raise ValueError(f"expected a {key} bound, got a boolean")
                parsed.append(
                    parse_quantity(v, "frequency") if key == "frequency_band" else float(v)
                )
            pair = parsed
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid {key}: {exc}") from exc
        if not np.isfinite(pair).all() or pair[0] < 0 or pair[1] <= pair[0]:
            raise ValueError(f"{key} requires finite, nonnegative, increasing bounds")
        options[key] = pair
    threshold = options["min_relative_amplitude"]
    if (
        isinstance(threshold, (bool, np.bool_))
        or not isinstance(threshold, (int, float, np.integer, np.floating))
        or not 0 < float(threshold) < 1
    ):
        raise ValueError("min_relative_amplitude must lie strictly between zero and one")
    options["min_relative_amplitude"] = float(threshold)
    return options


def _peak_measurements(
    t: np.ndarray, envelope: np.ndarray
) -> tuple[int, float, float | None, float | None]:
    """Interpolate the dominant envelope peak and its connected half-height width."""
    peak = int(np.argmax(envelope))
    amplitude = float(envelope[peak])
    arrival = float(t[peak])
    fwhm = None
    if amplitude <= 0:
        return peak, amplitude, None, fwhm
    if 0 < peak < len(t) - 1:
        local = envelope[peak - 1 : peak + 2]
        if np.all(local > 0.1 * amplitude):
            left, center, right = np.log(local)
            curvature = left - 2 * center + right
            if curvature < 0:
                offset = np.clip(0.5 * (left - right) / curvature, -0.5, 0.5)
                arrival += float(offset * (t[1] - t[0]))
                amplitude *= float(np.exp(0.25 * (right - left) * offset))
        half = min(amplitude, float(envelope[peak])) / 2
        before = np.flatnonzero(envelope[:peak] <= half)
        after = np.flatnonzero(envelope[peak + 1 :] <= half)
        if len(before) and len(after):
            lo, hi = before[-1], peak + 1 + after[0]
            start = np.interp(half, envelope[lo : lo + 2], t[lo : lo + 2])
            stop = np.interp(half, envelope[hi - 1 : hi + 1][::-1], t[hi - 1 : hi + 1][::-1])
            fwhm = float(stop - start)
    return peak, amplitude, arrival, fwhm


def measure_packets(
    times: np.ndarray,
    positions: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    options: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Measure isolated packets and report only resolved, coherent spatial fits.

    A constant baseline is removed from each complete trace before optional
    zero-phase filtering and Hilbert-envelope extraction. Sub-sample peak times
    use a three-point parabola of the log envelope. Receivers are excluded when
    weak, truncated (gate endpoint exceeds 10% of the peak), or ambiguous (a
    second resolved peak reaches 50% of the primary). Velocity needs at least
    three complete packets, R² >= 0.9, and a fitted arrival span >= two samples.
    These are deterministic quality checks, not statistical confidence limits.
    """
    resolved_options = validate_options({} if options is None else options)
    assert resolved_options is not None
    if resolved_options["component"] is None:
        resolved_options["component"] = "y"
    options = resolved_options
    times, positions, ux, uy = [np.asarray(a, dtype=float) for a in (times, positions, ux, uy)]
    if times.ndim != 1 or positions.ndim != 1 or len(times) < 32 or len(positions) < 3:
        raise ValueError("Characterisation requires >=32 time samples and >=3 receivers")
    if not all(np.isfinite(a).all() for a in (times, positions, ux, uy)):
        raise ValueError("Characterisation inputs must be finite")
    dt = np.diff(times)
    if np.any(dt <= 0) or not np.allclose(dt, dt[0], rtol=1e-7, atol=0):
        raise ValueError("Characterisation requires uniform increasing times")
    if np.any(np.diff(positions) <= 0):
        raise ValueError("Receiver positions must be increasing")
    if ux.shape != (len(times), len(positions)) or uy.shape != ux.shape:
        raise ValueError("Displacement histories must have shape (time, receiver)")
    # Subtract a reference first: this also makes a truly constant trace exactly
    # zero, without roundoff left by summing a repeated large constant.
    ux, uy = [a - a[:1] for a in (ux, uy)]
    baselines = [a.mean(axis=0) for a in (ux, uy)]
    ux, uy = [a - baseline for a, baseline in zip((ux, uy), baselines, strict=True)]
    fs = 1 / dt[0]
    band = options["frequency_band"]
    if band is not None:
        if band[1] >= fs / 2:
            raise ValueError("frequency_band must lie below the stored sampling Nyquist frequency")
        sos = (
            butter(4, band[1], btype="lowpass", fs=fs, output="sos")
            if band[0] == 0
            else butter(4, band, btype="bandpass", fs=fs, output="sos")
        )
        ux, uy = [sosfiltfilt(sos, a, axis=0) for a in (ux, uy)]
    window = options["time_window"] or [float(times[0]), float(times[-1])]
    tolerance = 8 * np.finfo(float).eps * max(abs(times[0]), abs(times[-1]), dt[0])
    if window[0] < times[0] - tolerance or window[1] > times[-1] + tolerance:
        raise ValueError("time_window must lie within the recorded time interval")
    mask = (times >= window[0] - tolerance) & (times <= window[1] + tolerance)
    if mask.sum() < 16:
        raise ValueError("time_window must contain at least 16 stored samples")
    signal = ux if options["component"] == "x" else uy
    envelope = np.abs(hilbert(signal, axis=0))[mask]
    t = times[mask]
    measurements = [_peak_measurements(t, envelope[:, i]) for i in range(len(positions))]
    amplitudes = np.array([item[1] for item in measurements])
    arrivals = np.array([item[2] if item[2] is not None else np.nan for item in measurements])
    # Normalised weights prevent underflow for very small displacement signals.
    weights = (
        np.divide(envelope, amplitudes, out=np.zeros_like(envelope), where=amplitudes > 0) ** 2
    )
    totals = weights.sum(axis=0)
    relative_t = t - t[0]
    centroid = np.divide(
        (weights * relative_t[:, None]).sum(axis=0),
        totals,
        out=np.zeros_like(totals),
        where=totals > 0,
    )
    width = np.sqrt(
        np.divide(
            (weights * (relative_t[:, None] - centroid) ** 2).sum(axis=0),
            totals,
            out=np.zeros_like(totals),
            where=totals > 0,
        )
    )
    rows = []
    for i, x in enumerate(positions):
        peak, amplitude, arrival, fwhm = measurements[i]
        boundary_ratio = (
            float(max(envelope[0, i], envelope[-1, i]) / amplitude) if amplitude else 0.0
        )
        candidates, _ = find_peaks(envelope[:, i], prominence=0.25 * amplitude)
        primary = int(np.argmax(envelope[candidates, i])) if len(candidates) else None
        secondary = np.delete(candidates, primary) if primary is not None else candidates
        secondary_ratio = (
            float(envelope[secondary, i].max() / amplitude) if len(secondary) and amplitude else 0.0
        )
        reasons = []
        if amplitude <= 0:
            reasons.append("no_signal")
        elif amplitude < amplitudes.max() * options["min_relative_amplitude"]:
            reasons.append("below_amplitude_threshold")
        if amplitude > 0 and (peak in (0, len(t) - 1) or boundary_ratio > 0.1):
            reasons.append("truncated_at_gate")
        if secondary_ratio >= 0.5:
            reasons.append("multiple_packets")
        rows.append(
            dict(
                receiver_index=i,
                x_m=float(x),
                arrival_s=arrival,
                peak_envelope_m=float(amplitude),
                envelope_rms_width_s=float(width[i]) if amplitude else None,
                included_in_fit=not reasons,
                exclusion_reason=";".join(reasons),
                gate_boundary_ratio=boundary_ratio,
                secondary_peak_ratio=secondary_ratio,
                energy_centroid_s=float(centroid[i] + t[0]) if amplitude else None,
                envelope_energy_m2_s=float(totals[i] * amplitude**2 * dt[0]),
                envelope_fwhm_s=fwhm,
            )
        )
    valid = np.array([row["included_in_fit"] for row in rows])
    diagnostics = dict(
        min_r_squared=0.9,
        min_time_span_samples=2,
        sample_interval_s=float(dt[0]),
        max_gate_boundary_ratio=0.1,
        max_secondary_peak_ratio=0.5,
        r_squared=None,
        residual_rms_s=None,
        fitted_time_span_s=None,
    )
    summary = dict(
        total_receivers=len(positions),
        valid_receivers=int(valid.sum()),
        fit_status="insufficient_receivers",
        fit_diagnostics=diagnostics,
        amplitude_fit_diagnostics=None,
        velocity=None,
        apparent_attenuation=None,
        notes=[
            "Envelope-peak tracking measures the selected packet, not an automatically identified mode.",
            "Apparent attenuation includes interference, spreading and numerical effects; it is not intrinsic material loss.",
            "Widths are envelope-energy RMS widths within the selected gate; FWHM describes the dominant peak.",
            "Fit checks are deterministic screening criteria, not statistical confidence intervals.",
        ],
    )
    if valid.sum() >= 3:
        arrival_fit = linregress(positions[valid], arrivals[valid])
        prediction = arrival_fit.intercept + arrival_fit.slope * positions[valid]
        residual_rms = float(np.sqrt(np.mean((arrivals[valid] - prediction) ** 2)))
        r_squared = float(arrival_fit.rvalue**2) if np.isfinite(arrival_fit.rvalue) else 0.0
        time_span = float(abs(arrival_fit.slope) * np.ptp(positions[valid]))
        diagnostics.update(
            r_squared=r_squared, residual_rms_s=residual_rms, fitted_time_span_s=time_span
        )
        if time_span < 2 * dt[0]:
            summary["fit_status"] = "unresolved_travel_time"
        elif r_squared < 0.9:
            summary["fit_status"] = "inconsistent_arrivals"
        else:
            summary["fit_status"] = "accepted"
            summary["velocity"] = dict(
                signed_m_per_s=float(1 / arrival_fit.slope),
                intercept_s=float(arrival_fit.intercept),
                r_squared=r_squared,
                residual_rms_s=residual_rms,
                fitted_time_span_s=time_span,
                receiver_span_m=float(np.ptp(positions[valid])),
            )
            # Distance increases in the measured propagation direction, so a
            # decaying left-going packet has positive apparent attenuation too.
            direction = np.sign(arrival_fit.slope)
            fit = linregress(direction * positions[valid], np.log(amplitudes[valid]))
            summary["apparent_attenuation"] = dict(
                np_per_m=float(-fit.slope),
                db_per_m=float(-fit.slope * 20 / np.log(10)),
                log_amplitude_intercept=float(fit.intercept),
                propagation_direction=int(direction),
                r_squared=float(fit.rvalue**2) if np.isfinite(fit.rvalue) else 0.0,
                log_amplitude_residual_rms=float(
                    np.sqrt(
                        np.mean(
                            (
                                np.log(amplitudes[valid])
                                - fit.intercept
                                - fit.slope * direction * positions[valid]
                            )
                            ** 2
                        )
                    )
                ),
            )
            amplitude_r2 = summary["apparent_attenuation"]["r_squared"]
            log_range = float(np.ptp(np.log(amplitudes[valid])))
            summary["amplitude_fit_diagnostics"] = dict(
                r_squared=amplitude_r2,
                min_r_squared=0.8,
                log_amplitude_range=log_range,
                near_constant_log_range=0.01,
            )
            if amplitude_r2 < 0.8 and log_range > 0.01:
                summary["apparent_attenuation"] = None
                summary["notes"].append(
                    "Peak amplitudes do not support an exponential spatial trend (R² < 0.8); apparent attenuation is unavailable."
                )
            elif fit.slope > 0 and log_range > 0.01:
                summary["notes"].append(
                    "Amplitude grows along the fitted propagation direction; negative attenuation denotes apparent gain."
                )
    if summary["velocity"] is None:
        explanation = {
            "insufficient_receivers": "Fewer than three complete, unambiguous packets pass receiver screening.",
            "unresolved_travel_time": "The fitted travel-time span is smaller than two stored samples.",
            "inconsistent_arrivals": "Arrival times do not support a single coherent linear propagation trend (R² < 0.9).",
        }[summary["fit_status"]]
        summary["notes"].append(explanation + " Velocity and attenuation are unavailable.")
    report = dict(
        options=options,
        summary=summary,
        receivers=rows,
        preprocessing=dict(
            baseline="Complete-record mean removed separately from each displacement component and receiver",
            filter="Fourth-order Butterworth, forward-backward zero phase" if band else None,
            envelope="Hilbert transform of the complete processed trace before time gating",
            peak_time="Three-point parabolic interpolation of log envelope",
            recorded_time_window_s=[float(times[0]), float(times[-1])],
            effective_time_window_s=[float(t[0]), float(t[-1])],
            sampling_interval_s=float(dt[0]),
            nyquist_frequency_hz=float(fs / 2),
            stored_samples=len(times),
            gated_samples=len(t),
        ),
    )
    return report, (t, positions, ux[mask], uy[mask], envelope)


def write_characterisation(
    archive_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    options: Mapping[str, Any] | None = None,
    formats: Sequence[str] = ("png",),
    usetex: bool | str = False,
) -> dict[str, Any]:
    """Write publication panels, a README, receiver CSV, and finite SI-valued metrics."""
    import matplotlib

    from wavesimfem.analysis import _PLOT_LOCK, _rc, _typography_note, resolve_usetex
    from wavesimfem.characterisation_plots import render_characterisation

    validated_options = validate_options({} if options is None else options)
    assert validated_options is not None
    with np.load(archive_path, allow_pickle=False) as archive:
        observable = (
            str(np.asarray(archive["sensor_observable"]).item())
            if "sensor_observable" in archive
            else "displacement"
        )
        if observable not in ("displacement", "pressure", "velocity", "particle_velocity"):
            raise ValueError(f"Unsupported sensor observable {observable!r}")
        vector_keys = (
            ("sensor_data_x", "sensor_data_y")
            if observable == "displacement"
            else ("sensor_data_vx", "sensor_data_vy")
        )
        required = (
            ("times", "sensor_x", "sensor_data_pressure")
            if observable == "pressure"
            else ("times", "sensor_x", *vector_keys)
        )
        missing = set(required) - set(archive.files)
        if missing:
            raise ValueError(f"Characterisation archive is missing {sorted(missing)}")
        if observable == "pressure":
            if validated_options["component"] in ("x", "y"):
                raise ValueError(
                    "Pressure archives do not contain horizontal/vertical motion components"
                )
            resolved_options = {**validated_options, "component": "pressure"}
            signal = archive["sensor_data_pressure"]
            report, arrays = measure_packets(
                archive["times"],
                archive["sensor_x"],
                np.zeros_like(signal),
                signal,
                options=resolved_options,
            )
        else:
            if validated_options["component"] == "pressure":
                raise ValueError("The selected archive does not contain pressure sensors")
            resolved_options = (
                {**validated_options, "component": "y"}
                if validated_options["component"] is None
                else validated_options
            )
            report, arrays = measure_packets(
                *(archive[k] for k in required), options=resolved_options
            )
        provenance = (
            str(archive["provenance"].item())
            if "provenance" in archive and archive["provenance"].ndim == 0
            else None
        )
    report.update(
        schema_version=2,
        provenance=provenance,
        archive=str(Path(archive_path).resolve()),
        method="Baseline-corrected Hilbert envelope; sub-sample peaks; quality-screened spatial fits",
        observable=observable,
        amplitude_units={
            "displacement": "m",
            "pressure": "Pa",
            "velocity": "m/s",
            "particle_velocity": "m/s",
        }[observable],
    )
    rows = report["receivers"]
    if observable != "displacement":
        for row in rows:
            row["peak_envelope_si"] = row.pop("peak_envelope_m")
            row["envelope_squared_integral_si"] = row.pop("envelope_energy_m2_s")
        report["preprocessing"]["baseline"] = (
            "Complete-record mean removed separately from each observable component and receiver"
        )
    if observable == "pressure":
        report["summary"]["notes"].append(
            "Pressure is scalar; particle-motion and polarization plots are unavailable."
        )
    destination = Path(output_dir or Path(archive_path).parent) / "characterisation"
    destination.mkdir(parents=True, exist_ok=True)
    tex = resolve_usetex(usetex)
    with _PLOT_LOCK, matplotlib.rc_context(_rc(tex)):
        paths = render_characterisation(report, arrays, destination, formats)
    report["figures"] = [str(p) for p in paths]
    with atomic_output(destination / "metrics.json") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with atomic_output(destination / "receivers.csv") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    parameters = report["figure_parameters"]
    details = [
        "# Packet characterisation",
        "",
        f"Source archive: `{report['archive']}`.",
        "",
        f"Figures are exported at 190 mm width with {_typography_note(tex)} labels. Panel titles, axes, legends and measurement annotations are retained; interpretation and processing details are recorded here.",
        "",
        "## Processing and provenance",
        "",
        f"- Provenance: {provenance or 'No provenance string was supplied in the archive.'}",
        f"- Observable: {observable}; component: {report['options']['component']}; amplitude units: {report['amplitude_units']}.",
        f"- Effective time gate [s]: {report['preprocessing']['effective_time_window_s']}.",
        f"- Frequency filter [Hz]: {report['options']['frequency_band'] or 'none'}.",
        f"- Receivers accepted: {report['summary']['valid_receivers']} of {len(rows)}.",
        f"- Arrival-fit status: {report['summary']['fit_status']}.",
        f"- Displayed receiver positions [m]: {parameters['selected_receivers_m']}.",
        f"- STFT window [s]: {parameters['stft_window_s']:.8g}.",
        f"- Unpadded STFT frequency spacing [Hz]: {parameters['stft_frequency_spacing_hz']:.8g}.",
        "",
        "Baseline correction removes the complete-record mean independently for each component and receiver. An optional fourth-order, forward-backward Butterworth filter precedes the Hilbert envelope and time gate. Arrival times use three-point interpolation of the log envelope. Full preprocessing, fit diagnostics and SI-valued measurements are stored in `metrics.json`; individual exclusion reasons are in `receivers.csv`.",
        "",
        "## Figure captions",
        "",
    ]
    for name, note in report["figure_notes"].items():
        matching = [Path(path).name for path in paths if Path(path).stem == name]
        links = ", ".join(f"[{Path(path).suffix.lstrip('.').upper()}]({path})" for path in matching)
        details.extend([f"### {name.replace('_', ' ').capitalize()}", "", note, "", links, ""])
    details.extend(
        [
            "## Interpretation and unavailable measurements",
            "",
            "An N/A panel title indicates that the selected packet does not support a qualified fit. Empty motion or spectrum panels may indicate no usable signal; numerical reasons and receiver screening are in the data files. The plotted packet speed is an apparent envelope propagation speed, not a mode identification or a phase velocity. Decay is reported only when the spatial amplitude trend passes its quality checks.",
            "",
            *report["summary"]["notes"],
            "",
        ]
    )
    with atomic_output(destination / "README.md") as stream:
        stream.write("\n".join(details))
    return report
