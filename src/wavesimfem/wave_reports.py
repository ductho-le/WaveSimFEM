"""Structure-aware, multi-panel reports of measured wave propagation.

The report separates observations, applicable reference solutions and sampling
limitations. Single-line pressure or motion histories cannot determine modal energy,
material loss, depth localization, or a unique subsurface/material inversion.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

from wavesimfem.analysis import (
    _PLOT_LOCK,
    _figure,
    _label,
    _line_axes,
    _rc,
    _robust_limit,
    _save,
    _typography_note,
    resolve_usetex,
)
from wavesimfem.reporting import log
from wavesimfem.storage import atomic_output
from wavesimfem.structure_analysis import _scalar, reference_dispersion, structure_profile
from wavesimfem.units import (
    Unit,
    frequency_unit,
    length_unit,
    observable_unit,
    parse_quantity,
    time_unit,
    wavenumber_unit,
)
from wavesimfem.wave_spectra import phase_velocity_spectrum, polarization_spectrum

_BLUE, _ORANGE, _PURPLE, _GREEN = "#167D9A", "#D97732", "#7356A6", "#248267"
_INK, _MUTED = "#233044", "#667085"
_COLORS = (_BLUE, _ORANGE, _PURPLE, _GREEN)


def _finite(value: Any) -> float | None:
    return float(value) if np.isfinite(value) else None


def _read_record(
    path: str | Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = dict(archive)
    observable = str(_scalar(data, "sensor_observable", "displacement"))
    if "sensor_data_pressure" in data and "sensor_observable" not in data:
        observable = "pressure"
    if observable == "pressure":
        keys, unit = (None, "sensor_data_pressure"), "Pa"
    elif observable in ("velocity", "particle_velocity"):
        keys, unit = ("sensor_data_vx", "sensor_data_vy"), "m/s"
    elif observable == "displacement":
        keys, unit = ("sensor_data_x", "sensor_data_y"), "m"
    else:
        raise ValueError(f"Unknown archived sensor_observable {observable!r}")
    data["sensor_observable"] = observable
    data.setdefault("sensor_units", unit)
    if str(_scalar(data, "sensor_units")) != unit:
        raise ValueError(f"Archived {observable} history must use SI units '{unit}'")
    missing = {"times", "sensor_x", keys[1]} - data.keys()
    if missing:
        raise ValueError(f"Wave report archive is missing {sorted(missing)}")
    t, x = [np.asarray(data[key], dtype=float) for key in ("times", "sensor_x")]
    if t.ndim != 1 or x.ndim != 1 or len(t) < 32 or len(x) < 3:
        raise ValueError("Wave report requires >=32 stored times and >=3 receivers")
    for axis in (t, x):
        step = np.diff(axis)
        if (
            not np.isfinite(axis).all()
            or np.any(step <= 0)
            or not np.allclose(step, step[0], rtol=1e-7, atol=0)
        ):
            raise ValueError("Wave report requires finite, uniform increasing coordinates")
    arrays = []
    # Two slots allow the existing spectral routine to process vector or scalar
    # observables. A scalar has one empty slot; it never becomes a motion field.
    for key in keys:
        values = np.asarray(data[key], dtype=float) if key in data else np.zeros((len(t), len(x)))
        if values.shape != (len(t), len(x)) or not np.isfinite(values).all():
            raise ValueError("Invalid observable history shape or nonfinite values")
        values = values - values[:1]
        arrays.append(values - values.mean(axis=0))
    return data, t, x, arrays[0], arrays[1]


def _unavailable_polarization(frequencies: np.ndarray, reason: str) -> dict[str, Any]:
    return dict(
        frequencies=frequencies,
        vertical_horizontal_ratio=np.full_like(frequencies, np.nan),
        cross_phase_deg=np.full_like(frequencies, np.nan),
        coherence=np.zeros_like(frequencies),
        valid=np.zeros(len(frequencies), bool),
        diagnostics={"status": "unavailable", "reason": reason},
    )


def _one_side(data: Mapping[str, Any], x: np.ndarray) -> tuple[np.ndarray, int, str, bool]:
    if "source_x" not in data:
        return np.arange(len(x)), 1, "Source x unavailable; +x selected", False
    source = float(data["source_x"])
    if not np.isfinite(source):
        raise ValueError("source_x must be finite")
    left, right = np.flatnonzero(x < source), np.flatnonzero(x > source)
    if len(left) >= len(right) and len(left) >= 3:
        return left, -1, "-x from source", True
    if len(right) >= 3:
        return right, 1, "+x from source", True
    return np.arange(len(x)), 1, "No one-sided aperture of >=3 receivers", False


def _frequency_band(
    data: Mapping[str, Any],
    t: np.ndarray,
    f_min: float | str | None,
    f_max: float | str | None,
) -> tuple[float, float]:
    nyquist = 0.5 / (t[1] - t[0])
    f0 = float(data.get("f_center", nyquist / 4))

    def bound(value: float | str | None, key: str, legacy: str, default: float) -> float:
        if value is not None:
            return parse_quantity(value, "frequency")
        if key in data:
            return float(data[key])
        if legacy in data:
            return float(data[legacy]) * 1e6
        return default

    lo = bound(f_min, "disp_f_min", "f_min_disp_MHz", 0.0)
    hi = bound(f_max, "disp_f_max", "f_max_disp_MHz", min(2 * f0, nyquist))
    if not np.isfinite([lo, hi]).all() or not 0 <= lo < hi:
        raise ValueError("Wave report frequency limits require finite 0 <= min < max")
    hi = min(hi, nyquist)
    if lo >= hi:
        raise ValueError("Requested report band is outside temporal Nyquist")
    return (lo, hi)


def _array_spectra(
    t: np.ndarray,
    x: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    band: tuple[float, float],
) -> dict[str, Any]:
    """Unpadded temporal spectrum and signed, spatially tapered Fourier image."""
    scale = max(float(np.max(np.abs(ux))), float(np.max(np.abs(uy)))) or 1.0
    dt, pitch = t[1] - t[0], x[1] - x[0]
    f = np.fft.rfftfreq(len(t), dt)
    select = np.flatnonzero((f > 0) & (f >= band[0]) & (f <= band[1]) & (f < 0.5 / dt))
    if not len(select):
        raise ValueError("Report band contains no positive FFT bins below Nyquist")
    transforms = [
        np.fft.rfft(a / scale * np.hanning(len(t))[:, None], axis=0)[select] for a in (ux, uy)
    ]
    power = [np.mean(np.abs(a) ** 2, axis=1) for a in transforms]
    reference = max(float(p.max()) for p in power) or 1.0
    signed_k = -2 * np.pi * np.fft.fftfreq(2 * len(x), pitch)
    order = np.argsort(signed_k)
    images = [
        np.abs(np.fft.fft(a * np.hanning(len(x))[None, :], n=2 * len(x), axis=1)[:, order]) ** 2
        for a in transforms
    ]
    return dict(
        frequency=f[select],
        wavenumber=signed_k[order],
        transforms=transforms,
        power=power,
        power_reference=reference,
        images=images,
    )


def _decay_spectrum(
    spectra: Mapping[str, Any], x: np.ndarray, direction: int
) -> list[dict[str, Any]]:
    """Apparent Fourier-amplitude slope; no compensation for mode interference."""
    f = spectra["frequency"]
    direction_x = direction * (x - x.mean())
    rows = []
    # The observed dominant component is selected by the caller.
    amplitudes = np.abs(spectra["selected_transform"])
    powers = np.mean(amplitudes**2, axis=1)
    reference = float(powers.max()) or 1.0
    for i, frequency in enumerate(f):
        a = amplitudes[i]
        valid = a > max(float(a.max()) * 0.01, np.finfo(float).tiny)
        alpha, candidate_alpha, r2, reason = None, None, None, "weak spectral support"
        if powers[i] / reference >= 1e-3 and valid.sum() >= 4:
            distance, log_a = direction_x[valid], np.log(a[valid])
            dx, dy = distance - distance.mean(), log_a - log_a.mean()
            slope = float(np.sum(dx * dy) / np.sum(dx * dx))
            candidate_alpha = -slope
            residual = dy - slope * dx
            variance = float(np.sum(dy * dy))
            r2 = (
                max(0.0, 1 - float(np.sum(residual * residual)) / variance)
                if variance > 1e-20
                else 1.0
            )
            if r2 >= 0.8 or np.ptp(log_a) <= 0.01:
                alpha, reason = -slope, "accepted"
            else:
                reason = "non-exponential spatial amplitudes"
        rows.append(
            dict(
                frequency_hz=float(frequency),
                apparent_decay_np_m=alpha,
                apparent_decay_db_m=alpha * 20 / np.log(10) if alpha is not None else None,
                candidate_decay_db_m=(
                    candidate_alpha * 20 / np.log(10) if candidate_alpha is not None else None
                ),
                r_squared=r2,
                status=reason,
            )
        )
    return rows


def _reference_selection(
    profile: Mapping[str, Any], curves: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    priority = (
        "lamb",
        "rayleigh",
        "constituent_rayleigh",
        "acoustic_radiation",
        "acoustic",
        "quasi_bulk",
        "bulk",
    )
    if profile["lamb_applicable"]:
        return [c for c in curves if c["kind"] == "lamb"]
    chosen, seen = [], set()
    for kind in priority:
        for curve in curves:
            speed = round(float(curve["phase_velocity_m_s"][0]), 4)
            if curve["kind"] == kind and speed not in seen:
                chosen.append(curve)
                seen.add(speed)
        if len(chosen) >= 4:
            break
    return chosen[:4]


def _short_reference(curve: Mapping[str, Any], profile: Mapping[str, Any]) -> str:
    base = (
        curve["label"]
        .split(" · ")[0]
        .replace("Constituent Rayleigh", "Rayleigh")
        .replace(" (elastic)", "")
    )
    if len(profile["regions"]) > 1 and "region" in curve:
        index = next(
            (i + 1 for i, r in enumerate(profile["regions"]) if r["name"] == curve["region"]), None
        )
        if index:
            base += f" ({index})"
    return base.replace("c_f", "$c_f$").replace("A0", "$A_0$").replace("S0", "$S_0$")


def _physics_diagnostic(
    ax: Axes,
    profile: Mapping[str, Any],
    scan: Mapping[str, Any],
    pol: Mapping[str, Any],
    decay: Sequence[Mapping[str, Any]],
    fu: Unit,
) -> tuple[str, str]:
    family = profile["family"]
    f, cp = scan["frequencies"], scan["ridge_velocity"]
    fluids = [r for r in profile["regions"] if r["is_fluid"] and r.get("c_fluid_m_s")]
    if family == "layered_halfspace" and profile["plate_thickness_m"]:
        h = profile["plate_thickness_m"]
        ax.plot(fu(f), cp / (f * h), "o-", ms=3, color=_BLUE)
        ax.axhline(1, ls="--", color=_ORANGE, label="$\\lambda/h = 1$")
        ax.set(
            ylabel="Wavelength ratio $\\lambda/h$",
            title="(d) Wavelength scale",
            ylim=(0, None),
        )
        ax.legend()
        kind, note = (
            "wavelength_thickness",
            "Wavelength / thickness describes scale sensitivity; it is not an investigation depth or inversion.",
        )
    elif fluids:
        seen_speeds = set()
        for i, region in enumerate(fluids):
            speed = region["c_fluid_m_s"]
            if round(speed, 6) in seen_speeds:
                continue
            seen_speeds.add(round(speed, 6))
            ax.plot(
                fu(f),
                cp / speed,
                "o-",
                ms=3,
                color=_COLORS[i % 4],
                label=f"$c_f$ = {speed:.0f} m/s",
            )
        ax.axhline(1, ls="--", color=_INK, lw=1)
        ax.set(
            ylabel="Velocity ratio $c_p/c_f$",
            title="(d) Fluid sound line",
            ylim=(0, None),
        )
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
        ax.legend()
        kind = "fluid_sound_line"
        note = "Above unity permits a propagating acoustic field in that fluid; below unity is evanescent there. This is not a leakage measurement."
    elif any(r["material_model"] == "viscoelastic" for r in profile["regions"]):
        loss = np.array(
            [
                r["apparent_decay_db_m"] if r["apparent_decay_db_m"] is not None else np.nan
                for r in decay
            ]
        )
        frequencies = fu(np.array([r["frequency_hz"] for r in decay]))
        candidates = np.array(
            [
                r["candidate_decay_db_m"] if r["candidate_decay_db_m"] is not None else np.nan
                for r in decay
            ]
        )
        ax.plot(frequencies, candidates, color="#B8C1CE", lw=1, label="Observed slope")
        ax.plot(frequencies, loss, "o-", ms=3, color=_ORANGE, label="Qualified fit")
        ax.axhline(0, color=_MUTED, lw=0.8)
        ax.set(ylabel="Apparent spectral decay [dB/m]", title="(d) Spectral decay")
        ax.legend(loc="lower left", frameon=True, framealpha=0.95)
        kind, note = (
            "spectral_decay",
            "Untapered full-record Fourier-amplitude fits (R² >= 0.8); interference, spreading, record truncation and dissipation all contribute.",
        )
    else:
        pf = pol["frequencies"]
        ax.plot(
            fu(pf), pol["vertical_horizontal_ratio"], "o-", ms=3, color=_BLUE, label="Measured V/H"
        )
        if family == "solid_halfspace":
            region = profile["regions"][0]
            cr, ct = region.get("c_R_m_s"), region.get("c_T_m_s")
            y = profile["sensor_y_m"]
            # Surface ellipticity does not apply to a receiver buried in the medium.
            if (
                cr
                and ct
                and y is not None
                and abs(y - region["y_top_m"]) <= 1e-9 * max(region["thickness_m"], 1e-9)
            ):
                q = np.sqrt(1 - (cr / ct) ** 2)
                ratio = (1 + q * q) / (2 * q)
                ax.axhline(ratio, color=_ORANGE, ls="--", label="Surface Rayleigh")
        ax.set(
            ylabel="Amplitude ratio $V/H$",
            title="(d) Polarization",
            ylim=(0, None),
        )
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
        ax.legend()
        kind, note = (
            "polarization",
            "V/H is a coherent vector-motion ratio, not an A0/S0 fraction. Mixed modes can bias the measured polarization.",
        )
        if not pol["valid"].any():
            # Show why the ratio is withheld instead of an empty amplitude axis.
            ax.clear()
            supported = pol.get("component_support", np.zeros(len(pf), bool)) & (
                pol.get("relative_power_db", np.full(len(pf), -np.inf)) >= -30
            )
            coherence = np.where(supported, pol["coherence"], np.nan)
            ax.plot(fu(pf), coherence, color=_BLUE, label="Measured")
            ax.axhline(0.8, color=_ORANGE, ls="--", label="Acceptance threshold")
            ax.set(
                ylabel="Component coherence",
                title="(d) Polarization coherence",
                ylim=(0, 1.05),
            )
            ax.legend(loc="best", frameon=True, framealpha=0.95)
            note = "V/H withheld. Component coherence must reach 0.8 (0.9 at low spectral contrast), with sufficient power in both components."
    ax.set_xlabel(_label("Frequency", fu))
    _line_axes(ax)
    return kind, note


def _write_readme(destination: Path, report: Mapping[str, Any], *, usetex: bool = False) -> None:
    """Keep figure captions, provenance and interpretation beside each export."""
    profile, summary = report["profile"], report["summary"]
    observable = summary.get("sensor_observable", "displacement")
    scalar = observable == "pressure"
    wavefield_caption = (
        "- **(a) Pressure wavefield:** all receivers with a pressure colour scale. "
        "**(b) Pressure at receivers:** first, middle and last receiver histories. "
        "Pressure is scalar; no vector polarization is inferred."
        if scalar
        else f"- **(a, b) Horizontal and vertical wavefields:** all receivers, with a common physical "
        f"{observable.replace('_', ' ')} colour scale and one shared colourbar to the right of both panels. "
        "The flat-ended colourbar saturates beyond the robust display limits; "
        "the recorded measurements are unchanged."
    )
    lines = [
        f"# {profile['title']}",
        "",
        f"Data provenance: **{report['data_provenance']}**.",
        f"Archive: `{report['archive']}`.",
        f"Structure identification: `{profile['provenance']}`; family: `{profile['family']}`.",
        "",
        f"Figures use a fixed 190 mm export width and {_typography_note(usetex)} text throughout. "
        "Titles identify panels only; captions, interpretation and limitations are recorded here.",
        "",
        "## Structure and wave families",
        "",
        "Layer numbers in reference legends follow this bottom-to-top order. "
        "A halfspace is represented by a truncated computational domain.",
        "",
        "| Layer | Region | Material | Thickness (m) | Domain |",
        "| --- | --- | --- | --- | --- |",
    ]
    for index, region in enumerate(profile["regions"], 1):
        thickness = region["thickness_m"]
        thickness_text = f"{thickness:.6g}" if thickness is not None else "Unspecified"
        name = region["name"].replace("|", "\\|")
        domain = "Halfspace" if region["is_halfspace"] else "Finite layer"
        lines.append(
            f"| {index} | {name} | {region['material_model']}"
            + (f" ({region['fluid_formulation']})" if region.get("fluid_formulation") else "")
            + f" | {thickness_text} | {domain} |"
        )
    lines.extend(["", "Expected wave families: " + "; ".join(profile["wave_families"]) + ".", ""])
    lines.extend(f"- {text}" for text in profile["interpretation"] + profile["exclusions"])
    lines.extend(
        [
            "",
            "## Dispersion and modes",
            "",
            "File: `dispersion_and_modes` in each requested format.",
            "",
            f"- **(a) Dispersion:** signed frequency-wavenumber power for the selected {observable.replace('_', ' ')} "
            "observable, normalized to the peak in the displayed frequency band. The colour scale "
            "covers -50 to 0 dB. Reference colours match panel (c); positive and negative "
            "wavenumbers represent opposite propagation directions.",
            "- **(b) Phase coherence:** amplitude-weighted phase alignment over trial phase speeds, "
            "on an absolute 0-1 scale. Grey cells fail signal or wavelength support. White points "
            "are qualified, isolated peaks; phase speed is not packet/group speed.",
            "- **(c) Phase velocity:** qualified measured peaks and applicable physical references. "
            "Dashed material guides are constituent comparisons, not dispersion solutions for the "
            "complete structure. Reference applicability is listed below.",
            f"- **(d) Structure diagnostic:** {report['interpretation']}",
            "",
            "## Propagation and components",
            "",
            "File: `propagation_and_components` in each requested format.",
            "",
            wavefield_caption,
            "- **(c) Component spectra:** array-mean squared Fourier amplitudes with one shared "
            "reference for both components. The dotted -30 dB line marks relative support. "
            "There is no receiver-by-receiver amplitude normalization.",
            "- **(d) Directionality:** opposite-direction squared-observable spectral power as "
            "a percentage of resolved bidirectional power. This is not elastic energy flux or "
            "a reflection coefficient. The central two wavenumber bins and ambiguous spatial "
            "Nyquist bin are excluded.",
            "",
            "## Measurement selection and limitations",
            "",
            f"- Sensor observable: `{observable}` [{summary.get('sensor_units', 'm')}].",
            f"- Selected component: `{summary['selected_component']}`.",
            f"- Receiver selection: {summary['selected_receiver_count']} of "
            f"{summary['total_receiver_count']}; {summary['direction']}.",
            f"- Qualified phase-velocity samples: {summary['accepted_velocity_samples']}.",
            f"- Coherent polarization samples: {summary['coherent_polarization_samples']}.",
            f"- Frequency band: {summary['frequency_band_hz'][0]:.6g} to "
            f"{summary['frequency_band_hz'][1]:.6g} Hz.",
        ]
    )
    for key, value in summary["sampling"].items():
        lines.append(f"- {key.replace('_', ' ')}: {value:.8g}.")
    lines.extend(["", "Unavailable or rejected measurements:", ""])
    lines.extend(f"- {status}" for status in summary["unavailable_measurements"])
    if not summary["unavailable_measurements"]:
        lines.append(
            "- None at the report level; per-frequency acceptance remains in the CSV files."
        )
    lines.extend(
        [
            "",
            "The complete-record mean is removed. Spectral maps use temporal and spatial Hann "
            "tapers. Apparent spectral decay uses untapered full-record Fourier amplitudes to "
            "avoid a moving packet acquiring artificial attenuation from a fixed temporal taper. "
            "No automatic time gate or band-pass is applied.",
            "",
            "A receiver line does not uniquely identify mode identity, material loss, energy flux, "
            "depth localization or an inverse structure. Modal interference, spreading, incomplete "
            "records and dissipation can all affect amplitude trends. Blank or grey regions are "
            "unsupported measurements, not evidence of zero propagation.",
            "",
            "## References and reproducible data",
            "",
        ]
    )
    for curve in report["references"]:
        lines.append(f"- **{curve['label']}**: {curve['applicability']}")
    lines.extend(["", "Physical references:", ""])
    lines.extend(f"- [{source['title']}]({source['url']})" for source in profile["sources"])
    lines.extend(
        [
            "",
            "`phase_velocity.csv`, `polarization.csv`, `spectral_decay.csv` and "
            "`directionality.csv` retain SI measurements, fit quality and acceptance status. "
            "`spectral_arrays.npz` preserves the velocity grid, coherence, support masks and "
            "selected receiver indices. `report.json` stores the archived structure, "
            "provenance, references and method diagnostics.",
            "",
        ]
    )
    with atomic_output(destination / "README.md") as stream:
        stream.write("\n".join(lines))


def write_structure_report(
    archive_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    formats: Sequence[str] = ("png", "pdf"),
    usetex: bool | str = False,
    f_min: float | str | None = None,
    f_max: float | str | None = None,
    k_min: float | str | None = None,
    k_max: float | str | None = None,
) -> dict[str, Any]:
    """Write two physics-aware multi-panel figures plus reproducible measurements.

    Files live in ``wave_physics/``. Physical identification comes exclusively
    from archive provenance, not the current command's simulation defaults.
    Phase analysis selects the larger one-sided receiver array when the source
    lies inside the array. All receivers remain visible in propagation panels.
    """
    data, t, x, ux, uy = _read_record(archive_path)
    observable = str(data["sensor_observable"])
    scalar = observable == "pressure"
    observable_label = observable.replace("_", " ").capitalize()
    profile = structure_profile(data)
    band = _frequency_band(data, t, f_min, f_max)
    side, direction, direction_note, known_side = _one_side(data, x)
    xs, usx, usy = x[side], ux[:, side], uy[:, side]
    component = 1
    if profile["preferred_component"] == "auto" or not np.any(usy):
        component = int(np.sum(usy**2) >= np.sum(usx**2))
    chosen = usy if component else usx
    spectra = _array_spectra(t, xs, usx, usy, band)
    # A fixed temporal taper changes amplitudes as a packet moves across the
    # record. Preserve translation-invariant amplitudes for spatial decay.
    raw_f = np.fft.rfftfreq(len(t), t[1] - t[0])
    amplitude_scale = float(np.max(np.abs(chosen))) or 1.0
    spectra["selected_transform"] = np.fft.rfft(chosen / amplitude_scale, axis=0)[
        np.searchsorted(raw_f, spectra["frequency"])
    ]
    fref = np.linspace(max(band[0], 1 / (len(t) * (t[1] - t[0]))), band[1], 100)
    curves = reference_dispersion(profile, fref)
    visible_refs = _reference_selection(profile, curves)
    speeds = [
        float(v)
        for curve in curves
        for v in curve["phase_velocity_m_s"]
        if np.isfinite(v) and v > 0
    ]
    if not speeds:
        speeds = [
            float(v)
            for v in (data.get("c_T"), data.get("c_L"), data.get("c_fluid"))
            if v is not None and np.isfinite(v) and float(v) > 0
        ] or [1000.0, 6000.0]
    cmin, cmax = max(min(speeds) * 0.25, 1e-6), max(speeds) * 1.12
    velocities = np.geomspace(cmin, cmax, 280)
    scan = phase_velocity_spectrum(
        t, xs, chosen, velocities, frequency_band=band, direction=direction, max_frequencies=240
    )
    if not known_side and "source_x" in data:
        scan["support_mask"][:] = False
        scan["ridge_velocity"][:] = np.nan
        scan["ridge_valid"][:] = False
        scan["ridge_reason"][:] = "no one-sided aperture"
        scan["diagnostics"]["accepted_ridge_count"] = 0
        scan["diagnostics"]["ridge_reason_counts"] = {
            "no one-sided aperture": len(scan["frequencies"])
        }
    if scalar:
        pol = _unavailable_polarization(
            scan["frequencies"],
            "Pressure is a scalar observable; vector polarization is unavailable",
        )
    else:
        try:
            pol = polarization_spectrum(t, usx, usy, frequency_band=band)
        except ValueError as exc:
            if "bins" not in str(exc):
                raise
            pol = _unavailable_polarization(
                scan["frequencies"], "No Welch frequency bins in the requested band"
            )
    decay = _decay_spectrum(spectra, xs, direction)
    fu, tu = frequency_unit(band[1]), time_unit(np.max(np.abs(t)))
    offsets = x - float(data.get("source_x", 0.0))
    xu = length_unit(np.max(np.abs(offsets)))
    pitch = x[1] - x[0]
    klim = np.pi / pitch
    for value, key, factor in ((k_max, "disp_k_max", 1.0),):
        if value is not None:
            klim = min(klim, parse_quantity(value, "wavenumber"))
        elif key in data:
            klim = min(klim, float(data[key]) * factor)
        elif "k_max_disp" in data:
            klim = min(klim, float(data["k_max_disp"]) * 1000)
    if k_min is not None:
        kfloor = parse_quantity(k_min, "wavenumber")
    elif "disp_k_min" in data:
        kfloor = float(data["disp_k_min"])
    elif "k_min_disp" in data:
        kfloor = float(data["k_min_disp"]) * 1000
    else:
        kfloor = 0.0
    if not np.isfinite([kfloor, klim]).all() or not 0 <= kfloor < klim:
        raise ValueError("Wave report wavenumber limits require finite 0 <= min < max")
    ku = wavenumber_unit(length_unit(2 * np.pi / klim))
    destination = Path(output_dir or Path(archive_path).parent) / "wave_physics"
    destination.mkdir(parents=True, exist_ok=True)
    velocity_scale = 1.0 if profile["family"] == "layered_halfspace" or cmax < 1000 else 1000.0
    velocity_label = "Phase velocity [m/s]" if velocity_scale == 1 else "Phase velocity [km/s]"
    paths = []
    summary = dict(
        selected_component="pressure" if scalar else "y" if component else "x",
        sensor_observable=observable,
        sensor_units=str(_scalar(data, "sensor_units")),
        direction=direction_note,
        selected_receiver_count=len(side),
        total_receiver_count=len(x),
        accepted_velocity_samples=int(scan["ridge_valid"].sum()),
        coherent_polarization_samples=int(pol["valid"].sum()),
        sampling=dict(
            time_step_s=float(t[1] - t[0]),
            receiver_pitch_m=float(pitch),
            temporal_nyquist_hz=float(0.5 / (t[1] - t[0])),
            spatial_nyquist_rad_m=float(np.pi / pitch),
        ),
        frequency_band_hz=list(band),
    )
    log(f"  Structure-aware report: {profile['title']}")
    tex = resolve_usetex(usetex)
    with _PLOT_LOCK, matplotlib.rc_context(_rc(tex)):
        fig = _figure(145)
        gs = fig.add_gridspec(
            2, 2, left=0.095, right=0.89, top=0.955, bottom=0.085, hspace=0.32, wspace=0.46
        )
        cells = [
            gs[i, j].subgridspec(1, 2, width_ratios=(1, 0.045), wspace=0.04)
            for i in range(2)
            for j in range(2)
        ]
        axes = [fig.add_subplot(cell[0, 0]) for cell in cells]
        colour_axes = [fig.add_subplot(cells[i][0, 1]) for i in range(2)]
        ax = axes[0]
        image = spectra["images"][component]
        peak = float(image.max()) or 1.0
        db = 10 * np.log10(np.maximum(image / peak, 1e-8))
        k = spectra["wavenumber"]
        selected_k = (np.abs(k) <= klim) & (np.abs(k) >= kfloor)
        display = np.ma.masked_where(~np.broadcast_to(selected_k, db.shape), db)
        artist = ax.pcolormesh(
            fu(spectra["frequency"]),
            ku(k),
            display.T,
            cmap="magma",
            vmin=-50,
            vmax=0,
            shading="auto",
            rasterized=True,
        )
        for i, curve in enumerate(visible_refs):
            color = ("#74D7EA", "#FFD48C", "#C3AAF1", "#9CDDCA")[i % 4]
            for sign in (-1, 1):
                ax.plot(
                    fu(np.array(curve["frequency_hz"])),
                    sign * ku(np.array(curve["wavenumber_rad_m"])),
                    color=color,
                    lw=1.05,
                    label=_short_reference(curve, profile) if sign == 1 else None,
                )
        ax.axhline(0, lw=0.6, color="white", alpha=0.4)
        ax.set(
            xlabel=_label("Frequency", fu),
            ylabel=_label("Wavenumber", ku),
            title="(a) Dispersion",
            ylim=(-ku(klim), ku(klim)),
            xlim=fu(np.array(band)),
        )
        fig.colorbar(artist, cax=colour_axes[0], label="Power [dB]")
        ax = axes[1]
        cmap = matplotlib.colormaps["viridis"].with_extremes(bad="#EDF1F5")
        artist = ax.pcolormesh(
            fu(scan["frequencies"]),
            scan["phase_velocities"] / velocity_scale,
            np.ma.masked_where(~scan["support_mask"], scan["coherence"]).T,
            cmap=cmap,
            vmin=0,
            vmax=1,
            shading="auto",
            rasterized=True,
        )
        ax.plot(
            fu(scan["frequencies"]),
            scan["ridge_velocity"] / velocity_scale,
            ".",
            color="white",
            ms=3,
            label="Qualified peak",
        )
        ax.set(
            xlabel=_label("Frequency", fu),
            ylabel=velocity_label,
            title="(b) Phase coherence",
            ylim=(cmin / velocity_scale, cmax / velocity_scale),
            xlim=fu(np.array(band)),
        )
        fig.colorbar(artist, cax=colour_axes[1], label="Coherence")
        ax = axes[2]
        for i, curve in enumerate(visible_refs):
            ax.plot(
                fu(np.array(curve["frequency_hz"])),
                np.array(curve["phase_velocity_m_s"]) / velocity_scale,
                color=_COLORS[i % 4],
                lw=1.25,
                ls="-" if curve["kind"] in ("lamb", "rayleigh") else "--",
                label=_short_reference(curve, profile),
            )
        ax.plot(
            fu(scan["frequencies"]),
            scan["ridge_velocity"] / velocity_scale,
            "o",
            mfc="white",
            mec=_INK,
            ms=3.5,
            label="Measured",
        )
        ax.set(
            xlabel=_label("Frequency", fu),
            ylabel=velocity_label,
            title="(c) Phase velocity",
            ylim=(cmin / velocity_scale, cmax / velocity_scale),
            xlim=fu(np.array(band)),
        )
        ax.legend(loc="upper right", frameon=True, framealpha=0.95)
        _line_axes(ax)
        metric_kind, metric_note = _physics_diagnostic(axes[3], profile, scan, pol, decay, fu)
        axes[3].set_xlim(fu(np.array(band)))
        summary["structure_diagnostic"] = metric_kind
        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        paths.extend(_save(fig, str(destination / "dispersion_and_modes"), formats, 300))

        fig = _figure(145)
        outer = fig.add_gridspec(
            1,
            2,
            left=0.095,
            right=0.89,
            top=0.955,
            bottom=0.085,
            width_ratios=(1, 0.022),
            wspace=0.025,
        )
        gs = outer[0].subgridspec(2, 2, hspace=0.32, wspace=0.27)
        axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
        colour_ax = fig.add_subplot(outer[1].subgridspec(2, 1, hspace=0.32)[0])
        uu = observable_unit(max(np.max(np.abs(ux)), np.max(np.abs(uy))), observable)
        limit = _robust_limit(uu(uy) if scalar else np.stack((uu(ux), uu(uy))))
        norm = matplotlib.colors.Normalize(vmin=-limit, vmax=limit)
        component_names = (
            ("Horizontal $u_1$", "Vertical $u_2$")
            if observable == "displacement"
            else ("Horizontal $v_1$", "Vertical $v_2$")
        )
        maps = (
            [(axes[0], uy, "Pressure $p$", "a")]
            if scalar
            else list(zip(axes[:2], (ux, uy), component_names, ("a", "b"), strict=True))
        )
        for ax, values, name, letter in maps:
            artist = ax.pcolormesh(
                tu(t),
                xu(offsets),
                uu(values).T,
                cmap="RdBu_r",
                norm=norm,
                shading="auto",
                rasterized=True,
            )
            ax.set(
                xlabel=_label("Time", tu),
                ylabel=_label("Source offset" if "source_x" in data else "Receiver x", xu),
                title=f"({letter}) {name}",
                xlim=(tu(t[0]), tu(t[-1])),
            )
        fig.colorbar(
            artist,
            cax=colour_ax,
            label=_label(observable_label, uu),
            extend="neither",
        )
        if scalar:
            ax = axes[1]
            for idx, i in enumerate(sorted({0, len(x) // 2, len(x) - 1})):
                ax.plot(
                    tu(t),
                    uu(uy[:, i]),
                    color=_COLORS[idx % len(_COLORS)],
                    label=f"$x = {xu(offsets[i]):.3g}$ {xu.tex}",
                )
            ax.set(
                xlabel=_label("Time", tu),
                ylabel=_label("Pressure", uu),
                title="(b) Pressure at receivers",
                xlim=(tu(t[0]), tu(t[-1])),
            )
            ax.legend(loc="best")
            _line_axes(ax)
        ax = axes[2]
        spectral_components = [(1, "Pressure $p$")] if scalar else list(enumerate(component_names))
        for i, label in spectral_components:
            db = 10 * np.log10(np.maximum(spectra["power"][i] / spectra["power_reference"], 1e-8))
            ax.plot(fu(spectra["frequency"]), db, color=_COLORS[i], label=label)
        ax.axhline(-30, ls=":", color=_MUTED, lw=0.9, label="Support threshold")
        ax.set(
            xlabel=_label("Frequency", fu),
            ylabel="Spectral power [dB]",
            title="(c) Component spectra",
            ylim=(-65, 3),
            xlim=fu(np.array(band)),
        )
        ax.legend()
        _line_axes(ax)
        ax = axes[3]
        k = spectra["wavenumber"]
        # Exclude the central Hann lobe and Nyquist bin, whose sign is ambiguous.
        dk = 2 * np.pi / (len(xs) * pitch)
        k_lo, k_hi = 2 * dk * (1 - 1e-12), np.pi / pitch * (1 - 1e-12)
        selected = spectra["images"][component]
        positive = np.sum(selected[:, (k >= k_lo) & (k < k_hi)], axis=1)
        negative = np.sum(selected[:, (k <= -k_lo) & (k > -k_hi)], axis=1)
        total = positive + negative
        band_power = spectra["power"][component]
        supported = (band_power >= (band_power.max() or 1) * 1e-3) & (total > 0)
        opposite = negative if direction == 1 else positive
        fraction = np.divide(opposite, total, out=np.full_like(total, np.nan), where=supported)
        ax.plot(fu(spectra["frequency"]), 100 * fraction, color=_PURPLE, lw=1.5)
        ax.set(
            xlabel=_label("Frequency", fu),
            ylabel=f"{'Opposite-direction' if known_side else '-x'} power [%]",
            title="(d) Directionality",
            ylim=(0, 100),
            xlim=fu(np.array(band)),
        )
        _line_axes(ax)
        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        paths.extend(_save(fig, str(destination / "propagation_and_components"), formats, 300))

    ridge_rows = [
        dict(
            frequency_hz=float(f),
            phase_velocity_m_s=_finite(scan["ridge_velocity"][i]),
            candidate_velocity_m_s=_finite(scan["candidate_velocity"][i]),
            aperture_velocity_scale_m_s=_finite(scan["velocity_resolution"][i]),
            coherence=float(scan["peak_coherence"][i]),
            accepted=bool(scan["ridge_valid"][i]),
            status=str(scan["ridge_reason"][i]),
        )
        for i, f in enumerate(scan["frequencies"])
    ]
    pol_rows = [
        dict(
            frequency_hz=float(f),
            vertical_horizontal_ratio=_finite(pol["vertical_horizontal_ratio"][i]),
            vertical_relative_phase_deg=_finite(pol["cross_phase_deg"][i]),
            coherence=float(pol["coherence"][i]),
            accepted=bool(pol["valid"][i]),
        )
        for i, f in enumerate(pol["frequencies"])
    ]
    directional_rows = [
        dict(frequency_hz=float(f), opposite_direction_power_fraction=_finite(fraction[i]))
        for i, f in enumerate(spectra["frequency"])
    ]
    for name, rows in (
        ("phase_velocity", ridge_rows),
        ("polarization", pol_rows),
        ("spectral_decay", decay),
        ("directionality", directional_rows),
    ):
        with atomic_output(destination / (name + ".csv")) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with atomic_output(destination / "spectral_arrays.npz", binary=True) as stream:
        np.savez_compressed(
            stream,
            frequencies_hz=scan["frequencies"],
            phase_velocities_m_s=scan["phase_velocities"],
            coherence=scan["coherence"],
            support_mask=scan["support_mask"],
            ridge_velocity_m_s=scan["ridge_velocity"],
            receiver_indices=side,
        )
    unavailable = []
    if not scalar and not np.any(ux):
        unavailable.append("Horizontal component absent or without dynamic signal.")
    if not scalar and not np.any(uy):
        unavailable.append("Vertical component absent or without dynamic signal.")
    if not scan["ridge_valid"].any():
        unavailable.append("No isolated phase-velocity ridge passes sampling and coherence checks.")
    if scalar:
        unavailable.append("Pressure is a scalar observable; vector polarization is unavailable.")
    elif not pol["valid"].any():
        unavailable.append("No supported coherent component pair; V/H polarization is withheld.")
    if not any(row["status"] == "accepted" for row in decay):
        unavailable.append("No exponential amplitude trend passes the spatial-decay fit check.")
    if not np.isfinite(fraction).any():
        unavailable.append("No supported directional content.")
    summary["unavailable_measurements"] = unavailable
    report = dict(
        schema_version=2,
        archive=str(Path(archive_path).resolve()),
        data_provenance=str(data.get("provenance", "Recorded simulation archive")),
        profile=profile,
        summary=summary,
        phase_velocity_method=scan["diagnostics"],
        polarization_method=pol["diagnostics"],
        references=curves,
        figures=paths,
        interpretation=metric_note,
    )
    with atomic_output(destination / "report.json") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    _write_readme(destination, report, usetex=tex)
    return report
