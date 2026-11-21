"""Theory-guided fundamental Lamb-wave filtering for free isotropic plates.

Finite-aperture spectral masks yield approximate displacement contributions, not
energy-normalised modal amplitudes. Unresolved content remains in the residual.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import brentq
from scipy.signal.windows import tukey

from wavesimfem.materials import a0_wavenumber, compute_rayleigh_speed, structure_from_config
from wavesimfem.storage import atomic_output


@lru_cache(maxsize=256)
def _cached_s0_wavenumber(frequency: float, thickness: float, c_l: float, c_t: float) -> float:
    s = np.pi * frequency * thickness / c_t
    if s < 1e-4:
        return float(2 * np.pi * frequency / (2 * c_t * np.sqrt(1 - (c_t / c_l) ** 2)))
    if s > 30:
        return float(2 * np.pi * frequency / compute_rayleigh_speed(c_t, c_L=c_l))

    def determinant(x: float | np.ndarray) -> np.ndarray:
        p = np.sqrt((s * c_t / c_l) ** 2 - np.asarray(x) ** 2 + 0j)
        q = np.sqrt(s * s - np.asarray(x) ** 2 + 0j)
        return np.real(
            (q * q - x * x) ** 2 * np.sinc(q / np.pi) * np.cos(p)
            + 4 * x * x * p * np.sin(p) * np.cos(q)
        )

    grid = np.linspace(0, 1.05 * s * c_t / compute_rayleigh_speed(c_t, c_L=c_l), 2049)
    values = determinant(grid)
    zeros = np.flatnonzero(values == 0.0)
    zeros = zeros[zeros > 0]
    crossings = np.flatnonzero(values[:-1] * values[1:] < 0)
    if len(zeros) and (not len(crossings) or zeros[-1] > crossings[-1]):
        return float(grid[zeros[-1]] / (thickness / 2))
    if not len(crossings):
        raise ValueError(f"Unable to bracket S0 at {frequency:g} Hz")
    i = crossings[-1]
    return float(brentq(determinant, grid[i], grid[i + 1], xtol=1e-12) / (thickness / 2))


def s0_wavenumber(frequency: float, thickness: float, c_l: float, c_t: float) -> float:
    """Largest real symmetric Rayleigh–Lamb root (fundamental S0 branch).

    The pole-free symmetric determinant is divided by q to remove its spurious
    q=0 zero. Half-thickness-scaled variables reduce dimensional conditioning.
    """
    if not all(np.isfinite(v) and v > 0 for v in (frequency, thickness, c_l, c_t)) or c_l <= c_t:
        raise ValueError("S0 requires positive inputs and c_L > c_T")
    return _cached_s0_wavenumber(float(frequency), float(thickness), float(c_l), float(c_t))


def separate_fundamentals(
    times: np.ndarray,
    positions: np.ndarray,
    signal: np.ndarray,
    thickness: float,
    c_l: float,
    c_t: float,
    band: Sequence[float],
    width_bins: float = 2.0,
) -> dict[str, Any]:
    """Split a tapered array into A0, S0, and residual with an exact sum identity.

    Symmetric masks retain both propagation directions. Rows where A0/S0 mask
    supports overlap or reach spatial Nyquist are deliberately left unassigned.
    No zero padding is used; aperture controls actual spectral resolution.
    """
    t, x, u = [np.asarray(v, dtype=float) for v in (times, positions, signal)]
    if t.ndim != 1 or x.ndim != 1 or len(t) < 32 or len(x) < LAMB_MIN_RECEIVERS:
        raise ValueError(
            f"Lamb separation requires >=32 samples and >={LAMB_MIN_RECEIVERS} receivers"
        )
    for axis in (t, x):
        steps = np.diff(axis)
        if (
            not np.isfinite(axis).all()
            or np.any(steps <= 0)
            or not np.allclose(steps, steps[0], rtol=1e-7, atol=0)
        ):
            raise ValueError("Lamb separation requires uniform increasing coordinates")
    if u.shape != (len(t), len(x)) or not np.isfinite(u).all():
        raise ValueError("Invalid Lamb displacement history")
    if not np.isfinite(width_bins) or width_bins < 1:
        raise ValueError("Mask half-width must be at least one wavenumber bin")
    nyquist_f = 0.5 / (t[1] - t[0])
    if len(band) != 2 or not 0 < band[0] < band[1] < nyquist_f:
        raise ValueError("Lamb band must be positive, increasing, and below temporal Nyquist")
    f = np.fft.fftfreq(len(t), t[1] - t[0])
    k = 2 * np.pi * np.fft.fftfreq(len(x), x[1] - x[0])
    half_width = width_bins * 2 * np.pi / (len(x) * (x[1] - x[0]))
    tapered = u * tukey(len(t), 0.1)[:, None] * tukey(len(x), 0.1)[None, :]
    spectrum = np.fft.fft2(tapered)
    masks = {name: np.zeros_like(tapered) for name in ("A0", "S0")}
    positive = np.unique(np.abs(f[(np.abs(f) >= band[0]) & (np.abs(f) <= band[1])]))
    curves = []
    for frequency in positive:
        ka = a0_wavenumber(frequency, thickness, c_l, c_t)
        ks = s0_wavenumber(frequency, thickness, c_l, c_t)
        resolved = abs(ka - ks) > 2 * half_width and max(ka, ks) + half_width < np.pi / (
            x[1] - x[0]
        )
        curves.append((frequency, ka, ks, resolved))
        if not resolved:
            continue
        rows = np.flatnonzero(np.isclose(np.abs(f), frequency, rtol=1e-12, atol=0))
        # Smooth band edges and compact cosine wavenumber support avoid hard cuts.
        edge = min(
            (frequency - band[0]) / (0.1 * (band[1] - band[0])),
            (band[1] - frequency) / (0.1 * (band[1] - band[0])),
            1,
        )
        gain = np.sin(np.pi * max(0, edge) / 2) ** 2
        for name, center in (("A0", ka), ("S0", ks)):
            distance = np.abs(np.abs(k) - center) / half_width
            profile = np.where(distance < 1, 0.5 * (1 + np.cos(np.pi * np.minimum(distance, 1))), 0)
            masks[name][rows] = gain * profile
    if not curves:
        raise ValueError("No frequency bins in the requested Lamb band")
    modes = {name: np.fft.ifft2(spectrum * mask).real for name, mask in masks.items()}
    residual = tapered - modes["A0"] - modes["S0"]
    return dict(
        times=t,
        positions=x,
        tapered=tapered,
        residual=residual,
        **modes,
        frequency=f,
        wavenumber=k,
        spectrum=spectrum,
        curves=np.asarray(curves),
        half_width_rad_m=half_width,
        masks=masks,
    )


def _frequency_intervals(frequency: np.ndarray, selected: np.ndarray) -> list[list[float]]:
    """Contiguous intervals of retained FFT-bin centres (not interpolated edges)."""
    indices = np.flatnonzero(selected)
    if not len(indices):
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return [[float(frequency[g[0]]), float(frequency[g[-1]])] for g in groups]


def _component_diagnostics(
    tapered: np.ndarray, a0: np.ndarray, s0: np.ndarray, residual: np.ndarray
) -> dict[str, Any]:
    # RMS ratios are intentionally not called modal fractions: smooth spectral
    # masks are not orthogonal and squared displacements are not elastic energy.
    peak = float(np.max(np.abs(tapered)))
    rms = float(np.sqrt(np.mean(tapered**2)))
    error = float(np.max(np.abs(tapered - (a0 + s0 + residual))))
    return {
        "input_rms_m": rms,
        "rms_m": {
            name: float(np.sqrt(np.mean(values**2)))
            for name, values in (("A0", a0), ("S0", s0), ("residual", residual))
        },
        "rms_relative_to_input": {
            name: float(np.sqrt(np.mean(values**2)) / rms) if rms else None
            for name, values in (("A0", a0), ("S0", s0), ("residual", residual))
        },
        "reconstruction_max_abs_m": error,
        "reconstruction_relative_to_input_peak": error / peak if peak else None,
    }


def _lamb_diagnostics(
    result: Mapping[str, Any],
    horizontal: Mapping[str, Any],
    band: Sequence[float],
    archive_path: str | Path,
) -> dict[str, Any]:
    """Numerical and sampling context for interpreting approximate modal filters."""
    t, x, curves = result["times"], result["positions"], result["curves"]
    dt, dx = float(t[1] - t[0]), float(x[1] - x[0])
    half_width = result["half_width_rad_m"]
    overlap = np.abs(curves[:, 1] - curves[:, 2]) <= 2 * half_width
    aliased = np.max(curves[:, 1:3], axis=1) + half_width >= np.pi / dx
    usable = np.any(result["masks"]["A0"] + result["masks"]["S0"] > 0, axis=1)
    usable_curves = np.asarray(
        [
            np.any(usable & np.isclose(result["frequency"], f, rtol=1e-12, atol=0))
            for f in curves[:, 0]
        ]
    )
    y_metrics = _component_diagnostics(
        *(result[name] for name in ("tapered", "A0", "S0", "residual"))
    )
    if not y_metrics["input_rms_m"]:
        status = "no_vertical_signal"
    elif not usable_curves.any():
        status = "unresolved"
    elif not usable_curves.all():
        status = "partially_resolved"
    else:
        status = "resolved"
    warnings = []
    if not y_metrics["input_rms_m"]:
        warnings.append(
            "No vertical displacement signal is present; modal content cannot be assessed."
        )
    if overlap.any():
        warnings.append(
            "A0/S0 mask supports overlap at some frequencies; increase array aperture to improve wavenumber resolution."
        )
    if aliased.any():
        warnings.append(
            "Some mask supports reach spatial Nyquist; decrease receiver pitch to retain those frequencies."
        )
    if not usable_curves.any():
        warnings.append(
            "No frequency bins can be assigned; all tapered displacement remains in the residual."
        )
    relative = y_metrics["rms_relative_to_input"]["residual"]
    if relative is not None and relative > 0.5 and usable_curves.any():
        warnings.append(
            "Residual RMS exceeds half the input RMS; inspect unmodelled, out-of-band, and unresolved content before interpreting A0/S0 traces."
        )
    return dict(
        schema_version=2,
        archive=str(Path(archive_path).resolve()),
        status=status,
        band_hz=[float(v) for v in band],
        frequency_bin_spacing_hz=1 / (len(t) * dt),
        wavenumber_bin_spacing_rad_m=2 * np.pi / (len(x) * dx),
        aperture_m=float(x[-1] - x[0]),
        receiver_pitch_m=dx,
        temporal_nyquist_hz=0.5 / dt,
        spatial_nyquist_rad_m=np.pi / dx,
        mask_half_width_rad_m=float(half_width),
        resolved_frequency_bins=int(curves[:, 3].sum()),
        usable_frequency_bins=int(usable_curves.sum()),
        frequency_bins=len(curves),
        usable_frequency_intervals_hz=_frequency_intervals(curves[:, 0], usable_curves),
        interval_convention="Inclusive FFT-bin centres; band-edge taper can suppress a geometrically resolved bin.",
        rejected_bins={
            "overlapping_supports": int(overlap.sum()),
            "spatial_nyquist": int(aliased.sum()),
        },
        components={
            "y": y_metrics,
            "x": _component_diagnostics(
                *(horizontal[name] for name in ("tapered", "A0", "S0", "residual"))
            ),
        },
        warnings=warnings,
        taper="10% Tukey in time and space; reconstruction is of tapered data",
        boundary_effects="Fourier filtering is nonlocal and periodic; edge ringing and apparent pre-arrivals are not physical arrivals. Interpret interior packets only.",
        interpretation="Approximate A0/S0 displacement filtering, not modal energy or material attenuation. RMS ratios are not additive modal fractions. Higher modes, out-of-band and overlapping branches remain in residual.",
        signed_wavenumber_convention="Positive displayed wavenumber indicates propagation toward increasing receiver x at positive temporal frequency (minus the spatial FFT wavenumber). Both directions are retained.",
    )


LAMB_MIN_RECEIVERS = 8


def lamb_eligibility_error(
    cfg: Mapping[str, Any], summary: Mapping[str, Any] | None = None
) -> str | None:
    """Reason a configuration cannot produce a Lamb report, or ``None`` if it can.

    Shared by configuration validation (before any solve) and the report itself.
    """
    regions = (summary or structure_from_config(cfg))["all_regions"]
    if (
        len(regions) != 1
        or regions[0]["is_halfspace"]
        or regions[0]["material_model"] != "isotropic"
    ):
        return "Lamb report supports only a free, single isotropic elastic plate"
    observable = cfg.get("sensor_observable")
    if observable not in (None, "auto", "displacement"):
        return f"Lamb report requires displacement sensors (got observable {observable!r})"
    n_sensors = cfg.get("n_sensors")
    if n_sensors is not None and int(n_sensors) < LAMB_MIN_RECEIVERS:
        return (
            f"Lamb report requires at least {LAMB_MIN_RECEIVERS} receivers (got {int(n_sensors)})"
        )
    return None


def write_lamb_report(
    archive_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    formats: Sequence[str] = ("png", "pdf"),
    usetex: bool | str = False,
) -> dict[str, Any]:
    """Report modal-filter resolvability, displacement contributions and reconstruction.

    Physical A0/S0 mode identification requires a free, uniform isotropic plate.
    The masks use finite-aperture resolution and retain both propagation directions.
    """
    import matplotlib

    from wavesimfem.analysis import (
        _PLOT_LOCK,
        _figure,
        _label,
        _rc,
        _save,
        resolve_usetex,
    )
    from wavesimfem.structure_analysis import _configuration, _scalar
    from wavesimfem.units import (
        displacement_unit,
        frequency_unit,
        length_unit,
        parse_quantity,
        time_unit,
        wavenumber_unit,
    )

    with np.load(archive_path, allow_pickle=False) as data:
        struct_cfg = _configuration(data)
        if struct_cfg is None:
            raise ValueError("Lamb report requires archived simulation_config_yaml provenance")
        cfg = yaml.safe_load(_scalar(data, "simulation_config_yaml"))
        provenance = str(_scalar(data, "provenance", "Recorded simulation archive"))
        summary = structure_from_config(struct_cfg)
        regions = summary["all_regions"]
        # Receiver count is enforced by separate_fundamentals on the actual array.
        reason = lamb_eligibility_error({}, summary)
        if reason is not None:
            raise ValueError(reason)
        if "sensor_data_x" not in data or "sensor_data_y" not in data:
            raise ValueError("Lamb report requires archived horizontal and vertical displacements")
        excitation = cfg.get("excitation") if isinstance(cfg.get("excitation"), Mapping) else {}
        mesh = cfg.get("mesh") if isinstance(cfg.get("mesh"), Mapping) else {}
        raw_fc = (
            cfg["f_center"]
            if cfg.get("f_center") is not None
            else excitation.get("f_center", _scalar(data, "f_center"))
        )
        raw_min = (
            cfg["min_frequency"]
            if cfg.get("min_frequency") is not None
            else mesh.get("min_frequency")
        )
        raw_max = (
            cfg["max_frequency"]
            if cfg.get("max_frequency") is not None
            else mesh.get("max_frequency")
        )
        f_center = parse_quantity(raw_fc, "frequency") if raw_fc is not None else None
        min_f = parse_quantity(raw_min, "frequency") if raw_min is not None else None
        max_f = parse_quantity(raw_max, "frequency") if raw_max is not None else None
        if (not min_f or not max_f) and (f_center is None or f_center <= 0):
            raise ValueError("Lamb report requires finite positive f_center or explicit band")
        upper = max_f or 3 * f_center
        band = [
            min_f or min(f_center / 5, 0.5 * upper),
            upper,
        ]
        t, x = data["times"], data["sensor_x"]
        c_l = float(data["c_L"]) if "c_L" in data else float(regions[0]["c_L"])
        c_t = float(data["c_T"]) if "c_T" in data else float(regions[0]["c_T"])
        result = separate_fundamentals(
            t,
            x,
            data["sensor_data_y"],
            regions[0]["thickness"],
            c_l,
            c_t,
            band,
        )
        ux = np.asarray(data["sensor_data_x"], dtype=float)
        if ux.shape != result["tapered"].shape or not np.isfinite(ux).all():
            raise ValueError("Invalid Lamb horizontal displacement history")
        ux = ux * tukey(len(t), 0.1)[:, None] * tukey(len(x), 0.1)[None, :]
        sx = np.fft.fft2(ux)
        horizontal = {name: np.fft.ifft2(sx * mask).real for name, mask in result["masks"].items()}
        horizontal.update(tapered=ux, residual=ux - horizontal["A0"] - horizontal["S0"])
        offset = x - (float(data["source_x"]) if "source_x" in data else 0.0)
    destination = Path(output_dir or Path(archive_path).parent) / "lamb_modes"
    destination.mkdir(parents=True, exist_ok=True)
    diagnostics = _lamb_diagnostics(result, horizontal, band, archive_path)
    diagnostics["data_provenance"] = provenance
    curves = result["curves"]
    tu, xu, fu = (
        time_unit(np.max(np.abs(t))),
        length_unit(np.max(np.abs(offset))),
        frequency_unit(band[1]),
    )
    ku = wavenumber_unit(length_unit(1 / np.max(curves[:, 1:3])))
    uu = displacement_unit(np.max(np.abs(result["tapered"])))
    ink, muted, blue, orange = "#233044", "#667085", "#167D9A", "#D97732"
    paths = []
    tex = resolve_usetex(usetex)
    with _PLOT_LOCK, matplotlib.rc_context(_rc(tex)):
        # Resolve the physical interpretation before showing separated waveforms.
        fig = _figure(145, layout="constrained")
        fig.get_layout_engine().set(w_pad=0.035, h_pad=0.035, wspace=0.02, hspace=0.035)
        grid = fig.add_gridspec(2, 1, height_ratios=(1.45, 1))
        ax = fig.add_subplot(grid[0])
        fm = (
            (result["frequency"] > 0)
            & (result["frequency"] >= band[0])
            & (result["frequency"] <= band[1])
        )
        order = np.argsort(-result["wavenumber"])
        signed_k = -result["wavenumber"][order]
        power = np.abs(result["spectrum"][fm][:, order]) ** 2
        db = 10 * np.log10(np.maximum(power / (power.max() or 1), 1e-6))
        image = ax.pcolormesh(
            ku(signed_k),
            fu(result["frequency"][fm]),
            db,
            cmap="magma",
            vmin=-60,
            vmax=0,
            shading="auto",
            rasterized=True,
        )
        for column, name, color in ((1, "A0", "#66D7EA"), (2, "S0", "#FFD394")):
            for direction in (-1, 1):
                ax.plot(
                    direction * ku(curves[:, column]),
                    fu(curves[:, 0]),
                    color=color,
                    lw=1.3,
                    label={"A0": r"$A_0$", "S0": r"$S_0$"}[name] if direction == 1 else None,
                )
        ax.axvline(0, color="white", lw=0.6, alpha=0.4)
        limit = min(
            diagnostics["spatial_nyquist_rad_m"],
            np.max(curves[:, 1:3]) + 2 * result["half_width_rad_m"],
        )
        ax.set(
            xlabel=_label("Signed wavenumber", ku),
            ylabel=_label("Frequency", fu),
            ylim=fu(np.asarray(band)),
            xlim=ku(np.array([-limit, limit])),
        )
        ax.set_title("(a) Dispersion", loc="center")
        ax.grid(False)
        ax.legend(
            loc="upper right", frameon=True, facecolor="#233044", labelcolor="white", framealpha=0.9
        )
        fig.colorbar(
            image,
            ax=ax,
            label="Power [dB]",
            shrink=0.85,
            pad=0.015,
        )

        ax = fig.add_subplot(grid[1])
        half_width = result["half_width_rad_m"]
        separation = np.abs(curves[:, 1] - curves[:, 2]) / (2 * half_width)
        nyquist_margin = (
            diagnostics["spatial_nyquist_rad_m"] - np.max(curves[:, 1:3], axis=1)
        ) / half_width
        ax.plot(fu(curves[:, 0]), separation, color=blue, label=r"$A_0/S_0$ separation")
        ax.plot(fu(curves[:, 0]), nyquist_margin, color=orange, label="Nyquist clearance")
        ax.axhline(1, color=ink, lw=0.9, ls="--", label="Threshold")
        ax.set(
            xlabel=_label("Frequency", fu),
            ylabel="Resolution margin [ratio]",
            xlim=fu(np.asarray(band)),
        )
        ax.set_title("(b) Resolution margins", loc="center")
        ax.legend(loc="best", ncols=3)
        ax.grid(axis="y")

        paths.extend(_save(fig, str(destination / "modal_resolution"), formats, 300))

        fig = _figure(140, layout="constrained")
        fig.get_layout_engine().set(w_pad=0.035, h_pad=0.035, wspace=0.02, hspace=0.035)
        axes = fig.subplots(2, 2, sharex=True, sharey=True)
        peak = float(np.max(np.abs(result["tapered"])))
        display_limit = float(np.percentile(np.abs(result["tapered"]), 99)) or peak or 1
        for ax, name, label in zip(
            axes.flat, ("tapered", "A0", "S0", "residual"), ("a", "b", "c", "d"), strict=True
        ):
            image = ax.pcolormesh(
                tu(t),
                xu(offset),
                uu(result[name]).T,
                cmap="RdBu_r",
                vmin=-float(uu(display_limit)),
                vmax=float(uu(display_limit)),
                shading="auto",
                rasterized=True,
            )
            title = {
                "tapered": "Input",
                "A0": r"$A_0$",
                "S0": r"$S_0$",
                "residual": "Residual",
            }[name]
            ax.set_title(f"({label}) {title}", loc="center")
            ax.grid(False)
        for ax in axes[-1]:
            ax.set_xlabel(_label("Time", tu))
        for ax in axes[:, 0]:
            ax.set_ylabel(_label("Source offset", xu))
        fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            label=_label(r"$u_y$", uu),
            location="bottom",
            aspect=50,
            extend="neither",
            shrink=0.9,
            pad=0.02,
        )
        paths.extend(_save(fig, str(destination / "modal_separation"), formats, 300))

        fig = _figure(140, layout="constrained")
        fig.get_layout_engine().set(w_pad=0.035, h_pad=0.035, wspace=0.02, hspace=0.035)
        axes = fig.subplots(3, 1, sharex=True, sharey=True)
        # Interior quantiles avoid the taper edges and provide a reproducible
        # view of propagation rather than selecting only the largest amplitude.
        indices = np.rint(np.linspace(0.2, 0.8, 3) * (len(x) - 1)).astype(int)
        reconstruction = result["A0"] + result["S0"] + result["residual"]
        for row, (ax, index) in enumerate(zip(axes, indices, strict=True)):
            ax.plot(
                tu(t),
                uu(result["tapered"][:, index]),
                color="#D0D5DD",
                lw=2.8,
                label="Input",
                zorder=1,
            )
            ax.plot(
                tu(t),
                uu(reconstruction[:, index]),
                color=ink,
                lw=0.85,
                ls="--",
                label="Sum",
                zorder=2,
            )
            for name, color in (("residual", muted), ("A0", blue), ("S0", orange)):
                ax.plot(
                    tu(t),
                    uu(result[name][:, index]),
                    color=color,
                    lw=1.0,
                    label={"residual": "Residual", "A0": r"$A_0$", "S0": r"$S_0$"}[name],
                )
            ax.set_ylabel(_label(r"$u_y$", uu))
            ax.set_title(
                f"({'abc'[row]}) Offset = {xu(offset[index]):.4g} {xu.tex}",
                loc="center",
            )
            ax.grid(axis="y")
        axes[0].legend(ncol=5, loc="upper right", frameon=True, framealpha=0.95)
        axes[-1].set_xlabel(_label("Time", tu))
        paths.extend(_save(fig, str(destination / "modal_waveforms"), formats, 300))
    payload = {
        name: result[name]
        for name in ("times", "positions", "tapered", "A0", "S0", "residual", "curves")
    }
    payload.update({f"{name}_x": values for name, values in horizontal.items()})
    with atomic_output(destination / "modal_signals.npz", binary=True) as stream:
        np.savez(stream, **payload)
    diagnostics["figures"] = paths
    with atomic_output(destination / "diagnostics.json") as stream:
        json.dump(diagnostics, stream, indent=2, allow_nan=False)
    _write_readme(destination, diagnostics, usetex=tex)
    return diagnostics


def _write_readme(
    destination: Path, diagnostics: Mapping[str, Any], *, usetex: bool = False
) -> None:
    """Keep interpretation and provenance beside the publication figures."""
    from wavesimfem.analysis import _typography_note

    component = diagnostics["components"]["y"]
    intervals = diagnostics["usable_frequency_intervals_hz"]
    interval_text = ", ".join(f"{a:.6g}–{b:.6g} Hz" for a, b in intervals) or "None"
    warnings = "\n".join(f"- {warning}" for warning in diagnostics["warnings"]) or "- None."
    content = f"""# Lamb-wave report

Archive: `{diagnostics["archive"]}`.

Data provenance: {diagnostics["data_provenance"]}.

## Figures

All exports are 190 mm wide with {_typography_note(usetex)} typography. Figure captions and interpretation belong here; the figures contain only data panels, panel titles, labels and legends.

- **modal_resolution**: (a) signed frequency–wavenumber power spectrum with free-plate Rayleigh–Lamb fundamental branches; (b) separation and spatial Nyquist margins. Both margins must exceed one for geometric resolution. Positive displayed wavenumber denotes travel toward increasing receiver x. Power is squared vertical-displacement Fourier magnitude relative to the peak within the displayed frequency band, shown over a 60 dB range; it is not mechanical energy.
- **modal_separation**: tapered vertical-displacement input, approximate A0 and S0 contributions, and the unassigned residual. All four panels share the same colour scale, clipped at the input's 99th absolute percentile. Values beyond the display limits saturate at the endpoint colours; the colour bar has flat ends. No-signal and unresolved cases are recorded below rather than annotated over the data.
- **modal_waveforms**: input, reconstruction (A0 + S0 + residual), and individual contributions at three interior receiver quantiles. All panels share amplitude and time scales. Offsets are measured from the source.

## Sampling and validity

- Status: `{diagnostics["status"]}`.
- Usable frequency bins: {diagnostics["usable_frequency_bins"]} / {diagnostics["frequency_bins"]}.
- Usable FFT-bin centres: {interval_text}.
- Receiver aperture: {diagnostics["aperture_m"]:.8g} m; receiver pitch: {diagnostics["receiver_pitch_m"]:.8g} m.
- Fourier spacings: {diagnostics["frequency_bin_spacing_hz"]:.8g} Hz and {diagnostics["wavenumber_bin_spacing_rad_m"]:.8g} rad/m.
- Mask half-width: {diagnostics["mask_half_width_rad_m"]:.8g} rad/m.
- Maximum reconstruction error: {component["reconstruction_max_abs_m"]:.8g} m.

{diagnostics["taper"]}. {diagnostics["boundary_effects"]}

{diagnostics["interpretation"]}

{diagnostics["interval_convention"]}

## Warnings

{warnings}

Full component RMS ratios, sampling checks and reconstruction diagnostics are in `diagnostics.json`. Filtered signals, theory curves and the tapered input are in `modal_signals.npz`. The archived material and boundary configuration, rather than current command defaults, determines whether free-plate Lamb theory applies.
"""
    with atomic_output(destination / "README.md") as stream:
        stream.write(content)
