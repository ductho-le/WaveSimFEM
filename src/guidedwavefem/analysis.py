"""Signal processing, 2D FFT dispersion analysis, and publication figures.

All labels are written so they render identically with Matplotlib's built-in
mathtext and with a full LaTeX installation (``usetex``); plotting settings are
scoped with :func:`matplotlib.pyplot.rc_context` and never leak to the caller.
Axis units are chosen from the data (e.g. s/ms/μs, m/mm, Hz/kHz/MHz, rad/m or
rad/mm), so seismic and ultrasonic records are both displayed naturally.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import butter, sosfiltfilt

from guidedwavefem.materials import compute_rayleigh_speed
from guidedwavefem.numerics import dispersion_spectrum
from guidedwavefem.reporting import log, warn
from guidedwavefem.sources import describe_source, source_delay, source_duration, source_signal
from guidedwavefem.units import (
    Unit,
    displacement_unit,
    fmt_frequency,
    frequency_unit,
    length_unit,
    parse_quantity,
    time_unit,
    wavenumber_unit,
)

# Figures are built with the object-oriented API (``Figure`` + its own canvas),
# so no pyplot state or global backend selection is involved; this is safe
# headless, under MPI, in threads, and inside interactive sessions.

FIGURE_FORMATS = ("png", "pdf", "svg", "eps")

_BASE_RC = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
}

#: Archive keys of the SI dispersion limits and their legacy (MHz, rad/mm) fallbacks.
_LIMIT_KEYS = {
    "f_min": ("disp_f_min", "f_min_disp_MHz", 1e6),
    "f_max": ("disp_f_max", "f_max_disp_MHz", 1e6),
    "k_min": ("disp_k_min", "k_min_disp", 1e3),
    "k_max": ("disp_k_max", "k_max_disp", 1e3),
}

#: Legacy archives without ``is_fluid_top``: a top region with ``c_T < ratio · c_L``
#: is a regularized fluid (its shear speed is ~1e-4 c_L). Scale-free, so very soft
#: solids (e.g. soils with Vs < 50 m/s, whose c_T/c_L stays above ~0.05) are not
#: misclassified.
FLUID_SHEAR_RATIO = 0.01


def latex_available() -> bool:
    """True if the external tools required by Matplotlib ``usetex`` are on PATH."""
    return all(shutil.which(tool) for tool in ("latex", "dvipng", "gs"))


def resolve_usetex(usetex: bool | str = "auto") -> bool:
    """Resolve ``'auto'``/bool into a concrete usetex decision."""
    if isinstance(usetex, str):
        if usetex.lower() != "auto":
            raise ValueError("usetex must be 'auto', True or False")
        return latex_available()
    if usetex and not latex_available():
        warn("usetex requested but latex/dvipng/ghostscript not found; using mathtext.")
        return False
    return bool(usetex)


def _rc(usetex: bool) -> dict[str, object]:
    rc: dict[str, object] = dict(_BASE_RC)
    rc["text.usetex"] = usetex
    if not usetex:
        rc["mathtext.fontset"] = "stix"
    return rc


def _title(text: str, usetex: bool) -> str:
    """Bold title text in both rendering modes."""
    return rf"\textbf{{{text}}}" if usetex else text


def bandpass_filter(
    signals: np.ndarray, fs: float, f_center: float
) -> tuple[np.ndarray, tuple[float, float] | None]:
    """Zero-phase 4th-order Butterworth band-pass ``[f_center/5, 3 f_center]``.

    The band scales with ``f_center`` (valid from seismic to ultrasonic records).
    Returns the filtered ``(n_t, n_x)`` array and the band, or the input and
    ``None`` if the band or record length is not admissible.
    """
    f_lo = f_center / 5.0
    f_hi = min(f_center * 3.0, 0.45 * fs)
    sos = butter(4, [f_lo, f_hi], btype="bandpass", fs=fs, output="sos") if f_lo < f_hi else None
    if sos is None or signals.shape[0] <= 3 * (2 * len(sos) + 1):
        return signals, None
    return sosfiltfilt(sos, signals, axis=0), (f_lo, f_hi)


def _save(fig: Figure, base: str, formats: Sequence[str], dpi: int) -> list[str]:
    paths = []
    ps_logger = logging.getLogger("matplotlib.backends.backend_ps")
    for ext in formats:
        ext = ext.lower().lstrip(".")
        if ext not in FIGURE_FORMATS:
            raise ValueError(f"Unsupported figure format '{ext}'")
        path = f"{base}.{ext}"
        # EPS has no alpha channel; translucent legend frames/grid lines are
        # flattened to opaque by design, so the backend's warning is suppressed.
        previous = ps_logger.level
        if ext == "eps":
            ps_logger.setLevel(logging.ERROR)
        try:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        finally:
            ps_logger.setLevel(previous)
        log(f"  Saved: {path}")
        paths.append(path)
    return paths


def _label(quantity: str, unit: Unit) -> str:
    return f"{quantity} [{unit.tex}]"


def _band_text(band: tuple[float, float] | None) -> str:
    if not band:
        return ""
    unit = frequency_unit(band[1])
    return f", band-pass {unit(band[0]):.3g}-{unit(band[1]):.3g} {unit.tex}"


def plot_quick_overview(
    times: np.ndarray,
    sensor_x: np.ndarray,
    sensor_data_y: np.ndarray,
    f_center: float,
    source_x: float,
    output_path: str,
    source_type: str = "ricker",
    n_cycles: int | None = None,
) -> str:
    """Fast diagnostic figure: excitation, waterfall, and B-scan of ``u_2``."""
    offsets = np.asarray(sensor_x, dtype=float) - source_x
    times = np.asarray(times, dtype=float)
    fs = 1.0 / (times[1] - times[0])
    filtered, band = bandpass_filter(np.asarray(sensor_data_y, dtype=float), fs, f_center)
    t_u = time_unit(times[-1])
    x_u = length_unit(np.max(np.abs(offsets)))
    t_disp, x_disp = t_u(times), x_u(offsets)

    with matplotlib.rc_context(_rc(False)):
        fig = Figure(figsize=(12, 10))
        axes = fig.subplots(3, 1)
        pulse = source_duration(source_type, f_center, n_cycles)
        t_exc = np.linspace(0.0, 1.3 * pulse, 1000)
        exc_u = time_unit(t_exc[-1])
        axes[0].plot(exc_u(t_exc), source_signal(t_exc, source_type, f_center, n_cycles), lw=1.5)
        axes[0].set(
            xlabel=_label("Time", exc_u),
            ylabel="Amplitude",
            title=f"Excitation: {describe_source(source_type, n_cycles)}, "
            f"$f_0$ = {fmt_frequency(f_center)}",
        )
        axes[0].grid(True, alpha=0.3)

        n_sensors = len(x_disp)
        step = max(n_sensors // 30, 1)
        peak = float(np.max(np.abs(filtered))) or 1.0
        scale = 0.8 * (x_disp[1] - x_disp[0]) * step / peak
        for j in range(0, n_sensors, step):
            axes[1].plot(t_disp, filtered[:, j] * scale + x_disp[j], "b-", lw=0.5)
        axes[1].set(
            xlabel=_label("Time", t_u),
            ylabel=_label("Offset from source", x_u),
            title=f"$u_2$ waterfall{_band_text(band)}",
        )
        axes[1].grid(True, alpha=0.3)

        dt_d = t_disp[1] - t_disp[0]
        dx_d = x_disp[1] - x_disp[0]
        vmax = 0.5 * float(np.max(np.abs(filtered)))
        if vmax > 0:
            axes[2].imshow(
                filtered.T,
                aspect="auto",
                origin="lower",
                cmap="seismic",
                vmin=-vmax,
                vmax=vmax,
                extent=[
                    t_disp[0] - dt_d / 2,
                    t_disp[-1] + dt_d / 2,
                    x_disp[0] - dx_d / 2,
                    x_disp[-1] + dx_d / 2,
                ],
            )
        axes[2].set_xlim(t_disp[0], t_disp[-1])
        axes[2].set_ylim(x_disp[0], x_disp[-1])
        axes[2].yaxis.set_major_locator(MaxNLocator(nbins=6))
        axes[2].set(
            xlabel=_label("Time", t_u),
            ylabel=_label("Offset from source", x_u),
            title="$u_2$ B-scan",
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    log(f"  Quick visualisation saved: {output_path}")
    return output_path


def _resolve_limits(
    data: Mapping[str, Any],
    overrides: Mapping[str, float | str | None],
    f_center: float,
    c_ref: float,
    pitch: float,
) -> dict[str, float]:
    """Dispersion limits in SI: explicit override > archive (SI or legacy) > automatic."""
    limits: dict[str, float] = {}
    for name, (si_key, legacy_key, factor) in _LIMIT_KEYS.items():
        override = overrides.get(name)
        if override is not None:
            kind = "frequency" if name.startswith("f") else "wavenumber"
            limits[name] = parse_quantity(override, kind)  # type: ignore[arg-type]
        elif si_key in data:
            limits[name] = float(data[si_key])
        elif legacy_key in data:
            limits[name] = float(data[legacy_key]) * factor
    limits.setdefault("f_min", 0.0)
    limits.setdefault("f_max", 2.0 * f_center)
    limits.setdefault("k_min", 0.0)
    limits.setdefault("k_max", min(2.0 * np.pi * limits["f_max"] / c_ref, np.pi / pitch))
    for lo, hi in (("f_min", "f_max"), ("k_min", "k_max")):
        if not np.isfinite([limits[lo], limits[hi]]).all() or not 0 <= limits[lo] < limits[hi]:
            raise ValueError("Dispersion limits must be finite with 0 <= min < max")
    return limits


def plot_full_analysis(
    sensor_data_filepath: str,
    output_dir: str | None = None,
    f_min: float | str | None = None,
    f_max: float | str | None = None,
    k_min: float | str | None = None,
    k_max: float | str | None = None,
    formats: Sequence[str] = FIGURE_FORMATS,
    usetex: bool | str = "auto",
    dpi: int = 300,
) -> str:
    """Generate the 2×2 publication figure from a ``sensor_data.npz`` archive.

    Layout:
      - Row 1: selected ``u_2`` waveforms and ``u_2`` B-scan
      - Row 2: forward-wave 2D FFT spectra of ``u_1`` and ``u_2`` (one colorbar)

    Parameters
    ----------
    sensor_data_filepath
        Path to the input archive.
    output_dir
        Destination directory (default: directory of the archive).
    f_min, f_max, k_min, k_max
        Dispersion display limits as SI numbers (Hz, rad/m) or unit strings
        (``"2 MHz"``, ``"3 rad/mm"``). Default: values stored in the archive
        (legacy MHz/rad-mm keys are understood), else automatic.
    formats
        Figure formats to write (subset of ``png, pdf, svg, eps``).
    usetex
        ``'auto'`` (LaTeX if available), ``True`` or ``False``.
    dpi
        Raster resolution.

    Returns
    -------
    str
        Path of the first written figure.
    """
    if not os.path.exists(sensor_data_filepath):
        raise FileNotFoundError(f"Data file {sensor_data_filepath} not found.")
    if output_dir is None:
        output_dir = os.path.dirname(sensor_data_filepath) or "."
    os.makedirs(output_dir, exist_ok=True)

    with np.load(sensor_data_filepath, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}

    times = np.asarray(data["times"], dtype=float)
    sensor_x = np.asarray(data["sensor_x"], dtype=float)
    sensor_data_y = np.asarray(data["sensor_data_y"], dtype=float)
    sensor_data_x = (
        np.asarray(data["sensor_data_x"], dtype=float) if "sensor_data_x" in data else None
    )
    f_center = float(data["f_center"])
    c_L = float(data["c_L"])
    c_T = float(data["c_T"])
    st_mode = str(data["structure_mode"]) if "structure_mode" in data else ""
    c_R = float(data["c_R"]) if "c_R" in data and np.isfinite(data["c_R"]) else None
    is_fluid_top = (
        bool(data["is_fluid_top"]) if "is_fluid_top" in data else c_T < FLUID_SHEAR_RATIO * c_L
    )
    has_fluid = bool(data["has_fluid"]) if "has_fluid" in data else is_fluid_top
    c_fluid = float(data["c_fluid"]) if "c_fluid" in data else (c_L if is_fluid_top else None)
    if c_R is None and st_mode == "halfspace" and "E" in data and "nu" in data:
        c_R = compute_rayleigh_speed(c_T, c_L=c_L)
    source_type = str(data["source_type"]) if "source_type" in data else "ricker"
    n_cycles = int(data["n_cycles"]) if "n_cycles" in data and int(data["n_cycles"]) > 0 else None

    # Validate before indexing/filtering. Coordinates carry physical units.
    if len(times) < 32 or len(sensor_x) < 3:
        raise ValueError("Postprocessing requires >=32 time samples and >=3 sensors")
    for axis in (times, sensor_x):
        delta = np.diff(axis)
        if (
            not np.isfinite(axis).all()
            or np.any(delta <= 0)
            or not np.allclose(delta, delta[0], rtol=1e-7, atol=0)
        ):
            raise ValueError("Postprocessing requires uniform increasing coordinates")
    for values in (sensor_data_y, sensor_data_x):
        if values is not None and (
            values.shape != (len(times), len(sensor_x)) or not np.isfinite(values).all()
        ):
            raise ValueError("Invalid displacement history shape or nonfinite values")
    if not np.isfinite(f_center) or f_center <= 0:
        raise ValueError("f_center must be finite and positive")

    s_pitch = float(data["sensor_pitch"]) if "sensor_pitch" in data else sensor_x[1] - sensor_x[0]
    c_ref = (c_fluid or c_L) if is_fluid_top else c_T
    limits = _resolve_limits(
        data,
        {"f_min": f_min, "f_max": f_max, "k_min": k_min, "k_max": k_max},
        f_center,
        c_ref,
        s_pitch,
    )

    fs = 1.0 / (times[1] - times[0])
    if f_center / 5.0 >= min(f_center * 3.0, 0.45 * fs):
        raise ValueError("Sampling frequency is too low for the requested bandpass")
    dy, _ = bandpass_filter(sensor_data_y, fs, f_center)
    dx_filt = bandpass_filter(sensor_data_x, fs, f_center)[0] if sensor_data_x is not None else None

    log(f"\n{'=' * 58}")
    log(">> Generating publication 2x2 analysis figure...")
    log(f"{'=' * 58}")
    log("  Computing forward-propagating 2D FFT dispersion (u1 and u2 separated)...")
    freqs, wavenumbers, power_u2 = dispersion_spectrum(dy, times, sensor_x)
    # Keep every bin whose cell overlaps the plot range so the image fills the axes.
    half_df, half_dk = 0.5 * (freqs[1] - freqs[0]), 0.5 * (wavenumbers[1] - wavenumbers[0])
    f_mask = (freqs >= limits["f_min"] - half_df) & (freqs <= limits["f_max"] + half_df)
    k_mask = (wavenumbers >= limits["k_min"] - half_dk) & (wavenumbers <= limits["k_max"] + half_dk)
    if not f_mask.any() or not k_mask.any():
        raise ValueError("Requested dispersion range contains no sampled FFT bins")

    # Display units chosen from the data scale.
    t_u = time_unit(times[-1])
    f_u = frequency_unit(limits["f_max"])
    k_u = wavenumber_unit(length_unit(2.0 * np.pi / limits["k_max"]))
    u_u = displacement_unit(float(np.max(np.abs(dy))))

    # Extents come from bin edges, never from the requested plot limits.
    sel_f, sel_k = f_u(freqs[f_mask]), k_u(wavenumbers[k_mask])
    df, dk = f_u(freqs[1] - freqs[0]), k_u(wavenumbers[1] - wavenumbers[0])
    extent_disp = [sel_k[0] - dk / 2, sel_k[-1] + dk / 2, sel_f[0] - df / 2, sel_f[-1] + df / 2]

    def _db(power: np.ndarray) -> np.ndarray:
        p = power[np.ix_(f_mask, k_mask)]
        return 10 * np.log10(p / (np.max(p) + 1e-30) + 1e-30)

    fk_db_u2 = _db(power_u2)
    fk_db_u1 = (
        _db(dispersion_spectrum(dx_filt, times, sensor_x)[2]) if dx_filt is not None else None
    )

    s_offset = (
        float(data["sensor_offset"])
        if "sensor_offset" in data
        else sensor_x[0] - float(data.get("source_x", 0.0))
    )
    n_sensors = dy.shape[1]
    probe_x = s_offset + np.arange(n_sensors) * s_pitch
    x_u = length_unit(float(np.max(np.abs(probe_x))))
    probe_d = x_u(probe_x)
    t_d = t_u(times)
    dy_d = u_u(dy)

    tex = resolve_usetex(usetex)
    with matplotlib.rc_context(_rc(tex)):
        fig = Figure(figsize=(190.0 / 25.4, 165.0 / 25.4))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.32, wspace=0.15)

        # (a) Selected u2 waveforms
        ax1 = fig.add_subplot(gs[0, 0])
        indices = [0, (n_sensors - 1) // 2, n_sensors - 1]
        colors = matplotlib.colormaps["viridis"](np.linspace(0.2, 0.9, 3))
        for color, idx in zip(colors, indices, strict=True):
            ax1.plot(
                t_d,
                dy_d[:, idx],
                lw=0.85,
                color=color,
                label=rf"$x = {probe_d[idx]:.4g}$ {x_u.tex}",
            )
        ax1.set_xlabel(_label("Time", t_u))
        ax1.set_ylabel(_label("Displacement $u_2$", u_u))
        ax1.set_title(_title("(a) Selected $u_2$ signals", tex))
        ax1.legend(loc="upper right", framealpha=0.85)
        ax1.grid(True, alpha=0.3, linestyle=":")
        make_axes_locatable(ax1).append_axes("right", size="3.5%", pad=0.06).axis("off")

        # (b) B-scan with arrival-time guides
        ax2 = fig.add_subplot(gs[0, 1])
        dt_d = t_d[1] - t_d[0]
        dxp = x_u(s_pitch)
        vmax = float(np.max(np.abs(dy_d))) * 0.3 or 1.0
        im2 = ax2.imshow(
            dy_d.T,
            aspect="auto",
            origin="lower",
            cmap="seismic",
            vmin=-vmax,
            vmax=vmax,
            extent=[
                t_d[0] - dt_d / 2,
                t_d[-1] + dt_d / 2,
                probe_d[0] - dxp / 2,
                probe_d[-1] + dxp / 2,
            ],
        )

        def _speed(sym: str, value: float, note: str = "") -> str:
            suffix = f" ({note})" if note else ""
            return rf"${sym} = {value:.0f}\,\mathrm{{m/s}}${suffix}"

        speed_lines = []
        # c_L/c_T describe the top plate layer; say so when several regions exist.
        layered = int(data.get("n_layers", 1)) > 1 or st_mode in (
            "plate_on_halfspace",
            "plate_in_halfspaces",
        )
        top_note = "top layer" if layered else ""
        if is_fluid_top:
            speed_lines.append((c_L, "--", "cyan", _speed("c_f", c_L, "fluid")))
        else:
            speed_lines.append((c_L, "--", "cyan", _speed("c_L", c_L, top_note)))
            speed_lines.append((c_T, ":", "lime", _speed("c_T", c_T, top_note)))
        if c_R is not None and not is_fluid_top:
            speed_lines.append((c_R, "-.", "gold", _speed("c_R", c_R, "Rayleigh")))
        elif has_fluid and not is_fluid_top and c_fluid is not None:
            speed_lines.append((c_fluid, "-.", "magenta", _speed("c_f", c_fluid, "fluid")))

        t0_pulse = source_delay(source_type, f_center, n_cycles)
        for spd, sty, clr, name in speed_lines:
            t_arr = t_u(t0_pulse + probe_x / spd)
            valid = (t_arr >= t_d[0]) & (t_arr <= t_d[-1])
            if np.any(valid):
                ax2.plot(t_arr[valid], probe_d[valid], color=clr, linestyle=sty, lw=1.1, label=name)
        ax2.set_xlim(t_d[0], t_d[-1])
        ax2.set_ylim(probe_d[0], probe_d[-1])
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax2.set_xlabel(_label("Time", t_u))
        ax2.set_ylabel(_label("Offset from source", x_u))
        ax2.set_title(_title("(b) B-scan ($u_2$)", tex))
        if ax2.get_legend_handles_labels()[0]:
            ax2.legend(loc="lower right", framealpha=0.85)
        cb2 = fig.colorbar(
            im2, cax=make_axes_locatable(ax2).append_axes("right", size="3.5%", pad=0.06)
        )
        cb2.set_label(_label("$u_2$", u_u))

        disp_lines = []
        if is_fluid_top:
            disp_lines.append((c_L, "--", "cyan", _speed("c_f", c_L)))
        elif c_R is not None:
            disp_lines.append((c_R, "--", "cyan", _speed("c_R", c_R)))
        if has_fluid and not is_fluid_top and c_fluid is not None:
            disp_lines.append((c_fluid, ":", "magenta", _speed("c_f", c_fluid)))

        k_lo, k_hi = k_u(limits["k_min"]), k_u(limits["k_max"])
        f_lo, f_hi = f_u(limits["f_min"]), f_u(limits["f_max"])

        def _dispersion_axes(ax: Axes, image: np.ndarray | None, title: str):
            im = None
            if image is not None:
                im = ax.imshow(
                    image,
                    aspect="auto",
                    extent=extent_disp,
                    origin="lower",
                    cmap="inferno",
                    vmin=-40,
                    vmax=0,
                )
            k_line = np.linspace(limits["k_min"], limits["k_max"], 200)
            for spd, sty, clr, lbl in disp_lines:
                f_line = f_u(spd * k_line / (2.0 * np.pi))
                ok = (f_line >= f_lo) & (f_line <= f_hi)
                if np.any(ok):
                    ax.plot(
                        k_u(k_line[ok]), f_line[ok], sty, color=clr, lw=1.1, alpha=0.9, label=lbl
                    )
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc="upper left", framealpha=0.8)
            ax.set_xlim(k_lo, k_hi)
            ax.set_ylim(f_lo, f_hi)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.set_xlabel(_label("Wavenumber", k_u))
            ax.set_title(_title(title, tex))
            return im

        ax3 = fig.add_subplot(gs[1, 0])
        _dispersion_axes(ax3, fk_db_u1, "(c) Dispersion ($u_1$)")
        ax3.set_ylabel(_label("Frequency", f_u))
        make_axes_locatable(ax3).append_axes("right", size="3.5%", pad=0.06).axis("off")

        ax4 = fig.add_subplot(gs[1, 1], sharey=ax3)
        im4 = _dispersion_axes(ax4, fk_db_u2, "(d) Dispersion ($u_2$)")
        cb4 = fig.colorbar(
            im4, cax=make_axes_locatable(ax4).append_axes("right", size="3.5%", pad=0.06)
        )
        cb4.set_label("Relative power [dB, per component]")

        paths = _save(fig, os.path.join(output_dir, "full_analysis"), formats, dpi)
    return paths[0]
