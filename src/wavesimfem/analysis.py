"""Signal processing, 2D FFT dispersion analysis, and publication figures.

All exports use a 190 mm canvas and 10-point LaTeX typography. Plotting settings
are scoped with :func:`matplotlib.pyplot.rc_context` and never leak to the caller.
Axis units are chosen from the data (e.g. s/ms/μs, m/mm, Hz/kHz/MHz, rad/m or
rad/mm), so seismic and ultrasonic records are both displayed naturally.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import RLock
from typing import Any

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import butter, sosfiltfilt

from wavesimfem.materials import compute_rayleigh_speed
from wavesimfem.numerics import dispersion_spectrum
from wavesimfem.reporting import log, warn
from wavesimfem.sources import describe_source, source_delay, source_duration, source_signal
from wavesimfem.storage import atomic_output
from wavesimfem.units import (
    Unit,
    fmt_frequency,
    frequency_unit,
    length_unit,
    observable_unit,
    parse_quantity,
    time_unit,
    wavenumber_unit,
)

# Figures are built with the object-oriented API (``Figure`` + its own canvas),
# so no pyplot state or global backend selection is involved. Matplotlib's
# rcParams and rendering internals are global: serialize both plotting paths.
_PLOT_LOCK = RLock()

FIGURE_FORMATS = ("png", "pdf", "svg", "eps")
FIGURE_WIDTH_MM = 190.0
FIGURE_WIDTH_IN = FIGURE_WIDTH_MM / 25.4
FONT_SIZE_PT = 10.0

_BASE_RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlepad": 4,
    "axes.labelpad": 3,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
    "axes.edgecolor": "#7C8794",
    "axes.linewidth": 0.7,
    "axes.labelcolor": "#233044",
    "text.color": "#233044",
    "xtick.color": "#536273",
    "ytick.color": "#536273",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.pad": 2.5,
    "ytick.major.pad": 2.5,
    "legend.fontsize": 10,
    "legend.title_fontsize": 10,
    "legend.borderpad": 0.3,
    "legend.labelspacing": 0.3,
    "legend.handlelength": 1.5,
    "legend.columnspacing": 0.8,
    "legend.frameon": False,
    "grid.color": "#E3E8ED",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "lines.linewidth": 1.4,
    "image.interpolation": "nearest",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "text.usetex": False,
    "savefig.bbox": None,
    "savefig.pad_inches": 0,
}

TRACE_COLORS = ("#167D9A", "#D97732", "#7356A6")


# Vertical-component axis labels for each archived physical observable.
_AMPLITUDE_LABELS = {
    "displacement": "Displacement $u_2$",
    "pressure": "Pressure $p$",
    "velocity": "Velocity $v_2$",
    "particle_velocity": "Particle velocity $v_2$",
}


def _figure(height_mm: float, **kwargs) -> Figure:
    """Build a publication canvas at its final physical size, before layout."""
    return Figure(figsize=(FIGURE_WIDTH_IN, height_mm / 25.4), **kwargs)


def _line_axes(ax: Axes) -> None:
    """Quiet reference grid for quantitative line charts (not heatmaps)."""
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.margins(x=0.02)


def _robust_limit(values: np.ndarray) -> float:
    """A shared signed colour limit, with an explicit nonzero empty-data fallback."""
    peak = float(np.max(np.abs(values)))
    return max(float(np.percentile(np.abs(values), 99)), 0.05 * peak) if peak else 1.0


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


def resolve_usetex(usetex: bool | str = False) -> bool:
    """Resolve optional external LaTeX rendering, disabled by default."""
    if isinstance(usetex, str):
        if usetex.lower() != "auto":
            raise ValueError("usetex must be 'auto', True or False")
        return latex_available()
    elif not isinstance(usetex, (bool, np.bool_)):
        raise ValueError("usetex must be 'auto', True or False")
    if not usetex:
        return False
    if not latex_available():
        raise RuntimeError(
            "usetex=True requires LaTeX, dvipng and Ghostscript on PATH. "
            "Install these plotting dependencies or set output.usetex to false."
        )
    return True


def _rc(usetex: bool = False) -> dict[str, object]:
    return {
        **_BASE_RC,
        "text.usetex": usetex,
        "font.serif": ["Computer Modern Roman"] if usetex else ["DejaVu Serif"],
    }


def _title(text: str, usetex: bool) -> str:
    """Bold title text in both rendering modes."""
    return rf"\textbf{{{text}}}" if usetex else text


def _latex_text(text: str) -> str:
    """Escape literal labels while preserving intentional math and TeX commands."""
    substitutions = {
        "μ": r"\ensuremath{\mu}",
        "µ": r"\ensuremath{\mu}",
        "²": r"\ensuremath{^{2}}",
        "³": r"\ensuremath{^{3}}",
        "≥": r"\ensuremath{\geq}",
        "≤": r"\ensuremath{\leq}",
        "Δ": r"\ensuremath{\Delta}",
        "λ": r"\ensuremath{\lambda}",
        "α": r"\ensuremath{\alpha}",
        "β": r"\ensuremath{\beta}",
        "θ": r"\ensuremath{\theta}",
        "π": r"\ensuremath{\pi}",
        "−": "-",
        "×": r"\ensuremath{\times}",
        "°": r"\ensuremath{^{\circ}}",
        "≈": r"\ensuremath{\approx}",
        "→": r"\ensuremath{\rightarrow}",
    }
    for char, replacement in substitutions.items():
        text = text.replace(char, replacement)
    pieces = re.split(r"(?<!\\)\$", text)
    for i, piece in enumerate(pieces):
        piece = re.sub(r"(?<!\\)([%&#])", r"\\\1", piece)
        if i % 2 == 0:
            piece = re.sub(r"(?<!\\)_", r"\_", piece)
        pieces[i] = piece
    return "$".join(pieces)


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
    formats = tuple(ext.lower().lstrip(".") for ext in formats)
    if not formats:
        raise ValueError("At least one figure format is required")
    for ext in formats:
        if ext not in FIGURE_FORMATS:
            raise ValueError(f"Unsupported figure format '{ext}'")
    # Preserve the figure's rendering mode even outside its creation context.
    text = next(iter(fig.findobj(match=Text)), None)
    tex = resolve_usetex(text.get_usetex() if text is not None else False)
    fig.set_size_inches(FIGURE_WIDTH_IN, fig.get_figheight(), forward=False)
    # Tick labels can be created lazily. Set their defaults as well as all
    # existing Text objects, including legends and colorbar offset numbers.
    for ax in fig.axes:
        ax.tick_params(axis="both", which="both", labelsize=FONT_SIZE_PT)
        ax.get_xticklabels()
        ax.get_yticklabels()
        title = ax.get_title()
        if tex and title and not title.startswith(r"\textbf{"):
            ax.set_title(_title(title, True))
    for artist in fig.findobj(match=Text):
        artist.set_fontsize(FONT_SIZE_PT)
        artist.set_fontfamily("serif")
        artist.set_usetex(tex)
        if tex:
            artist.set_text(_latex_text(artist.get_text()))
    paths = []
    ps_logger = logging.getLogger("matplotlib.backends.backend_ps")
    for ext in formats:
        path = f"{base}.{ext}"
        # EPS has no alpha channel; translucent legend frames/grid lines are
        # flattened to opaque by design, so the backend's warning is suppressed.
        previous = ps_logger.level
        if ext == "eps":
            ps_logger.setLevel(logging.ERROR)
        try:
            with matplotlib.rc_context(_rc(tex)), atomic_output(path, binary=True) as stream:
                fig.savefig(stream, format=ext, dpi=dpi, bbox_inches=None)
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
    *,
    usetex: bool | str = False,
    observable: str = "displacement",
) -> str:
    """Compact acquisition diagnostic with a shared physical amplitude scale."""
    times, sensor_x, values = [np.asarray(v, dtype=float) for v in (times, sensor_x, sensor_data_y)]
    if times.ndim != 1 or sensor_x.ndim != 1 or len(times) < 2 or not len(sensor_x):
        raise ValueError("Overview requires at least two times and one sensor")
    if values.shape != (len(times), len(sensor_x)) or not np.isfinite(values).all():
        raise ValueError("Invalid displacement history shape or nonfinite values")
    for axis in (times, sensor_x):
        delta = np.diff(axis)
        if not np.isfinite(axis).all() or (
            len(delta)
            and (np.any(delta <= 0) or not np.allclose(delta, delta[0], rtol=1e-7, atol=0))
        ):
            raise ValueError("Overview requires uniform increasing coordinates")
    if not np.isfinite([f_center, source_x]).all() or f_center <= 0:
        raise ValueError("Source frequency must be positive and source position finite")
    offsets = sensor_x - source_x
    fs = 1 / (times[1] - times[0])
    filtered, band = bandpass_filter(values, fs, f_center)
    tu, xu = time_unit(np.max(np.abs(times))), length_unit(np.max(np.abs(offsets)))
    uu = observable_unit(np.max(np.abs(filtered)), observable)
    amplitude_label = _AMPLITUDE_LABELS[observable]
    td, xd, ud = tu(times), xu(offsets), uu(filtered)
    with _PLOT_LOCK, matplotlib.rc_context(_rc(resolve_usetex(usetex))):
        fig = _figure(135, layout="constrained")
        fig.get_layout_engine().set(w_pad=0.035, h_pad=0.035, wspace=0.035, hspace=0.035)
        gs = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.4])
        source_ax, trace_ax, scan_ax = (
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, :]),
        )
        pulse = source_duration(source_type, f_center, n_cycles)
        te = np.linspace(0, 1.3 * pulse, 1000)
        eu = time_unit(te[-1])
        source_ax.plot(
            eu(te), source_signal(te, source_type, f_center, n_cycles), color=TRACE_COLORS[0]
        )
        source_ax.set(
            xlabel=_label("Time", eu),
            ylabel="Source amplitude",
            title="(a) Excitation",
        )
        selected = sorted(set([0, len(offsets) // 2, len(offsets) - 1]))
        for color, j in zip(TRACE_COLORS, selected, strict=False):
            trace_ax.plot(td, ud[:, j], color=color, label=f"{xd[j]:.3g} {xu.tex}")
        trace_ax.set(
            xlabel=_label("Time", tu),
            ylabel=_label(amplitude_label, uu),
            title="(b) Receiver signals",
        )
        trace_ax.legend(title="Source offset", title_fontsize=10, loc="upper right")
        for ax in (source_ax, trace_ax):
            _line_axes(ax)
        dt = td[1] - td[0]
        pitch = xd[1] - xd[0] if len(xd) > 1 else 1.0
        limit = _robust_limit(ud)
        image = scan_ax.imshow(
            ud.T,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            extent=[td[0] - dt / 2, td[-1] + dt / 2, xd[0] - pitch / 2, xd[-1] + pitch / 2],
        )
        scan_ax.set(
            xlabel=_label("Time", tu),
            ylabel=_label("Offset from source", xu),
            title="(c) Recorded wavefield",
            xlim=(td[0], td[-1]),
        )
        fig.colorbar(
            image,
            ax=scan_ax,
            label=_label(amplitude_label, uu),
            extend="neither",
            pad=0.02,
        )
        extension = os.path.splitext(output_path)[1].lower().lstrip(".") or "png"
        _save(fig, os.path.splitext(output_path)[0], [extension], 300)
        notes = Path(output_path).with_suffix("").with_name(Path(output_path).stem + "_README.md")
        with atomic_output(notes) as stream:
            stream.write(
                "# Acquisition overview\n\n"
                f"Source: {describe_source(source_type, n_cycles)}, {fmt_frequency(f_center)}; "
                f"{len(offsets)} receivers.\n\n"
                "(a) Normalized source waveform. (b) Selected receiver traces. "
                "(c) Recorded vertical wavefield on one physical amplitude scale.\n\n"
                f"Zero-phase{_band_text(band)}. Flat-ended colourbars saturate beyond "
                f"the robust display limits of +/-{limit:.6g} {uu.label}; "
                "recorded amplitudes are unchanged. "
                "Empty fields indicate no recorded dynamic signal.\n\n"
                "Export: 190 mm width, 10-point LaTeX text, 300 dpi raster.\n"
            )
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
    # A bulk shear speed is not a lower bound on guided-wave phase velocity.
    # Retain the available spatial band instead of silently cropping slow modes.
    limits.setdefault("k_max", np.pi / pitch)
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
    usetex: bool | str = False,
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
        False (default) uses built-in fonts/mathtext; True requires external
        LaTeX. ``'auto'`` uses LaTeX only when its tools are available.
        Both modes use the same 190 mm, 10-point publication style.
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
    observable = str(data.get("sensor_observable", "displacement"))
    keys = {
        "displacement": ("sensor_data_x", "sensor_data_y"),
        "pressure": (None, "sensor_data_pressure"),
        "velocity": ("sensor_data_vx", "sensor_data_vy"),
        "particle_velocity": ("sensor_data_vx", "sensor_data_vy"),
    }
    if observable not in keys:
        raise ValueError(f"Unsupported sensor observable {observable!r}")
    key_x, key_y = keys[observable]
    sensor_data_y = np.asarray(data[key_y], dtype=float)
    sensor_data_x = np.asarray(data[key_x], dtype=float) if key_x in data else None
    f_center = float(data["f_center"])
    c_L = float(data["c_L"])
    c_T = float(data["c_T"]) if "c_T" in data else None
    st_mode = str(data["structure_mode"]) if "structure_mode" in data else ""
    c_R = float(data["c_R"]) if "c_R" in data and np.isfinite(data["c_R"]) else None
    is_fluid_top = (
        bool(data["is_fluid_top"])
        if "is_fluid_top" in data
        else c_T is None or c_T < FLUID_SHEAR_RATIO * c_L
    )
    has_fluid = bool(data["has_fluid"]) if "has_fluid" in data else is_fluid_top
    c_fluid = float(data["c_fluid"]) if "c_fluid" in data else (c_L if is_fluid_top else None)
    if c_R is None and c_T is not None and st_mode == "halfspace" and "E" in data and "nu" in data:
        c_R = compute_rayleigh_speed(c_T, c_L=c_L)
    source_type = str(data["source_type"]) if "source_type" in data else "ricker"
    n_cycles = int(data["n_cycles"]) if "n_cycles" in data and int(data["n_cycles"]) > 0 else None

    # Validate before indexing/filtering. Coordinates carry physical units.
    if times.ndim != 1 or sensor_x.ndim != 1 or len(times) < 32 or len(sensor_x) < 3:
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
            raise ValueError("Invalid observable history shape or nonfinite values")
    if not np.isfinite(f_center) or f_center <= 0:
        raise ValueError("f_center must be finite and positive")

    s_pitch = float(sensor_x[1] - sensor_x[0])
    if "source_x" in data:
        probe_x = sensor_x - float(data["source_x"])
    else:
        probe_x = sensor_x - sensor_x[0] + float(data.get("sensor_offset", sensor_x[0]))
    if not np.isfinite(probe_x).all():
        raise ValueError("Source position and receiver offsets must be finite")
    if not np.isfinite(c_L) or c_L <= 0 or (c_T is not None and (not np.isfinite(c_T) or c_T < 0)):
        raise ValueError("Wave speeds must be finite and physical")
    if not is_fluid_top and c_T is None:
        raise ValueError("A solid top layer requires an archived shear speed")
    c_ref = (c_fluid or c_L) if is_fluid_top else c_T
    limits = _resolve_limits(
        data,
        {"f_min": f_min, "f_max": f_max, "k_min": k_min, "k_max": k_max},
        f_center,
        c_ref,
        s_pitch,
    )

    fs = 1.0 / (times[1] - times[0])
    for key, ceiling in (("f_max", fs / 2), ("k_max", np.pi / s_pitch)):
        if limits[key] > ceiling:
            warn(f"{key} exceeds the sampled Nyquist limit; displaying up to {ceiling:g} SI.")
            limits[key] = ceiling
        if limits[key.replace("max", "min")] >= limits[key]:
            raise ValueError("Requested dispersion range is outside the sampled Nyquist limits")
    if f_center / 5.0 >= min(f_center * 3.0, 0.45 * fs):
        raise ValueError("Sampling frequency is too low for the requested bandpass")
    dy, filter_band = bandpass_filter(sensor_data_y, fs, f_center)
    dx_filt = bandpass_filter(sensor_data_x, fs, f_center)[0] if sensor_data_x is not None else None

    log(f"\n{'=' * 58}")
    log(">> Generating publication 2x2 analysis figure...")
    log(f"{'=' * 58}")
    log(f"  Computing forward-propagating 2D FFT dispersion ({observable})...")
    leftward = bool(np.all(probe_x <= 0) and np.any(probe_x < 0))
    direction = "-x" if leftward else "+x"
    fft_positions = -sensor_x[::-1] if leftward else sensor_x

    def spectrum(values):
        return dispersion_spectrum(values[:, ::-1] if leftward else values, times, fft_positions)

    freqs, wavenumbers, power_u2 = spectrum(dy)
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
    u_u = observable_unit(float(np.max(np.abs(dy))), observable)
    amplitude_label = _AMPLITUDE_LABELS[observable]
    symbol_x, symbol_y = ("u_1", "u_2") if observable == "displacement" else ("v_1", "v_2")
    if observable == "pressure":
        symbol_y = "p"

    # Extents come from bin edges, never from the requested plot limits.
    sel_f, sel_k = f_u(freqs[f_mask]), k_u(wavenumbers[k_mask])
    df, dk = f_u(freqs[1] - freqs[0]), k_u(wavenumbers[1] - wavenumbers[0])
    extent_disp = [sel_k[0] - dk / 2, sel_k[-1] + dk / 2, sel_f[0] - df / 2, sel_f[-1] + df / 2]

    def _db(power: np.ndarray) -> np.ndarray:
        p = power[np.ix_(f_mask, k_mask)]
        peak = float(np.max(p))
        relative = p / peak if peak > 0 else np.zeros_like(p)
        return 10 * np.log10(np.maximum(relative, 1e-30))

    fk_db_u2 = _db(power_u2)
    fk_db_u1 = _db(spectrum(dx_filt)[2]) if dx_filt is not None else None

    n_sensors = dy.shape[1]
    x_u = length_unit(float(np.max(np.abs(probe_x))))
    probe_d = x_u(probe_x)
    t_d = t_u(times)
    dy_d = u_u(dy)

    tex = resolve_usetex(usetex)
    with _PLOT_LOCK, matplotlib.rc_context(_rc(tex)):
        # Physical geometry: equal square plotting areas, including the width
        # reserved by each axes divider for a colourbar (or its blank spacer).
        left_mm, right_mm, column_gap_mm = 16.0, 18.0, 18.0
        bottom_mm, top_mm, row_gap_mm = 14.0, 8.0, 18.0
        slot_mm = (FIGURE_WIDTH_MM - left_mm - right_mm - column_gap_mm) / 2
        panel_mm = (slot_mm - 0.04 * 25.4) / 1.035
        height_mm = bottom_mm + top_mm + row_gap_mm + 2 * panel_mm
        fig = _figure(height_mm)
        gs = fig.add_gridspec(
            2,
            2,
            left=left_mm / FIGURE_WIDTH_MM,
            right=1 - right_mm / FIGURE_WIDTH_MM,
            bottom=bottom_mm / height_mm,
            top=1 - top_mm / height_mm,
            hspace=row_gap_mm / panel_mm,
            wspace=column_gap_mm / slot_mm,
        )
        # (a) Selected u2 waveforms
        ax1 = fig.add_subplot(gs[0, 0])
        indices = [0, (n_sensors - 1) // 2, n_sensors - 1]
        colors = TRACE_COLORS
        for color, idx in zip(colors, indices, strict=True):
            ax1.plot(
                t_d,
                dy_d[:, idx],
                lw=1.25,
                color=color,
                label=rf"$x = {probe_d[idx]:.4g}$ {x_u.tex}",
            )
        ax1.set_xlabel(_label("Time", t_u))
        ax1.set_ylabel(_label(amplitude_label, u_u))
        ax1.set_title(_title("(a) Receiver signals", tex))
        ax1.legend(loc="upper right")
        _line_axes(ax1)
        ax1.set_xlim(t_d[0], t_d[-1])
        make_axes_locatable(ax1).append_axes("right", size="3.5%", pad=0.04).axis("off")

        # (b) B-scan with arrival-time guides
        ax2 = fig.add_subplot(gs[0, 1])
        dt_d = t_d[1] - t_d[0]
        dxp = x_u(s_pitch)
        vmax = _robust_limit(dy_d)
        im2 = ax2.imshow(
            dy_d.T,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
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
            speed_lines.append((c_L, "--", "#167D9A", _speed("c_f", c_L, "fluid")))
        else:
            speed_lines.append((c_L, "--", "#167D9A", _speed("c_L", c_L, top_note)))
            speed_lines.append((c_T, ":", "#D97732", _speed("c_T", c_T, top_note)))
        if c_R is not None and not is_fluid_top:
            speed_lines.append((c_R, "-.", "#7356A6", _speed("c_R", c_R, "Rayleigh")))
        elif has_fluid and not is_fluid_top and c_fluid is not None:
            speed_lines.append((c_fluid, "-.", "#7356A6", _speed("c_f", c_fluid, "fluid")))

        t0_pulse = source_delay(source_type, f_center, n_cycles)
        for spd, sty, clr, name in speed_lines:
            t_arr = t_u(t0_pulse + np.abs(probe_x) / spd)
            valid = (t_arr >= t_d[0]) & (t_arr <= t_d[-1])
            if np.any(valid):
                ax2.plot(t_arr[valid], probe_d[valid], color=clr, linestyle=sty, lw=1.1, label=name)
        ax2.set_xlim(t_d[0], t_d[-1])
        ax2.set_ylim(probe_d[0], probe_d[-1])
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax2.set_xlabel(_label("Time", t_u))
        ax2.set_ylabel(_label("Offset from source", x_u))
        ax2.set_title(_title(f"(b) B-scan (${symbol_y}$)", tex))
        if ax2.get_legend_handles_labels()[0]:
            ax2.legend(
                loc="lower right",
                frameon=True,
                framealpha=0.95,
                facecolor="white",
                edgecolor="none",
                fontsize=10,
            )
        cb2 = fig.colorbar(
            im2,
            cax=make_axes_locatable(ax2).append_axes("right", size="3.5%", pad=0.04),
            extend="neither",
        )
        cb2.set_label(_label(f"${symbol_y}$", u_u))

        disp_lines = []
        if is_fluid_top:
            disp_lines.append((c_L, "--", "#167D9A", _speed("c_f", c_L)))
        elif c_R is not None:
            disp_lines.append((c_R, "--", "#167D9A", _speed("c_R", c_R)))
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
            else:
                ax.set_facecolor("#F3F6F8")
            k_line = np.linspace(limits["k_min"], limits["k_max"], 200)
            for spd, sty, clr, lbl in disp_lines:
                f_line = f_u(spd * k_line / (2.0 * np.pi))
                ok = (f_line >= f_lo) & (f_line <= f_hi)
                if np.any(ok):
                    ax.plot(
                        k_u(k_line[ok]), f_line[ok], sty, color=clr, lw=1.1, alpha=0.9, label=lbl
                    )
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.95)
            ax.set_xlim(k_lo, k_hi)
            ax.set_ylim(f_lo, f_hi)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.set_xlabel(_label("Wavenumber", k_u))
            ax.set_title(_title(title, tex))
            return im

        ax3 = fig.add_subplot(gs[1, 0])
        if observable == "pressure":
            tapered = (dy - dy.mean(axis=0)) * np.hanning(len(times))[:, None]
            temporal_power = np.mean(
                np.abs(np.fft.rfft(tapered, n=2 * len(times), axis=0)) ** 2, axis=1
            )
            peak = float(temporal_power.max())
            relative = temporal_power / peak if peak > 0 else np.zeros_like(temporal_power)
            temporal_freqs = np.fft.rfftfreq(2 * len(times), d=1 / fs)
            ax3.plot(
                f_u(temporal_freqs), 10 * np.log10(np.maximum(relative, 1e-30)), color="#167D9A"
            )
            ax3.set_xlim(f_lo, f_hi)
            ax3.set_ylim(-60, 1)
            ax3.set_xlabel(_label("Frequency", f_u))
            ax3.set_ylabel("Relative power [dB]")
            ax3.set_title(_title("(c) Pressure spectrum", tex))
            ax3.grid(True)
        else:
            _dispersion_axes(ax3, fk_db_u1, f"(c) Dispersion (${symbol_x}$)")
            ax3.set_ylabel(_label("Frequency", f_u))
        make_axes_locatable(ax3).append_axes("right", size="3.5%", pad=0.04).axis("off")

        ax4 = fig.add_subplot(gs[1, 1], sharey=None if observable == "pressure" else ax3)
        im4 = _dispersion_axes(ax4, fk_db_u2, f"(d) Dispersion (${symbol_y}$)")
        cb4 = fig.colorbar(
            im4, cax=make_axes_locatable(ax4).append_axes("right", size="3.5%", pad=0.04)
        )
        cb4.set_label("Relative power [dB]")
        ax4.set_ylabel(_label("Frequency", f_u))

        paths = _save(fig, os.path.join(output_dir, "full_analysis"), formats, dpi)
        with atomic_output(Path(output_dir) / "full_analysis_README.md") as stream:
            stream.write(
                "# Wave propagation and dispersion\n\n"
                f"Archive: `{Path(sensor_data_filepath).resolve()}`.\n\n"
                f"{n_sensors} receivers; {len(times)} samples; source {fmt_frequency(f_center)}. "
                f"Propagation: {direction}.\n\n"
                f"Observable: {observable}, units {data.get('sensor_units', 'm')}. "
                "(a) Selected receiver signals. (b) B-scan with reference "
                "arrival-time guides. "
                + (
                    "(c) Mean temporal pressure FFT power across receivers, normalized by its peak. "
                    "(d) Pressure dispersion.\n\n"
                    if observable == "pressure"
                    else "(c,d) Available vector components and dispersion.\n\n"
                )
                + f"Zero-phase{_band_text(filter_band)}. Hann taper; 2x zero-padding interpolates "
                f"only. Unpadded DFT spacing: {fs / len(times):.6g} Hz and "
                f"{2 * np.pi / (n_sensors * s_pitch):.6g} rad/m.\n\n"
                "Bulk-speed guides refer to the archived top layer where layered; they are "
                "not a layered-system dispersion solution. Each dispersion component has its "
                "own peak power reference. Flat-ended amplitude colourbars saturate beyond "
                f"the robust display limits of +/-{vmax:.6g} {u_u.label}; "
                "recorded amplitudes are unchanged. "
                "Absent components are blank; zero signals remain at the spectral floor.\n\n"
                f"Archive provenance: {data.get('provenance', 'Recorded simulation archive')}.\n\n"
                "Export: 190 mm width and 10-point LaTeX text.\n"
            )
    return paths[0]
