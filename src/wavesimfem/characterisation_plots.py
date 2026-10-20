"""Publication panels for measured packets; interpretation is saved in the README."""

from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.signal import spectrogram

from wavesimfem.analysis import (
    TRACE_COLORS,
    _figure,
    _label,
    _line_axes,
    _robust_limit,
    _save,
)
from wavesimfem.units import frequency_unit, length_unit, observable_unit, time_unit


def render_characterisation(report, arrays, destination, formats, dpi=300):
    """Render four 190 mm panel figures under the caller's typography lock."""
    t, x, ux, uy, envelope = arrays
    component = report["options"]["component"]
    signal = ux if component == "x" else uy
    rows, summary = report["receivers"], report["summary"]
    valid = np.array([row["included_in_fit"] for row in rows])
    arrival = np.array([r["arrival_s"] if r["arrival_s"] is not None else np.nan for r in rows])
    amplitude = np.array([r.get("peak_envelope_si", r.get("peak_envelope_m")) for r in rows])
    tu, xu = time_unit(np.max(np.abs(t))), length_unit(np.max(np.abs(x)))
    observable = report.get("observable", "displacement")
    uu = observable_unit(max(np.max(np.abs(ux)), np.max(np.abs(uy))), observable)
    symbol = "u" if observable == "displacement" else "v"
    component_label = "$p$" if observable == "pressure" else f"${symbol}_{component}$"
    td, xd = tu(t), xu(x)
    selected = np.unique(np.linspace(0, len(x) - 1, min(4, len(x)), dtype=int))
    velocity, attenuation = summary["velocity"], summary["apparent_attenuation"]
    paths = []
    band = report["options"]["frequency_band"]

    def finish(fig, name):
        paths.extend(_save(fig, str(destination / name), formats, dpi))

    fig = _figure(150)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.105,
        right=0.925,
        top=0.955,
        bottom=0.10,
        hspace=0.37,
        wspace=0.32,
        height_ratios=[1.15, 1],
    )
    scan = fig.add_subplot(gs[0, :])
    fit_ax, residual_ax = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    data = uu(signal)
    limit = _robust_limit(data)
    image = scan.pcolormesh(
        td, xd, data.T, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto", rasterized=True
    )
    scan.plot(
        tu(arrival[valid]),
        xd[valid],
        "o",
        ms=3,
        mfc="white",
        mec="#233044",
        mew=0.7,
        label="Accepted",
    )
    if (~valid).any():
        scan.plot(tu(arrival[~valid]), xd[~valid], "x", ms=4, color="#7356A6", label="Excluded")
    scan.set(
        xlabel=_label("Time", tu), ylabel=_label("Receiver $x$", xu), title="(a) Packet tracking"
    )
    scan.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95, ncols=2)
    fig.colorbar(
        image,
        ax=scan,
        label=_label(component_label, uu),
        pad=0.015,
        fraction=0.035,
        extend="neither",
    )
    fit_ax.plot(xd[valid], tu(arrival[valid]), "o", ms=4, color=TRACE_COLORS[0], label="Accepted")
    if (~valid).any():
        fit_ax.plot(xd[~valid], tu(arrival[~valid]), "x", color="#8A95A3", label="Excluded")
    if velocity:
        fitted = velocity["intercept_s"] + x / velocity["signed_m_per_s"]
        fit_ax.plot(
            xd,
            tu(fitted),
            color=TRACE_COLORS[1],
            label=rf"$v={velocity['signed_m_per_s']:.4g}$ m/s"
            + "\n"
            + rf"$R^2={velocity['r_squared']:.3f}$",
        )
        residual_ax.axhline(0, lw=1, color="#8A95A3")
        residual_ax.plot(
            xd[valid], tu(arrival[valid] - fitted[valid]), "o", ms=4, color=TRACE_COLORS[0]
        )
        residual_ax.axhspan(
            -tu(t[1] - t[0]), tu(t[1] - t[0]), color="#E9F2F5", label=r"$\pm\Delta t$"
        )
        residual_ax.legend(loc="best")
    fit_ax.set(
        xlabel=_label("Receiver $x$", xu),
        ylabel=_label("Peak arrival", tu),
        title="(b) Packet velocity" + ("" if velocity else " (N/A)"),
    )
    fit_ax.legend(loc="best")
    residual_ax.set(
        xlabel=_label("Receiver $x$", xu),
        ylabel=_label("Arrival residual", tu),
        title="(c) Fit residuals" + ("" if velocity else " (N/A)"),
    )
    for ax in (fit_ax, residual_ax):
        _line_axes(ax)
    finish(fig, "arrival_velocity")

    fig = _figure(148)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.105,
        right=0.98,
        bottom=0.10,
        top=0.955,
        hspace=0.37,
        wspace=0.32,
    )
    axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for color, i in zip((*TRACE_COLORS, "#526273"), selected, strict=False):
        axes[0].plot(
            td, envelope[:, i] / (amplitude[i] or 1), color=color, label=f"$x={xd[i]:.3g}$ {xu.tex}"
        )
    axes[0].set(
        xlabel=_label("Time", tu),
        ylabel="Normalised envelope",
        title="(a) Packet shape",
        ylim=(0, 1.12),
    )
    axes[0].legend(loc="upper right", ncols=2)
    reference = amplitude.max() or 1
    nonzero = amplitude > 0
    amplitude_db = 20 * np.log10(np.maximum(amplitude / reference, 1e-15))
    axes[1].plot(
        xd[nonzero & valid],
        amplitude_db[nonzero & valid],
        "o",
        ms=4,
        color=TRACE_COLORS[0],
        label="Accepted",
    )
    if (nonzero & ~valid).any():
        axes[1].plot(
            xd[nonzero & ~valid],
            amplitude_db[nonzero & ~valid],
            "x",
            color="#8A95A3",
            label="Excluded",
        )
    if attenuation:
        log_prediction = (
            attenuation["log_amplitude_intercept"]
            - attenuation["np_per_m"] * attenuation["propagation_direction"] * x
        )
        axes[1].plot(
            xd,
            20 / np.log(10) * (log_prediction - np.log(reference)),
            color=TRACE_COLORS[1],
            label=rf"$\alpha={attenuation['db_per_m']:.3g}$ dB/m",
        )
    axes[1].set(
        xlabel=_label("Receiver $x$", xu),
        ylabel="Relative peak [dB]",
        title="(b) Amplitude decay" + ("" if attenuation else " (N/A)"),
    )
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend()
    widths = np.array(
        [
            r["envelope_rms_width_s"] if r["envelope_rms_width_s"] is not None else np.nan
            for r in rows
        ]
    )
    axes[2].plot(xd[valid], tu(widths[valid]), "o", ms=4, color=TRACE_COLORS[0])
    axes[2].plot(xd[~valid], tu(widths[~valid]), "x", color="#8A95A3")
    axes[2].set(
        xlabel=_label("Receiver $x$", xu),
        ylabel=_label("RMS width", tu),
        title="(c) Pulse spreading",
    )
    width_limit = float(np.nanmax(tu(widths))) * 1.15 if np.isfinite(widths).any() else 1.0
    axes[2].set_ylim(0, width_limit or 1.0)
    for ax in axes:
        _line_axes(ax)
    finish(fig, "pulse_evolution")

    # Two columns preserve room for 10 pt labels at the fixed publication width.
    nrows = int(np.ceil(len(selected) / 2))
    if observable != "pressure":
        fig = _figure(76 * nrows)
        gs = fig.add_gridspec(
            nrows,
            2,
            left=0.105,
            right=0.88,
            bottom=0.105,
            top=0.955,
            hspace=0.34,
            wspace=0.35,
        )
        axes = [fig.add_subplot(gs[j // 2, j % 2]) for j in range(len(selected))]
        motion_limit = (
            max(
                float(np.max(np.abs(uu(ux[:, selected])))),
                float(np.max(np.abs(uu(uy[:, selected])))),
            )
            * 1.08
            or 1
        )
        norm = Normalize(td[0], td[-1])
        image = None
        for j, (ax, i) in enumerate(zip(axes, selected, strict=True)):
            points = np.column_stack((uu(ux[:, i]), uu(uy[:, i])))
            segments = np.stack((points[:-1], points[1:]), axis=1)
            active = (envelope[:-1, i] > 0.05 * amplitude[i]) | (
                envelope[1:, i] > 0.05 * amplitude[i]
            )
            image = LineCollection(
                segments[active], cmap="viridis", norm=norm, linewidths=1.2, rasterized=True
            )
            image.set_array(td[:-1][active])
            ax.add_collection(image)
            ax.set(
                xlabel=_label(f"${symbol}_x$", uu),
                ylabel=_label(f"${symbol}_y$", uu),
                title=f"({chr(97 + j)}) $x={xd[i]:.3g}$ {xu.tex}",
                xlim=(-motion_limit, motion_limit),
                ylim=(-motion_limit, motion_limit),
            )
            ax.set_aspect("equal", adjustable="box")
            _line_axes(ax)
        cax = fig.add_axes((0.9, 0.2, 0.017, 0.64))
        fig.colorbar(image, cax=cax, label=_label("Time", tu))
        finish(fig, "particle_motion")

    dt = t[1] - t[0]
    spectrum = np.sum(np.abs(np.fft.rfft(signal[:, selected], axis=0)) ** 2, axis=1)
    frequencies = np.fft.rfftfreq(len(t), dt)
    peak_frequency = frequencies[np.argmax(spectrum[1:]) + 1] if spectrum[1:].any() else 0
    nperseg = min(
        len(t), max(8, min(len(t) // 3, round(6 / (peak_frequency * dt)) if peak_frequency else 64))
    )
    spectra = [
        spectrogram(
            signal[:, i],
            fs=1 / dt,
            nperseg=nperseg,
            noverlap=7 * nperseg // 8,
            nfft=4 * nperseg,
            detrend="constant",
            window="hann",
        )
        for i in selected
    ]
    power_ref = max(float(p.max()) for _, _, p in spectra) or 1
    if band:
        bounds = band
    elif spectrum.sum() > 0:
        upper = frequencies[
            min(len(frequencies) - 1, np.searchsorted(np.cumsum(spectrum), 0.995 * spectrum.sum()))
        ]
        bounds = [0, min(0.5 / dt, max(1.25 * upper, 3 / (nperseg * dt)))]
    else:
        bounds = [0, 0.5 / dt]
    fu = frequency_unit(bounds[1])
    fig = _figure(74 * nrows)
    gs = fig.add_gridspec(
        nrows,
        2,
        left=0.10,
        right=0.88,
        bottom=0.10,
        top=0.955,
        hspace=0.36,
        wspace=0.34,
    )
    axes = [fig.add_subplot(gs[j // 2, j % 2]) for j in range(len(selected))]
    for j, (ax, i, (f, st, power)) in enumerate(zip(axes, selected, spectra, strict=True)):
        db = 10 * np.log10(np.maximum(power / power_ref, 1e-12))
        image = ax.pcolormesh(
            tu(st + t[0]),
            fu(f),
            db,
            cmap="inferno",
            vmin=-60,
            vmax=0,
            shading="auto",
            rasterized=True,
        )
        ax.set(
            xlabel=_label("Time", tu),
            ylabel=_label("Frequency", fu),
            title=f"({chr(97 + j)}) $x={xd[i]:.3g}$ {xu.tex}",
            ylim=fu(np.array(bounds)),
        )
    cax = fig.add_axes((0.9, 0.17, 0.017, 0.69))
    fig.colorbar(image, cax=cax, label="Relative power [dB]")
    finish(fig, "time_frequency")
    report["figure_notes"] = dict(
        arrival_velocity="Packet tracking, arrival regression and residuals. The observable colour scale saturates at its robust display limits; the colour bar has flat ends. Accepted/excluded receivers are screened for complete, unambiguous packets. The residual band is one stored time sample. Peak tracking describes the selected packet and does not identify a guided mode.",
        pulse_evolution="Envelopes are normalised independently; peak decay uses the maximum across the complete receiver array as a shared reference. Apparent decay includes spreading, interference and numerical loss, and is not an intrinsic material attenuation measurement. Pulse width is the envelope-weighted RMS duration.",
        particle_motion="Equal physical axes and shared limits preserve polarisation shape and relative amplitude. Segments below 5% of the selected-component envelope peak are hidden. Colour indicates time within the selected gate.",
        time_frequency="Hann short-time Fourier spectra share one power reference, 87.5% overlap and a -60 dB display floor. Fourfold FFT padding interpolates the spectrum; it does not improve physical frequency resolution. Gate edges are not extrapolated.",
    )
    if observable == "pressure":
        del report["figure_notes"]["particle_motion"]
    report["figure_parameters"] = dict(
        export_width_mm=190,
        font_size_pt=10,
        selected_receivers_m=x[selected].tolist(),
        stft_window_s=float(nperseg * dt),
        stft_frequency_spacing_hz=float(1 / (nperseg * dt)),
        stft_samples=nperseg,
        stft_overlap_samples=7 * nperseg // 8,
    )
    return paths
