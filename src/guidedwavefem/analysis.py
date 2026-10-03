"""Signal processing, 2D FFT dispersion analysis, and publication figures."""

from __future__ import annotations

import os
import shutil

import numpy as np
from scipy.signal import butter, sosfiltfilt

from guidedwavefem.materials import compute_rayleigh_speed, log
from guidedwavefem.numerics import dispersion_spectrum

import matplotlib

matplotlib.use("Agg")  # safe for headless & MPI execution
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MaxNLocator  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402


def plot_full_analysis(
    sensor_data_filepath: str,
    output_dir: str | None = None,
    f_min_disp_MHz: float | None = None,
    f_max_disp_MHz: float | None = None,
    k_min_disp: float | None = None,
    k_max_disp: float | None = None,
) -> str:
    """Generate consolidated publication 2x2 master plot and export vector images.

    Layout:
      - Row 1: Selected sensor waveforms (u_2, nm) + B-scan (u_2, nm) [same size]
      - Row 2: FEM 2D FFT for u_1 (horizontal) and u_2 (vertical/out-of-plane)
               [strictly separated, no folding, single colorbar on far right]

    Parameters
    ----------
    sensor_data_filepath : str
        Path to input ``sensor_data.npz`` archive containing simulation time-history.
    output_dir : str, optional
        Destination directory for exported figures. If None, uses directory of data file.
    f_min_disp_MHz : float, optional
        Minimum frequency display bound [MHz] for 2D FFT plots.
    f_max_disp_MHz : float, optional
        Maximum frequency display bound [MHz] for 2D FFT plots.
    k_min_disp : float, optional
        Minimum wavenumber display bound [rad/mm] for 2D FFT plots.
    k_max_disp : float, optional
        Maximum wavenumber display bound [rad/mm] for 2D FFT plots.

    Returns
    -------
    str
        Path to the primary generated PNG visualization.
    """
    if not os.path.exists(sensor_data_filepath):
        raise FileNotFoundError(f"Data file {sensor_data_filepath} not found.")

    if output_dir is None:
        output_dir = os.path.dirname(sensor_data_filepath) or "."
    os.makedirs(output_dir, exist_ok=True)

    with np.load(sensor_data_filepath, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}

    if f_min_disp_MHz is None:
        f_min_disp_MHz = (
            float(data["f_min_disp_MHz"]) if "f_min_disp_MHz" in data else 0.0
        )
    if f_max_disp_MHz is None:
        f_max_disp_MHz = (
            float(data["f_max_disp_MHz"]) if "f_max_disp_MHz" in data else 2.0
        )
    if k_min_disp is None:
        k_min_disp = float(data["k_min_disp"]) if "k_min_disp" in data else 0.0
    if k_max_disp is None:
        k_max_disp = float(data["k_max_disp"]) if "k_max_disp" in data else 4.0

    log(f"\n{'=' * 58}")
    log(">> Generating Publication 2x2 Master Analysis Figure...")
    log(f"{'=' * 58}")

    # LaTeX configuration: use pdflatex if available, fallback to stix
    if shutil.which("pdflatex") or os.path.exists("/Library/TeX/texbin/pdflatex"):
        os.environ["PATH"] = os.environ.get("PATH", "") + ":/Library/TeX/texbin"
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.size": 10,
                "axes.labelsize": 10,
                "axes.titlesize": 10,
                "axes.titleweight": "bold",
                "xtick.labelsize": 8.5,
                "ytick.labelsize": 8.5,
                "legend.fontsize": 8,
            }
        )
    else:
        plt.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "serif",
                "mathtext.fontset": "stix",
                "font.size": 10,
                "axes.labelsize": 10,
                "axes.titlesize": 10,
                "axes.titleweight": "bold",
                "xtick.labelsize": 8.5,
                "ytick.labelsize": 8.5,
                "legend.fontsize": 8,
            }
        )

    times = data["times"]
    sensor_x = data["sensor_x"]
    sensor_data_y = data["sensor_data_y"]  # Out-of-plane displacement (u_2)
    sensor_data_x = data.get("sensor_data_x", None)  # Horizontal displacement (u_1)
    f_center = float(data["f_center"])
    c_L = float(data["c_L"])
    c_T = float(data["c_T"])
    st_mode = str(data["structure_mode"]) if "structure_mode" in data else ""
    c_R = float(data["c_R"]) if "c_R" in data and not np.isnan(data["c_R"]) else None
    is_fluid_top = (
        bool(data["is_fluid_top"]) if "is_fluid_top" in data else (c_T < 50.0)
    )
    has_fluid = bool(data["has_fluid"]) if "has_fluid" in data else is_fluid_top
    c_fluid = (
        float(data["c_fluid"]) if "c_fluid" in data else (c_L if is_fluid_top else None)
    )

    if c_R is None and st_mode == "halfspace" and "E" in data and "nu" in data:
        c_R = compute_rayleigh_speed(c_T, c_L=c_L)

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
    for lo, hi in ((f_min_disp_MHz, f_max_disp_MHz), (k_min_disp, k_max_disp)):
        if not np.isfinite([lo, hi]).all() or not 0 <= lo < hi:
            raise ValueError("Dispersion limits must be finite with 0 <= min < max")
    dt = times[1] - times[0]
    fs = 1.0 / dt
    n_sensors = len(sensor_x)

    # Bandpass filter on displacements
    f_lo = max(f_center / 5.0, 100.0)
    f_hi = min(f_center * 3.0, fs / 2.0 * 0.9)
    if f_lo >= f_hi:
        raise ValueError("Sampling frequency is too low for the requested bandpass")
    sos = butter(4, [f_lo, f_hi], btype="bandpass", fs=fs, output="sos")
    dy = np.column_stack(
        [sosfiltfilt(sos, sensor_data_y[:, j]) for j in range(n_sensors)]
    )
    if sensor_data_x is not None:
        dx_filt = np.column_stack(
            [sosfiltfilt(sos, sensor_data_x[:, j]) for j in range(n_sensors)]
        )
    else:
        dx_filt = None

    # Displacement in nm
    dy_nm = dy * 1e9

    # 1. 2D FFT forward-propagating dispersion (+k only, no folding, u1 & u2 strictly separated)
    log("  Computing 2D FFT forward-propagating dispersion (u1 and u2 separated)...")
    freqs_fk_MHz, k_fk_radmm, power_u2 = dispersion_spectrum(dy, times, sensor_x)
    f_disp_mask = (freqs_fk_MHz >= f_min_disp_MHz) & (freqs_fk_MHz <= f_max_disp_MHz)
    k_disp_mask = (k_fk_radmm >= k_min_disp) & (k_fk_radmm <= k_max_disp)
    if not f_disp_mask.any() or not k_disp_mask.any():
        raise ValueError("Requested dispersion range contains no sampled FFT bins")
    # Coordinate extent comes from bin edges, NEVER requested plot limits.
    selected_f, selected_k = freqs_fk_MHz[f_disp_mask], k_fk_radmm[k_disp_mask]
    df, dk = freqs_fk_MHz[1] - freqs_fk_MHz[0], k_fk_radmm[1] - k_fk_radmm[0]
    extent_disp = [
        selected_k[0] - dk / 2,
        selected_k[-1] + dk / 2,
        selected_f[0] - df / 2,
        selected_f[-1] + df / 2,
    ]

    # 2D FFT for u_1 (horizontal displacement alone, no folding, no combination)
    if dx_filt is not None:
        _, _, power_u1 = dispersion_spectrum(dx_filt, times, sensor_x)
        disp_power_u1 = power_u1[np.ix_(f_disp_mask, k_disp_mask)]
        fk_db_u1 = 10 * np.log10(
            disp_power_u1 / (np.max(disp_power_u1) + 1e-30) + 1e-30
        )
    else:
        fk_db_u1 = None

    # 2D FFT for u_2 (vertical / out-of-plane displacement alone, no folding, no combination)
    disp_power_u2 = power_u2[np.ix_(f_disp_mask, k_disp_mask)]
    fk_db_u2 = 10 * np.log10(disp_power_u2 / (np.max(disp_power_u2) + 1e-30) + 1e-30)

    # 2. Assemble publication figure (2x2 layout)
    fig_w = 190.0 / 25.4  # 7.48 in
    fig_h = 165.0 / 25.4  # 6.50 in
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.32, wspace=0.15)

    # Probe sensor spatial array using setup parameters (sensor_offset, sensor_pitch)
    s_offset = (
        float(data["sensor_offset"])
        if "sensor_offset" in data
        else (
            sensor_x[0] - float(data.get("source_x", 0.0))
            if len(sensor_x) > 0
            else 0.05
        )
    )
    s_pitch = (
        float(data["sensor_pitch"])
        if "sensor_pitch" in data
        else (sensor_x[1] - sensor_x[0] if len(sensor_x) > 1 else 0.5e-3)
    )
    n_sensors_data = dy.shape[1]
    probe_x = s_offset + np.arange(n_sensors_data) * s_pitch

    # Panel (a): Out-of-plane waveforms (u_2) — 3 signals: first, middle, last
    ax1 = fig.add_subplot(gs[0, 0])
    indices = [0, (n_sensors_data - 1) // 2, n_sensors_data - 1]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 3))
    for i, idx in enumerate(indices):
        ax1.plot(
            times * 1e6,
            dy_nm[:, idx],
            lw=0.85,
            color=colors[i],
            label=rf"$x = {probe_x[idx] * 1e3:.0f}$\,mm",
        )
    ax1.set_xlabel(r"Time [$\mu$s]")
    ax1.set_ylabel(r"Displacement $u_2$ [nm]")
    ax1.set_title(r"\textbf{(a) Selected $u_2$ signals}")
    ax1.legend(loc="upper right", framealpha=0.85)
    ax1.grid(True, alpha=0.3, linestyle=":")

    divider1 = make_axes_locatable(ax1)
    cax1_dummy = divider1.append_axes("right", size="3.5%", pad=0.06)
    cax1_dummy.axis("off")

    # Panel (b): B-scan (u_2)
    ax2 = fig.add_subplot(gs[0, 1])
    probe_x_mm = probe_x * 1e3
    dx_probe_mm = s_pitch * 1e3
    dt_us = (times[1] - times[0]) * 1e6
    extent_bs = [
        times[0] * 1e6 - dt_us / 2.0,
        times[-1] * 1e6 + dt_us / 2.0,
        probe_x_mm[0] - dx_probe_mm / 2.0,
        probe_x_mm[-1] + dx_probe_mm / 2.0,
    ]
    vmax = np.max(np.abs(dy_nm)) * 0.3
    im2 = ax2.imshow(
        dy_nm.T,
        aspect="auto",
        extent=extent_bs,
        origin="lower",
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
    )

    speed_lines = []
    if is_fluid_top:
        speed_lines.append((c_L, "--", "cyan", rf"$c_f = {c_L:.0f}$\,m/s (fluid)"))
    else:
        speed_lines.append((c_L, "--", "cyan", rf"$c_L = {c_L:.0f}$\,m/s"))
        if c_T >= 50.0:
            speed_lines.append((c_T, ":", "lime", rf"$c_T = {c_T:.0f}$\,m/s"))
    if c_R is not None and not is_fluid_top and c_T >= 50.0:
        speed_lines.append((c_R, "-.", "gold", rf"$c_R = {c_R:.0f}$\,m/s (Rayleigh)"))
    elif has_fluid and not is_fluid_top and c_fluid is not None:
        speed_lines.append(
            (c_fluid, "-.", "magenta", rf"$c_f = {c_fluid:.0f}$\,m/s (fluid)")
        )

    t0_pulse_us = (1.5 / f_center) * 1e6
    for spd, sty, clr, name in speed_lines:
        t_arr = t0_pulse_us + (probe_x / spd) * 1e6
        valid = (t_arr >= times[0] * 1e6) & (t_arr <= times[-1] * 1e6)
        if np.any(valid):
            ax2.plot(
                t_arr[valid],
                probe_x_mm[valid],
                color=clr,
                linestyle=sty,
                lw=1.1,
                label=rf"{name} ($t_0 + x/c$)",
            )
    ax2.set_xlim(times[0] * 1e6, times[-1] * 1e6)
    ax2.set_ylim(probe_x_mm[0], probe_x_mm[-1])
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax2.set_xlabel(r"Time [$\mu$s]")
    ax2.set_ylabel(r"Offset [mm]")
    ax2.set_title(r"\textbf{(b) B-scan ($u_2$)}")
    ax2.legend(loc="lower right", framealpha=0.85)

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="3.5%", pad=0.06)
    cb2 = fig.colorbar(im2, cax=cax2)
    cb2.set_label(r"$u_2$ [nm]")

    # Tick positions for 2nd row vertical axis (1 digit after decimal point)
    disp_yticks = np.arange(f_min_disp_MHz, f_max_disp_MHz + 1e-6, 0.5)

    disp_lines = []
    if is_fluid_top:
        disp_lines.append((c_L, "--", "cyan", rf"$c_f = {c_L:.0f}$\,m/s"))
    elif c_R is not None and c_T >= 50.0:
        disp_lines.append((c_R, "--", "cyan", rf"$c_R = {c_R:.0f}$\,m/s"))
    if has_fluid and not is_fluid_top and c_fluid is not None:
        disp_lines.append((c_fluid, ":", "magenta", rf"$c_f = {c_fluid:.0f}$\,m/s"))

    def _plot_disp_lines(ax_target):
        for d_spd, d_sty, d_clr, d_lbl in disp_lines:
            k_line = np.linspace(k_min_disp, k_max_disp, 200)
            f_line = (d_spd / (2000.0 * np.pi)) * k_line
            valid_line = (f_line >= f_min_disp_MHz) & (f_line <= f_max_disp_MHz)
            if np.any(valid_line):
                ax_target.plot(
                    k_line[valid_line],
                    f_line[valid_line],
                    d_sty,
                    color=d_clr,
                    lw=1.1,
                    alpha=0.9,
                    label=d_lbl,
                )
                ax_target.legend(loc="upper left", framealpha=0.8)

    # Panel (c): 2D FFT for u_1 (horizontal displacement alone)
    ax3 = fig.add_subplot(gs[1, 0])
    if fk_db_u1 is not None:
        ax3.imshow(
            fk_db_u1,
            aspect="auto",
            extent=extent_disp,
            origin="lower",
            cmap="inferno",
            vmin=-40,
            vmax=0,
        )
    _plot_disp_lines(ax3)
    ax3.set_xlim(k_min_disp, k_max_disp)
    ax3.set_ylim(f_min_disp_MHz, f_max_disp_MHz)
    ax3.set_yticks(disp_yticks)
    ax3.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax3.set_xlabel(r"Wavenumber [rad/mm]")
    ax3.set_ylabel(r"Frequency [MHz]")
    ax3.set_title(r"\textbf{(c) Dispersion spectra ($u_1$ sensor)}")

    # Dummy divider on ax3 matches width of ax1 and ax2/ax4
    divider3 = make_axes_locatable(ax3)
    cax3_dummy = divider3.append_axes("right", size="3.5%", pad=0.06)
    cax3_dummy.axis("off")

    # Panel (d): 2D FFT for u_2 (vertical displacement alone)
    ax4 = fig.add_subplot(gs[1, 1], sharey=ax3)
    im4 = ax4.imshow(
        fk_db_u2,
        aspect="auto",
        extent=extent_disp,
        origin="lower",
        cmap="inferno",
        vmin=-40,
        vmax=0,
    )
    _plot_disp_lines(ax4)
    ax4.set_xlim(k_min_disp, k_max_disp)
    ax4.set_ylim(f_min_disp_MHz, f_max_disp_MHz)
    ax4.set_yticks(disp_yticks)
    ax4.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax4.set_xlabel(r"Wavenumber [rad/mm]")
    ax4.set_ylabel("")
    ax4.set_title(r"\textbf{(d) Dispersion spectra ($u_2$ sensor)}")

    # Single colorbar for the second row placed on ax4 (right edge)
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes("right", size="3.5%", pad=0.06)
    cb4 = fig.colorbar(im4, cax=cax4)
    cb4.set_label(r"Relative power [dB, per component]")

    # Export vector images (.pdf, .svg, .eps) and publication-quality .png
    out_base = os.path.join(output_dir, "full_analysis")
    for ext in [".png", ".pdf", ".svg", ".eps"]:
        fig.savefig(f"{out_base}{ext}", dpi=300, bbox_inches="tight")
        log(f"  Saved: {out_base}{ext}")

    plt.close(fig)
    return f"{out_base}.png"
