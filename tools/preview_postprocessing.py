#!/usr/bin/env python3
"""Render reproducible analytical examples without running the FEM solver.

Usage: python tools/preview_postprocessing.py --output results_postprocess_preview
The packet and Lamb examples are synthetic verification data, not FEM results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from guidedwavefem.analysis import plot_full_analysis, plot_quick_overview
from guidedwavefem.characterisation import write_characterisation
from guidedwavefem.lamb import s0_wavenumber, write_lamb_report
from guidedwavefem.materials import a0_wavenumber


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results_postprocess_preview"))
    args = parser.parse_args()
    destination = args.output
    destination.mkdir(parents=True, exist_ok=True)
    times = np.arange(1600) * 1e-7
    positions = np.linspace(0.012, 0.132, 96)
    phase = times[:, None] - 18e-6 - positions / 1800
    envelope = 50e-9 * np.exp(-((phase / 5e-6) ** 2)) * np.exp(-5 * positions)
    uy = envelope * np.sin(2 * np.pi * 150e3 * phase)
    ux = 0.4 * envelope * np.cos(2 * np.pi * 150e3 * phase)
    packet_path = destination / "synthetic_packet.npz"
    np.savez(
        packet_path,
        times=times,
        sensor_x=positions,
        sensor_data_x=ux,
        sensor_data_y=uy,
        source_x=0.0,
        f_center=150e3,
        c_L=6200.0,
        c_T=3120.0,
        source_type="tone_burst",
        n_cycles=5,
        provenance="Analytical Gaussian packet: speed 1800 m/s, amplitude decay 5 Np/m",
    )
    plot_full_analysis(str(packet_path), str(destination), formats=("png", "pdf"), usetex=True)
    plot_quick_overview(
        times,
        positions,
        uy,
        150e3,
        0,
        str(destination / "quick_overview.png"),
        source_type="tone_burst",
        n_cycles=5,
    )
    write_characterisation(packet_path, destination, formats=("png", "pdf"))

    # Superpose independently evaluated plate branches over a Gaussian frequency band.
    # The inverse transform makes dispersive packets instead of a monochromatic image.
    nt, dt = 1536, 2e-7
    t = np.arange(nt) * dt
    x = np.linspace(0.024, 0.264, 256)
    f = np.fft.rfftfreq(nt, dt)
    components = {mode: np.zeros((len(f), len(x)), dtype=complex) for mode in ("A0", "S0")}
    for i in np.flatnonzero((f > 1e5) & (f < 3e5)):
        weight = np.exp(-0.5 * ((f[i] - 2e5) / 3e4) ** 2)
        for mode, function in (("A0", a0_wavenumber), ("S0", s0_wavenumber)):
            k = function(f[i], 0.003, 6200, 3120)
            components[mode][i] = weight * np.exp(-1j * (2 * np.pi * f[i] * 30e-6 + k * x))
    a, s = [np.fft.irfft(components[mode], n=nt, axis=0) for mode in ("A0", "S0")]
    scale = 40e-9 / np.max(np.abs(a))
    cfg = dict(
        structure_mode="1_layer",
        single_layer=dict(thickness=0.003, rho=2700, E=70e9, nu=0.33),
        f_center=2e5,
        min_frequency=1e5,
        max_frequency=3e5,
    )
    lamb_path = destination / "synthetic_lamb.npz"
    np.savez(
        lamb_path,
        times=t,
        sensor_x=x,
        sensor_data_y=scale * (a + 0.45 * s),
        sensor_data_x=scale * (0.3 * a + s),
        c_L=6200.0,
        c_T=3120.0,
        source_x=0.0,
        simulation_config_yaml=yaml.safe_dump(cfg),
        provenance="Analytical Rayleigh-Lamb A0/S0 packet superposition; not a FEM solve",
    )
    write_lamb_report(lamb_path, destination, formats=("png", "pdf"))
    print(f"Synthetic verification previews saved to {destination.resolve()}")


if __name__ == "__main__":
    main()
