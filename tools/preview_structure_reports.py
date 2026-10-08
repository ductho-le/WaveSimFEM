#!/usr/bin/env python3
"""Render structure-specific reports from explicitly synthetic array records.

These examples validate report behavior, not the finite-element solver or a
multilayer dispersion model. Only the free-plate wavenumbers and homogeneous
halfspace Rayleigh speed use analytical physical reference solutions. Other
branches are prescribed, illustrative dispersive packets. Every output carries
this provenance; ``manifest.json`` records the generating assumptions.

Usage: python tools/preview_structure_reports.py --output results_structure_preview
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from guidedwavefem.config import parse_and_resolve_config
from guidedwavefem.lamb import s0_wavenumber
from guidedwavefem.materials import a0_wavenumber, compute_rayleigh_speed, structure_from_config


@dataclass(frozen=True)
class Branch:
    name: str
    wavenumber: Callable[[np.ndarray], np.ndarray]
    amplitude: float
    horizontal_ratio: complex
    attenuation: float
    assumption: str


def _power_branch(name, f0, speed, exponent, amplitude, ratio, attenuation):
    """A prescribed c_p(f)=speed*(f/f0)**exponent, with positive group speed."""
    return Branch(
        name,
        lambda f: 2 * np.pi * f / (speed * (f / f0) ** exponent),
        amplitude,
        ratio,
        attenuation,
        f"Illustrative c_p={speed:g}*(f/{f0:g})**{exponent:g} m/s; not modal theory",
    )


def _branches(case, cfg, structure):
    f0 = cfg["f_center"]
    if case == "01":
        layer = structure["plate_layers"][0]
        arguments = (layer["thickness"], layer["c_L"], layer["c_T"])
        return [
            Branch(
                "A0",
                lambda f: np.array([a0_wavenumber(fi, *arguments) for fi in f]),
                1.0,
                -0.32j,
                0.0,
                "Exact real A0 Rayleigh-Lamb wavenumber; prescribed surface polarization",
            ),
            Branch(
                "S0",
                lambda f: np.array([s0_wavenumber(fi, *arguments) for fi in f]),
                0.45,
                2.1j,
                0.0,
                "Exact real S0 Rayleigh-Lamb wavenumber; prescribed surface polarization",
            ),
        ]
    if case == "02":
        return [
            _power_branch("Slow sandwich packet", f0, 2100, 0.18, 1.0, -0.55j, 14.0),
            _power_branch("Fast sandwich packet", f0, 4800, -0.08, 0.45, 1.8j, 4.0),
        ]
    if case == "03":
        region = structure["bottom_halfspace"]
        cr = compute_rayleigh_speed(region["c_T"], c_L=region["c_L"])
        # Exact free-surface Rayleigh H/V for a homogeneous isotropic halfspace.
        # Overall rotation sign depends on coordinate/propagation conventions.
        q = np.sqrt(1 - (cr / region["c_T"]) ** 2)
        hv = 2 * q / (1 + q * q)
        return [
            Branch(
                "Rayleigh packet",
                lambda f: 2 * np.pi * f / cr,
                1.0,
                -1j * hv,
                0.0,
                f"Homogeneous Rayleigh speed {cr:.8g} m/s and free-surface H/V {hv:.8g}",
            ),
            _power_branch("Weak bulk-speed packet", f0, region["c_L"], 0.0, 0.12, 2.0, 0.0),
        ]
    if case == "04":
        soil = structure["plate_layers"][-1]
        rock = structure["bottom_halfspace"]
        slow = compute_rayleigh_speed(soil["c_T"], c_L=soil["c_L"])
        fast = compute_rayleigh_speed(rock["c_T"], c_L=rock["c_L"])
        return [
            Branch(
                "Illustrative dispersive surface packet",
                lambda f: 2 * np.pi * f / (slow + (fast - slow) / (1 + (f / 8.0) ** 2)),
                1.0,
                -0.72j,
                0.0,
                "Prescribed c_p=c_R,soil+(c_R,rock-c_R,soil)/(1+(f/8 Hz)^2); no inversion",
            ),
            _power_branch("Faster surface packet", f0, 760, -0.16, 0.30, 0.38j, 0.0),
        ]
    if case == "05":
        return [
            _power_branch("Slow guided packet", f0, 1800, 0.15, 1.0, -0.60j, 18.0),
            _power_branch("Fast guided packet", f0, 3500, -0.06, 0.50, 1.6j, 6.0),
            _power_branch("Tissue-speed packet", f0, 1500, 0.0, 0.20, 1.0, 3.0),
        ]
    return [
        _power_branch("Subsonic interface-like packet", f0, 1360, 0.035, 1.0, -0.65j, 2.0),
        _power_branch("Supersonic plate-like packet", f0, 3100, 0.12, 0.65, 0.60j, 22.0),
    ]


def create_synthetic_archive(config_path: Path, output_dir: Path) -> tuple[Path, dict]:
    """Use a shipped structure with a well-sampled, prescribed traveling wavefield."""
    cfg = parse_and_resolve_config(config_path=str(config_path))
    structure = structure_from_config(cfg)
    case = config_path.name[:2]
    overrides = {}
    if case == "01":
        # Fundamentals are separated clearly below higher-mode cutoffs here.
        overrides = {
            "f_center": 250e3,
            "min_frequency": 80e3,
            "max_frequency": 500e3,
            "disp_f_min": 50e3,
            "disp_f_max": 500e3,
        }
        cfg.update(overrides)
    f0 = cfg["f_center"]
    branches = _branches(case, cfg, structure)
    band = (max(0.24 * f0, cfg["disp_f_min"]), min(1.8 * f0, cfg["disp_f_max"]))
    frequencies = np.linspace(*band, 160)
    speeds = [2 * np.pi * frequencies / branch.wavenumber(frequencies) for branch in branches]
    minimum_phase_speed = min(float(np.min(speed)) for speed in speeds)
    minimum_group_speed = min(
        float(np.min(2 * np.pi / np.gradient(branch.wavenumber(frequencies), frequencies)))
        for branch in branches
    )
    receivers = 129
    original_span = (cfg["n_sensors"] - 1) * cfg["sensor_pitch"]
    span = min(original_span, (receivers - 1) * minimum_phase_speed / (4.5 * band[1]))
    offsets = cfg["sensor_offset"] + np.linspace(0, span, receivers)
    positions = cfg["source_x"] + offsets
    delay = 6.0 / f0
    # Extra duration captures a deliberately weak counter-propagating return.
    duration = delay + (float(offsets[-1]) + span) / minimum_group_speed + 12.0 / f0
    samples = max(1024, int(np.ceil(duration * 6 * band[1])))
    times = np.arange(samples) * duration / samples
    frequency = np.fft.rfftfreq(samples, times[1])
    active = (frequency >= band[0]) & (frequency <= band[1])
    f = frequency[active]
    weights = np.exp(-0.5 * ((f - f0) / (0.27 * f0)) ** 2)
    # Compact smooth spectral support prevents hard band-edge ringing.
    edge = (f - band[0]) / (band[1] - band[0])
    weights *= np.sin(np.pi * edge) ** 2
    sy = np.zeros((len(frequency), receivers), dtype=complex)
    sx = np.zeros_like(sy)
    assumptions = []
    for branch in branches:
        k = branch.wavenumber(f)
        alpha = branch.attenuation * (f / f0) ** 1.25
        forward = np.exp(-1j * (2 * np.pi * f[:, None] * delay + k[:, None] * offsets))
        forward *= np.exp(-alpha[:, None] * offsets)
        return_path = 2 * offsets[-1] - offsets
        backward = np.exp(
            -1j * (2 * np.pi * f[:, None] * delay + k[:, None] * return_path)
        ) * np.exp(-alpha[:, None] * return_path)
        component = branch.amplitude * weights[:, None] * (forward + 0.06 * backward)
        sy[active] += component
        sx[active] += branch.horizontal_ratio * component
        assumptions.append(
            {
                "name": branch.name,
                "dispersion": branch.assumption,
                "vertical_relative_amplitude": branch.amplitude,
                "horizontal_over_vertical_complex": [
                    branch.horizontal_ratio.real,
                    branch.horizontal_ratio.imag,
                ],
                "prescribed_amplitude_decay_Np_per_m_at_f0": branch.attenuation,
                "decay_frequency_power": 1.25,
            }
        )
    uy = np.fft.irfft(sy, n=samples, axis=0)
    ux = np.fft.irfft(sx, n=samples, axis=0)
    amplitude = 1e-6 if case == "04" else 30e-9
    scale = amplitude / np.max(np.abs(uy))
    uy, ux = scale * uy, scale * ux
    rng = np.random.default_rng(100 + int(case))
    uy += rng.normal(0, amplitude * 2e-4, uy.shape)
    ux += rng.normal(0, amplitude * 2e-4, ux.shape)
    sensor_y = structure["H_plate"] if structure["has_top_halfspace"] else structure["y_max"]
    cfg.update(
        n_sensors=receivers,
        sensor_pitch=float(positions[1] - positions[0]),
        sensor_y=sensor_y,
        duration=float(duration),
        sample_rate=float(1 / times[1]),
    )
    provenance = "SYNTHETIC prescribed wave packets; report demonstration, not an FEM result"
    regions = structure["all_regions"]
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "synthetic_sensor_data.npz"
    np.savez(
        archive_path,
        times=times,
        sensor_x=positions,
        sensor_y=sensor_y,
        load_y=sensor_y,
        sensor_data_x=ux,
        sensor_data_y=uy,
        source_x=cfg["source_x"],
        f_center=f0,
        c_L=structure["c_L_plate_top"],
        c_T=structure["c_T_plate_top"],
        plate_thickness=structure["H_plate"],
        structure_mode=cfg["structure_mode"],
        all_regions_names=np.array([r["name"] for r in regions]),
        all_regions_thicknesses=np.array([r["thickness"] for r in regions]),
        all_regions_models=np.array([r["material_model"] for r in regions]),
        all_regions_roles=np.array([r["role"] for r in regions]),
        disp_f_min=cfg["disp_f_min"],
        disp_f_max=cfg["disp_f_max"],
        disp_k_min=cfg["disp_k_min"],
        disp_k_max=cfg["disp_k_max"],
        simulation_config_yaml=yaml.safe_dump(cfg),
        provenance=provenance,
    )
    return archive_path, {
        "config": config_path.name,
        "archive": str(archive_path.resolve()),
        "provenance": provenance,
        "source_frequency_overrides": overrides,
        "receivers": receivers,
        "samples": samples,
        "duration_s": duration,
        "spectral_support_Hz": list(band),
        "prescribed_backward_amplitude_ratio": 0.06,
        "independent_noise_standard_deviation_m": amplitude * 2e-4,
        "seed": 100 + int(case),
        "branches": assumptions,
        "limitations": [
            "No FEM solve or source-response calculation is performed.",
            "Multilayer, bone, soil and immersed branches are illustrative, not solved modes.",
            "Surface polarization is prescribed except the homogeneous Rayleigh packet.",
            "Counter-propagating packet is prescribed and is not a boundary-reflection prediction.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results_structure_preview"))
    parser.add_argument(
        "--formats", nargs="+", choices=("png", "pdf", "svg"), default=["png", "pdf"]
    )
    parser.add_argument("--case", choices=[f"{i:02d}" for i in range(1, 7)], help="Render one case")
    args = parser.parse_args()
    from guidedwavefem.wave_reports import write_structure_report

    configs = Path(__file__).resolve().parents[1] / "configs"
    records = []
    for config in sorted(configs.glob("0*.yaml")):
        if args.case and not config.name.startswith(args.case):
            continue
        destination = args.output / config.stem
        archive, description = create_synthetic_archive(config, destination)
        result = write_structure_report(archive, destination, formats=args.formats, usetex=True)
        description["figures"] = result["figures"]
        description["title"] = result["profile"]["title"]
        records.append(description)
    manifest = args.output / (f"manifest_{args.case}.json" if args.case else "manifest.json")
    manifest.write_text(json.dumps(records, indent=2, default=str) + "\n", encoding="utf-8")
    index = [
        "# Structure-specific wave reports",
        "",
        "Synthetic verification examples; these are not FEM simulation results. "
        "Layered and fluid-loaded branches use prescribed illustrative dispersion. "
        f"See [generation assumptions]({manifest.name}).",
        "",
        "| Structure | Dispersion and interpretation | Propagation and components |",
        "|---|---|---|",
    ]
    for record in records:
        folder = Path(record["config"]).stem + "/wave_physics/"
        links = [
            " · ".join(f"[{ext.upper()}]({folder}{name}.{ext})" for ext in args.formats)
            for name in ("dispersion_and_modes", "propagation_and_components")
        ]
        index.append(
            "| " + record["title"].replace(" · ", " / ") + " | " + " | ".join(links) + " |"
        )
    index += [
        "",
        "Each structure folder includes its synthetic archive. The `wave_physics` folder "
        "contains both figures, `report.json`, four CSV tables, and `spectral_arrays.npz`.",
    ]
    index_name = f"README_{args.case}.md" if args.case else "README.md"
    (args.output / index_name).write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"Synthetic structure-report demonstrations saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
