#!/usr/bin/env python3
"""Compare example resolutions on a fixed early receiver aperture.

Preserves domain, materials, source, and receiver pitch. Reports unshifted
waveform errors on a common time grid; no amplitude normalization or alignment
is applied. These are early-arrival sensitivity checks, not full-run convergence.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import yaml
from mpi4py import MPI

from guidedwavefem import run_simulation, set_verbosity
from guidedwavefem.config import parse_and_resolve_config
from guidedwavefem.storage import atomic_output


def resolution_variants(cfg, refinement_factor, step_factor):
    """Refine explicit counts and the configured time control independently."""
    if not math.isfinite(refinement_factor) or refinement_factor <= 1:
        raise ValueError("refinement_factor must be finite and greater than one")
    if not math.isfinite(step_factor) or not 0 < step_factor < 1:
        raise ValueError("step_factor must be finite and in (0, 1)")
    if cfg["dt"] is not None:
        control = "dt"
    else:
        control = "steps_per_period" if cfg["solver_type"] == "implicit" else "CFL"
    tighter = (
        cfg[control] / step_factor if control == "steps_per_period" else cfg[control] * step_factor
    )
    refined = dict(
        cfg,
        nx=math.ceil(cfg["nx"] * refinement_factor),
        ny_per_region=[math.ceil(n * refinement_factor) for n in cfg["ny_per_region"]],
    )
    refined[control] = tighter
    temporal = dict(cfg)
    temporal[control] = tighter
    return control, [refined, temporal, dict(cfg)]


def check_resumed_configuration(saved, requested):
    """Do not silently compare cached runs made with different model inputs."""
    ignored = {"runtime", "load_y"}
    if requested.get("sensor_y") is None:
        ignored.add("sensor_y")  # the solver saves the resolved automatic height
    differences = [
        key
        for key, value in requested.items()
        if key not in ignored and not key.startswith("_") and saved.get(key) != value
    ]
    if differences:
        raise ValueError("Cached run configuration differs: " + ", ".join(sorted(differences)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--duration", required=True, type=float)
    parser.add_argument("--frequency", type=float)
    parser.add_argument(
        "--refinement-factor",
        default=2.0,
        type=float,
        help="Multiply nx and every region count by this factor (>1)",
    )
    parser.add_argument(
        "--step-factor",
        default=0.5,
        type=float,
        help="Fraction of the configured time step or CFL for tighter runs (0,1)",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse completed variant archives")
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        args.output.mkdir(parents=True, exist_ok=args.resume)
    comm.Barrier()
    cfg = parse_and_resolve_config(config_path=args.config)
    # Resolve coordinates before varying resolution; only the receiver count is
    # reduced to the first nine original positions to limit archive size.
    cfg.update(
        duration=args.duration,
        n_sensors=min(9, cfg["n_sensors"]),
        run_postprocess=False,
        write_vtx=False,
    )
    if args.frequency is not None:
        cfg["max_frequency"] = args.frequency
    cfg = parse_and_resolve_config(custom_cfg=cfg)
    control, variants = resolution_variants(cfg, args.refinement_factor, args.step_factor)
    set_verbosity("WARNING")
    reports, histories = [], []
    for index, variant in enumerate(variants):
        current = dict(variant, output_dir=str(args.output / str(index)))
        archive = Path(current["output_dir"]) / "sensor_data.npz"
        reuse = comm.bcast(args.resume and archive.exists() if comm.rank == 0 else None, root=0)
        wall = None
        if not reuse:
            start = time.perf_counter()
            archive, _ = run_simulation(cfg=current)
            wall = comm.allreduce(time.perf_counter() - start, op=MPI.MAX)
        if comm.rank != 0:
            continue
        with np.load(archive) as data:
            times = data["times"]
            values = np.stack([data["sensor_data_x"], data["sensor_data_y"]], axis=-1)
            saved = yaml.safe_load(str(data["simulation_config_yaml"]))
            check_resumed_configuration(saved, current)
            runtime = saved["runtime"]
            dt = float(data["dt"])
        histories.append((times, values))
        report = dict(
            nx=current["nx"],
            ny_per_region=current["ny_per_region"],
            step_control={control: current[control]},
            wall_seconds=wall,
            ndofs=runtime["ndofs"],
            dt=dt,
            mpi_ranks=runtime["mpi_ranks"],
        )
        reports.append(report)
        print(json.dumps(report), flush=True)
    if comm.rank != 0:
        return
    # Interpolate each reference onto the candidate's sample grid. Sampling is
    # far denser than the source band; compare both components without shifting.
    for index, reference in ((1, 0), (2, 1), (2, 0)):
        times, values = histories[index]
        ref_times, ref_values = histories[reference]
        interpolated = np.empty_like(values)
        for sensor in range(values.shape[1]):
            for component in range(2):
                interpolated[:, sensor, component] = np.interp(
                    times, ref_times, ref_values[:, sensor, component]
                )
        error = values - interpolated
        metrics = dict(
            relative_l2=float(np.linalg.norm(error) / np.linalg.norm(interpolated)),
            relative_peak_error=float(np.max(np.abs(error)) / np.max(np.abs(interpolated))),
            relative_peak_amplitude_change=float(
                abs(np.max(np.abs(values)) / np.max(np.abs(interpolated)) - 1)
            ),
        )
        reports[index][f"versus_{reference}"] = metrics
    report = dict(
        config=args.config,
        duration=args.duration,
        receivers=cfg["n_sensors"],
        frequency=cfg["max_frequency"],
        runs=reports,
    )
    with atomic_output(args.output / "comparison.json") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    from guidedwavefem.solvers import mpi_fail_fast

    mpi_fail_fast(main)()
