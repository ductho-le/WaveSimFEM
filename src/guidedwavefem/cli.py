"""Command-line interface and execution orchestration for GuidedWaveFEM."""

from __future__ import annotations

import argparse
import os
import sys

from mpi4py import MPI

from guidedwavefem.analysis import plot_full_analysis
from guidedwavefem.config import generate_template_config, parse_and_resolve_config
from guidedwavefem.materials import log, preflight_estimate
from guidedwavefem.solvers import mpi_fail_fast, run_simulation

comm = MPI.COMM_WORLD
rank = comm.rank


@mpi_fail_fast
def main() -> None:
    """Parse command-line arguments and orchestrate simulation and post-processing."""
    parser = argparse.ArgumentParser(
        description="Unified Guided Wave FEM Simulation & Dispersion Extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to YAML, TOML, or JSON configuration file (REQUIRED to run simulation)",
    )
    parser.add_argument(
        "--generate-config",
        type=str,
        default=None,
        metavar="PATH",
        help="Generate a fully commented template YAML configuration file and exit",
    )
    parser.add_argument(
        "--solver",
        choices=["implicit", "explicit"],
        default=None,
        help="Time integration solver scheme override ('implicit' or 'explicit')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device for explicit solver ('auto', 'cuda', 'cuda:0', 'mps', 'cpu')",
    )
    parser.add_argument(
        "--precision",
        choices=["double", "single"],
        default=None,
        help="Floating point precision for explicit solver ('double' or 'single')",
    )
    parser.add_argument(
        "--structure-mode",
        choices=[
            "1_layer",
            "N_layers",
            "halfspace",
            "plate_on_halfspace",
            "plate_in_halfspaces",
        ],
        default=None,
        help="Structural configuration mode override",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory override for results and figures",
    )
    parser.add_argument(
        "--postprocess",
        type=str,
        default=None,
        help="Path to existing sensor_data.npz file to run signal processing without simulation",
    )
    parser.add_argument(
        "--f-min",
        type=float,
        default=None,
        help="Minimum frequency override for dispersion extraction (MHz)",
    )
    parser.add_argument(
        "--f-max",
        type=float,
        default=None,
        help="Maximum frequency override for dispersion extraction (MHz)",
    )
    parser.add_argument(
        "--k-min",
        type=float,
        default=None,
        help="Minimum wavenumber override for dispersion extraction (rad/mm)",
    )
    parser.add_argument(
        "--k-max",
        type=float,
        default=None,
        help="Maximum wavenumber override for dispersion extraction (rad/mm)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and display preflight mesh, DOF, step, and memory estimates, then exit without executing simulation",
    )
    parser.add_argument(
        "--max-frequency",
        type=float,
        default=None,
        help="Mesh sizing frequency override [Hz] (defaults to 3*f_center in config)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Total simulation duration override [s]",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Maximum time-step limit override [s]",
    )
    parser.add_argument(
        "--cfl",
        type=float,
        default=None,
        help="Courant-Friedrichs-Lewy (CFL) stability factor override",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Disable automated post-processing and figure generation",
    )
    args = parser.parse_args()

    # Fast template generation path
    if args.generate_config:
        if rank == 0:
            generate_template_config(args.generate_config)
            log(f"Template configuration file created: {args.generate_config}")
        comm.Barrier()
        return

    # Check for mandatory config when running simulation
    if not args.postprocess and not args.config:
        if rank == 0:
            parser.print_help()
            log(
                "\n"
                + "=" * 70
                + "\nERROR: Configuration file is required to run simulation.\n"
                "Please provide a YAML/TOML/JSON configuration file using:\n"
                "    guided-waves -c configs/01_single_layer_aluminum.yaml\n\n"
                "Or generate a template configuration with:\n"
                "    guided-waves --generate-config my_config.yaml\n"
                + "=" * 70
                + "\n"
            )
        sys.exit(1)

    # Resolve active configuration across Defaults -> File -> CLI overrides
    if args.config:
        cfg = parse_and_resolve_config(
            config_path=args.config, args=args, validate_simulation=not args.postprocess
        )
    else:
        cfg = {"run_postprocess": True}

    # Preflight dry-run path
    if getattr(args, "dry_run", False):
        preflight_estimate(cfg, rank=rank)
        if rank == 0:
            log("\n  Dry run complete: preflight estimation finished successfully.")
            log("  Exiting without mesh generation or FEM assembly.\n")
        comm.Barrier()
        return

    # Step 1: Simulation (if not postprocess-only)
    if args.postprocess:
        sensor_data_file = args.postprocess
        out_dir = args.output or os.path.dirname(sensor_data_file) or "."
    else:
        sensor_data_file, out_dir = run_simulation(args, cfg=cfg)

    # Step 2: Signal Processing (Rank 0 only)
    if rank == 0:
        if cfg.get("run_postprocess", True) and not args.no_postprocess:
            plot_full_analysis(
                sensor_data_file,
                output_dir=out_dir,
                f_min_disp_MHz=args.f_min
                if args.f_min is not None
                else cfg.get("f_min_disp_MHz"),
                f_max_disp_MHz=args.f_max
                if args.f_max is not None
                else cfg.get("f_max_disp_MHz"),
                k_min_disp=args.k_min
                if args.k_min is not None
                else cfg.get("k_min_disp"),
                k_max_disp=args.k_max
                if args.k_max is not None
                else cfg.get("k_max_disp"),
            )

        log(f"\n{'═' * 58}")
        log("  All tasks completed successfully!")
        log(f"  Output directory: {os.path.abspath(out_dir)}")
        log(f"{'═' * 58}\n")

    comm.Barrier()


if __name__ == "__main__":
    main()
