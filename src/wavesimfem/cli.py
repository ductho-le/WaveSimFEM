"""Command-line interface and execution orchestration for WaveSimFEM."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from typing import NoReturn

from mpi4py import MPI

from wavesimfem import __version__
from wavesimfem.config import generate_template_config, parse_and_resolve_config
from wavesimfem.device import DEVICE_CHOICES_HELP, DeviceUnavailableError
from wavesimfem.materials import STRUCTURE_MODES, preflight_estimate
from wavesimfem.reporting import log, set_verbosity, warn
from wavesimfem.solvers import mpi_fail_fast, run_simulation
from wavesimfem.units import parse_quantity

comm = MPI.COMM_WORLD
rank = comm.rank

_MISSING_CONFIG_MESSAGE = (
    "\n" + "=" * 70 + "\nERROR: Configuration file is required to run simulation.\n"
    "Please provide a YAML/TOML/JSON configuration file using:\n"
    "    wavesim -c configs/01_plate_aluminum_lamb.yaml\n\n"
    "Or generate a template configuration with:\n"
    "    wavesim --generate-config my_config.yaml\n" + "=" * 70 + "\n"
)

#: Exit status for invalid user input (unreadable or invalid configuration, missing files).
EXIT_USAGE = 2
#: Exit status when an explicitly requested compute device is unavailable.
EXIT_DEVICE_UNAVAILABLE = 3


def _exit_with_error(
    message: str, code: int, *, exc: BaseException | None = None, verbose: bool = False
) -> NoReturn:
    """Report a user-facing error on rank 0 and exit; in verbose mode re-raise ``exc``."""
    if verbose and exc is not None:
        raise exc
    if rank == 0:
        print(f"wavesim: error: {message}", file=sys.stderr)
    sys.exit(code)


def _plotting_preflight_error(cfg: dict) -> str | None:
    """Check figure prerequisites before any solve; identical result on every rank.

    Only rank 0 plots, so only it imports Matplotlib; the verdict is broadcast so
    all ranks exit together instead of failing after a long simulation.
    """
    error = None
    if rank == 0:
        from wavesimfem.analysis import resolve_usetex

        try:
            resolve_usetex(cfg["usetex"])
        except (RuntimeError, ValueError) as exc:
            error = str(exc)
    return comm.bcast(error, root=0)


def _frequency_arg(text: str) -> float:
    """Argparse type: SI number or quantity with a unit (``"2MHz"``), in Hz."""
    try:
        return parse_quantity(text, "frequency")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``wavesim`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="wavesim",
        description="WaveSimFEM: A GPU-accelerated multi-scale finite element framework for acoustic and elastic wave simulation in heterogeneous layered media",
        epilog="Options left unset fall back to the configuration file, then built-in defaults.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
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
        "--force",
        action="store_true",
        help="Allow --generate-config to overwrite an existing file",
    )
    parser.add_argument(
        "--solver",
        choices=["implicit", "explicit"],
        default=None,
        help="Time integration solver scheme override",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=f"Compute device for the explicit solver ({DEVICE_CHOICES_HELP}). "
        "'auto' falls back to CPU; an explicit GPU request fails if unavailable",
    )
    parser.add_argument(
        "--precision",
        choices=["double", "single"],
        default=None,
        help="Floating point precision for the single-process explicit engine",
    )
    parser.add_argument(
        "--structure-mode",
        choices=list(STRUCTURE_MODES),
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
        metavar="NPZ",
        help="Path to existing sensor_data.npz file to run signal processing without simulation",
    )
    for flag, what, example in (
        ("--f-min", "minimum frequency", "0"),
        ("--f-max", "maximum frequency", "2MHz"),
        ("--k-min", "minimum wavenumber", "0"),
        ("--k-max", "maximum wavenumber", "3rad/mm"),
    ):
        parser.add_argument(
            flag,
            type=str,
            default=None,
            metavar="Q",
            help=f"Dispersion plot {what}; SI number (Hz, rad/m) or with a unit, e.g. {example}",
        )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print preflight mesh, DOF, time-step, and memory estimates, then exit "
        "(requires -c/--config)",
    )
    parser.add_argument(
        "--min-frequency",
        type=_frequency_arg,
        default=None,
        metavar="Q",
        help="Analysis/sizing lower frequency bound override; Hz or with a unit, e.g. 200kHz",
    )
    parser.add_argument(
        "--max-frequency",
        type=_frequency_arg,
        default=None,
        metavar="Q",
        help="Analysis/sampling band and implicit accuracy frequency; Hz or with a unit "
        "(defaults to 3*f_center; does not change the mesh)",
    )
    parser.add_argument(
        "--duration", type=float, default=None, help="Total simulation duration override [s]"
    )
    parser.add_argument("--dt", type=float, default=None, help="Maximum time-step override [s]")
    parser.add_argument(
        "--cfl", type=float, default=None, help="Courant-Friedrichs-Lewy (CFL) factor override"
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Disable automated post-processing and figure generation",
    )
    parser.add_argument(
        "--usetex",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable external LaTeX figure rendering (overrides output.usetex)",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-q", "--quiet", action="store_true", help="Only print warnings and errors"
    )
    verbosity.add_argument(
        "-v", "--verbose", action="store_true", help="Print debug-level diagnostics"
    )
    return parser


@mpi_fail_fast
def main(argv: Sequence[str] | None = None) -> None:
    """Parse command-line arguments and orchestrate simulation and post-processing."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.quiet:
        set_verbosity("WARNING")
    elif args.verbose:
        set_verbosity("DEBUG")

    if args.postprocess and args.no_postprocess:
        _exit_with_error(
            "--postprocess and --no-postprocess cannot be used together",
            EXIT_USAGE,
        )

    # Fast template generation path
    if args.generate_config:
        if rank == 0:
            try:
                generate_template_config(args.generate_config, overwrite=args.force)
            except OSError as exc:
                _exit_with_error(
                    f"cannot write template configuration: {exc}",
                    EXIT_USAGE,
                    exc=exc,
                    verbose=args.verbose,
                )
            log(f"Template configuration file created: {args.generate_config}")
        comm.Barrier()
        return

    # A configuration file is mandatory for simulation and dry runs
    if not args.config and (args.dry_run or not args.postprocess):
        if rank == 0:
            if args.dry_run:
                parser.print_usage(sys.stderr)
                print("wavesim: error: --dry-run requires -c/--config", file=sys.stderr)
            else:
                parser.print_help()
                print(_MISSING_CONFIG_MESSAGE, file=sys.stderr)
        sys.exit(2 if args.dry_run else 1)

    # Resolve Defaults -> File -> CLI overrides on each rank. Filesystem errors
    # may differ by rank; mpi_fail_fast handles unsuccessful CLI exits as well.
    try:
        if args.config:
            cfg = parse_and_resolve_config(
                config_path=args.config, args=args, validate_simulation=not args.postprocess
            )
        else:  # postprocess-only: defaults + CLI overrides; unset plot limits come from the archive
            cfg = parse_and_resolve_config(custom_cfg={}, args=args, validate_simulation=False)
    except (OSError, ValueError, TypeError) as exc:
        _exit_with_error(f"invalid configuration: {exc}", EXIT_USAGE, exc=exc, verbose=args.verbose)

    will_postprocess = bool(args.postprocess or cfg["run_postprocess"]) and not args.no_postprocess
    if will_postprocess:
        error = _plotting_preflight_error(cfg)
        if error is not None:
            if not args.dry_run:
                _exit_with_error(f"invalid configuration: {error}", EXIT_USAGE)
            if rank == 0:
                # Dry runs are often made on a login node without the plotting
                # toolchain, so report rather than fail.
                warn(f"post-processing would fail on this machine: {error}")

    # Preflight dry-run path
    if args.dry_run:
        try:
            preflight_estimate(cfg, rank=rank)
        except (ValueError, TypeError) as exc:
            _exit_with_error(
                f"invalid configuration: {exc}", EXIT_USAGE, exc=exc, verbose=args.verbose
            )
        if rank == 0:
            log("\n  Dry run complete: preflight estimation finished successfully.")
            log("  Exiting without mesh generation or FEM assembly.\n")
        comm.Barrier()
        return

    # Step 1: Simulation (if not postprocess-only)
    if args.postprocess:
        if not os.path.isfile(args.postprocess):
            _exit_with_error(f"sensor data file not found: {args.postprocess}", EXIT_USAGE)
        sensor_data_file = args.postprocess
        out_dir = args.output or os.path.dirname(sensor_data_file) or "."
    else:
        try:
            sensor_data_file, out_dir = run_simulation(args, cfg=cfg)
        except DeviceUnavailableError as exc:
            _exit_with_error(str(exc), EXIT_DEVICE_UNAVAILABLE, exc=exc, verbose=args.verbose)
        except (OSError, ValueError, TypeError) as exc:
            _exit_with_error(str(exc), EXIT_USAGE, exc=exc, verbose=args.verbose)

    # Step 2: Signal Processing (Rank 0 only)
    if rank == 0:
        if will_postprocess:
            try:
                from wavesimfem.analysis import plot_full_analysis

                # SI limits (Hz, rad/m); None falls back to the archive's stored limits.
                limits = {
                    name: cfg[f"disp_{name}"] for name in ("f_min", "f_max", "k_min", "k_max")
                }
                plot_full_analysis(
                    sensor_data_file,
                    output_dir=out_dir,
                    **limits,
                    formats=cfg["figure_formats"],
                    usetex=cfg["usetex"],
                )
                if cfg["structure_analysis"]:
                    from wavesimfem.wave_reports import write_structure_report

                    write_structure_report(
                        sensor_data_file,
                        output_dir=out_dir,
                        **limits,
                        formats=cfg["figure_formats"],
                        usetex=cfg["usetex"],
                    )
                characterisation = cfg.get("wave_characterisation")
                if cfg.get("lamb_modes"):
                    from wavesimfem.lamb import write_lamb_report

                    write_lamb_report(
                        sensor_data_file,
                        output_dir=out_dir,
                        formats=cfg["figure_formats"],
                        usetex=cfg["usetex"],
                    )
                if characterisation and characterisation["enabled"]:
                    from wavesimfem.characterisation import write_characterisation

                    write_characterisation(
                        sensor_data_file,
                        output_dir=out_dir,
                        options=characterisation,
                        formats=cfg["figure_formats"],
                        usetex=cfg["usetex"],
                    )
            except (OSError, ValueError, TypeError, RuntimeError) as exc:
                # RuntimeError covers plotting-tool failures (e.g. LaTeX).
                _exit_with_error(str(exc), EXIT_USAGE, exc=exc, verbose=args.verbose)

        log(f"\n{'═' * 58}")
        log("  All tasks completed successfully!")
        log(f"  Output directory: {os.path.abspath(out_dir)}")
        log(f"{'═' * 58}\n")

    comm.Barrier()


if __name__ == "__main__":
    from wavesimfem._launcher import main as _launcher_main

    _launcher_main()
