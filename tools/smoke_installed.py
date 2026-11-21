#!/usr/bin/env python3
"""Exercise an installed WaveSimFEM wheel or sdist from outside its source checkout."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import sysconfig
import tempfile
from importlib.resources import files
from pathlib import Path

import numpy as np


def _check_not_from_checkout(package_file: Path) -> None:
    checkout = Path(__file__).resolve().parents[1]
    resolved = package_file.resolve()
    for forbidden in (checkout / "src", Path.cwd().resolve() / "src"):
        if forbidden in resolved.parents:
            raise RuntimeError(
                f"wavesimfem imported from source tree instead of installed package: {resolved}"
            )


def _check_pure_python_api() -> str:
    import wavesimfem

    if not wavesimfem.__file__:
        raise RuntimeError("wavesimfem.__file__ is unexpectedly None")
    _check_not_from_checkout(Path(wavesimfem.__file__))

    if not wavesimfem.__version__ or wavesimfem.__version__ == "0.0.0+unknown":
        raise RuntimeError(f"Invalid installed package version: {wavesimfem.__version__!r}")

    if not files("wavesimfem").joinpath("py.typed").is_file():
        raise RuntimeError("Packaged wavesimfem/py.typed marker is missing")

    # Exercise lazy pure-Python public exports without importing DOLFINx/MPI.
    assert wavesimfem.parse_quantity("2 MHz", "frequency") == 2.0e6
    wavelet = wavesimfem.ricker(np.linspace(0.0, 2e-6, 32), 1.0e6, 1.0e-6)
    if wavelet.shape != (32,) or not np.all(np.isfinite(wavelet)):
        raise RuntimeError("ricker() returned unexpected waveform")

    with tempfile.TemporaryDirectory(prefix="wavesim-smoke-") as tmp:
        template_path = Path(tmp) / "template.yaml"
        wavesimfem.generate_template_config(str(template_path))
        cfg = wavesimfem.parse_and_resolve_config(str(template_path))
        if cfg["structure_mode"] != "1_layer":
            raise RuntimeError(f"Unexpected template structure_mode: {cfg['structure_mode']!r}")

    return wavesimfem.__version__


def _has_fem_stack() -> bool:
    return all(importlib.util.find_spec(mod) is not None for mod in ("mpi4py", "dolfinx"))


def _check_cli(version: str, *, require_fem: bool) -> None:
    env = os.environ.copy()
    scripts_dir = sysconfig.get_path("scripts")
    if scripts_dir:
        env["PATH"] = f"{scripts_dir}{os.pathsep}{env.get('PATH', '')}"

    has_fem = _has_fem_stack()
    if require_fem and not has_fem:
        raise RuntimeError("--require-fem specified, but dolfinx/mpi4py are not installed")

    if has_fem:
        for cmd in (["wavesim", "--version"], [sys.executable, "-m", "wavesimfem", "--version"]):
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
            if version not in result.stdout:
                raise RuntimeError(f"{cmd} did not report version {version}: {result.stdout!r}")
        subprocess.run(["wavesim", "--help"], check=True, capture_output=True, text=True, env=env)
        with tempfile.TemporaryDirectory(prefix="wavesim-cli-") as tmp:
            out_yaml = Path(tmp) / "generated.yaml"
            subprocess.run(
                ["wavesim", "--generate-config", str(out_yaml)],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            if not out_yaml.is_file() or out_yaml.stat().st_size == 0:
                raise RuntimeError("wavesim --generate-config failed to create template file")
    else:
        # In pure-Python environments without conda-forge DOLFINx/MPI, verify the
        # entry point exists and fails with a clear guidance message (not a traceback).
        result = subprocess.run(
            ["wavesim", "--version"], check=False, capture_output=True, text=True, env=env
        )
        if result.returncode == 0:
            if version not in result.stdout:
                raise RuntimeError(f"wavesim --version mismatch: {result.stdout!r}")
        elif "conda-forge" not in result.stderr and "environment.yml" not in result.stderr:
            raise RuntimeError(
                "wavesim CLI failed without helpful FEM installation guidance:\n"
                f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-fem",
        action="store_true",
        help="Require DOLFINx and mpi4py and exercise full CLI entry points",
    )
    args = parser.parse_args(argv)
    version = _check_pure_python_api()
    _check_cli(version, require_fem=args.require_fem)
    print(f"Smoke test passed for installed wavesimfem {version}")


if __name__ == "__main__":
    main()
