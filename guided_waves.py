"""Guided Waves FEM Simulation & Dispersion Extraction
===================================================

Architecture & Code Organization:
---------------------------------
guided_waves.py
├── Module Docstring & High-Level Architecture
├── MPI Communicator Initialization
├── SECTION 1: Configuration Management (Loaders, Serializers, & Templates)
│   ├── load_config(...)
│   ├── dump_config(...)
│   ├── flatten_config(...)
│   ├── parse_and_resolve_config(...)
│   └── generate_template_config(...)
├── SECTION 2: Structural Geometry & Material Constitutive Modeling
│   ├── _canonicalize_mode(...)
│   ├── compute_rayleigh_speed(...)
│   ├── _normalize_region(...)
│   ├── _normalize_structure(...)
│   ├── _format_banner_box(...)
│   └── _print_structure_table(...)
├── SECTION 3: Signal Processing & 2D FFT Dispersion Analysis
│   └── plot_full_analysis(...)
├── SECTION 4: Finite Element Wave Propagation Simulation
│   └── run_simulation(...)  [Implicit Newmark-β & Explicit Lumped RK4]
└── SECTION 5: Command-Line Interface & Application Orchestration
    └── main()

Features:
---------
  1. Multi-Format Configuration Management:
     - External config files: YAML (.yaml/.yml), TOML (.toml), or JSON (.json)
     - Interactive command line overrides for any parameter
     - Mandatory configuration file to ensure explicit, reproducible simulation parameters
     - Automatic generation of fully documented template configuration files
  2. Structural Waveguide Modes (selectable in config file or via CLI):
     [Category A: Free Plates in Vacuum — Guided Lamb Waves]
     - "1_layer"             : Single homogeneous plate in vacuum
     - "N_layers"            : Multilayered plate laminate in vacuum (piecewise constitutive tensors)

     [Category B: Semi-Infinite Medium — Rayleigh Surface Waves]
     - "halfspace"           : Pristine elastic halfspace (traction-free surface, bottom sponge)

     [Category C: Plate-Halfspace Systems — Leaky & Interface Waves]
     - "plate_on_halfspace"  : Multilayered (or single) plate on bottom substrate halfspace
     - "plate_in_halfspaces" : Plate embedded between 2 halfspaces (substrate below, superstrate above)
  3. Dual Time-Integration Solvers:
      - "explicit" : Lumped mass RK4 with conservative assembled-operator step bounds
     - "implicit" : Unconditionally stable Newmark-β (average acceleration) with PETSc linear solvers
  4. Signal Processing & Dispersion Analysis:
     - 2D spatio-temporal Fourier transform (2D FFT) for u_1 (in-plane) and u_2 (out-of-plane)
     - Hann windowing, zero-padding, and dynamic range filtering (dB)
     - Analytical bulk wave speed reference slopes (c_L, c_T, c_R, c_f)
     - Vector exports: .png (300 DPI), .pdf, .svg, and .eps
  5. High-Performance Parallel Execution:
     - MPI parallelization across mesh partitioning, assembly, and time integration
     - Automated rank-0 console logging, serialization, and figure generation
  6. Reproducibility & Provenance:
     - Exact resolved configuration snapshot serialized alongside simulation results

Usage:
------
  # Run simulation with a required configuration file:
  python guided_waves.py -c configs/01_single_layer_aluminum.yaml
  mpirun -np 4 python guided_waves.py -c configs/03_multilayer_bone_aluminum.yaml

  # CLI overrides on top of configuration:
  python guided_waves.py -c configs/01_single_layer_aluminum.yaml --solver implicit --output my_output

  # Post-process existing simulation data:
  python guided_waves.py --postprocess results_guided_waves_explicit/sensor_data.npz

  # Generate a documented template configuration:
  python guided_waves.py --generate-config my_custom_case.yaml
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import platform
import shutil
import sys
import time as pytime
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import tomllib

    HAS_TOMLLIB = True
except ImportError:
    try:
        import tomli as tomllib

        HAS_TOMLLIB = True
    except ImportError:
        HAS_TOMLLIB = False

try:
    import dolfinx.fem.petsc
    import ufl
    from dolfinx import default_scalar_type, fem, io
    from dolfinx.mesh import (
        CellType,
        create_rectangle,
        locate_entities,
        locate_entities_boundary,
        meshtags,
    )
    from mpi4py import MPI
    from petsc4py import PETSc
except ImportError as _err:
    raise ImportError(
        f"GuidedWaveFEM requires the FEniCSx/PETSc numerical stack ({_err}).\n"
        "Please install and activate the 'fenicsx-waves' Conda environment:\n"
        "  conda env create -f environment.yml\n"
        "  conda activate fenicsx-waves"
    ) from _err

from wave_numerics import (
    PointSensors,
    dispersion_spectrum,
    operator_bound,
    rk4_step,
    time_grid,
    validate_tensor,
)

import matplotlib

matplotlib.use("Agg")  # safe for headless & MPI execution
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MaxNLocator  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

SOURCE_DIR = Path(__file__).resolve().parent



def log(msg: str = "") -> None:
    """Print message with immediate flush for real-time progress logging.

    Parameters
    ----------
    msg : str, optional
        Message text to log. Defaults to empty string.
    """
    print(msg, flush=True)


# ==============================================================================
# MPI Communicator Setup (Must precede rank-dependent operations)
# ==============================================================================
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size


def mpi_fail_fast(function):
    """Prevent peers hanging at collectives after a rank-local exception."""
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception:
            if nprocs > 1:
                import traceback
                traceback.print_exc()
                comm.Abort(1)
            raise
    return wrapped


# ==============================================================================
# SECTION 1: Configuration Management (Loaders, Serializers, & Templates)
# ==============================================================================


def load_config(config_path: str) -> dict[str, Any]:
    """Load simulation configuration from a YAML, TOML, or JSON file.

    Parameters
    ----------
    config_path : str
        Path to configuration file (.yaml, .yml, .toml, or .json).

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    ext = os.path.splitext(config_path)[1].lower()
    if ext in (".yaml", ".yml"):
        if not HAS_YAML:
            raise ImportError(
                "PyYAML package is required to load YAML files. Install with: conda install pyyaml"
            )
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif ext == ".toml":
        if not HAS_TOMLLIB:
            raise ImportError(
                "Python 3.11+ built-in tomllib or tomli package is required to load TOML files."
            )
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
    elif ext == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(
            f"Unsupported configuration format '{ext}'. Supported formats: .yaml, .yml, .toml, .json"
        )

    if not isinstance(data, dict):
        raise TypeError(
            f"Configuration root in {config_path} must be a key-value mapping."
        )

    return data


def dump_config(cfg_dict: dict[str, Any], out_path: str) -> None:
    """Serialize and save configuration dictionary to YAML or JSON format.

    Parameters
    ----------
    cfg_dict : dict
        Configuration dictionary to serialize.
    out_path : str
        Destination path for serialized configuration file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    ext = os.path.splitext(out_path)[1].lower()

    def _sanitize(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        elif isinstance(item, (bool, np.bool_)):
            return bool(item)
        elif isinstance(item, (np.floating, float)):
            return float(item)
        elif isinstance(item, (np.integer, int)):
            return int(item)
        elif isinstance(item, dict):
            return {k: _sanitize(v) for k, v in item.items() if not k.startswith("_")}
        elif isinstance(item, (list, tuple)):
            return [_sanitize(v) for v in item]
        return item

    cleaned = _sanitize(cfg_dict)
    if ext in (".yaml", ".yml") and HAS_YAML:
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(cleaned, f, default_flow_style=False, sort_keys=False)
        return

    # Fallback to JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2)


def flatten_config(cfg_in: dict[str, Any]) -> dict[str, Any]:
    """Normalize nested hierarchical configuration dictionary into a flat parameter mapping.

    Extracts settings from sections ('solver', 'geometry', 'structure',
    'absorbing_boundaries', 'excitation', 'mesh', 'sensors', 'postprocess', 'output')
    into standardized top-level keys.

    Parameters
    ----------
    cfg_in : dict
        Hierarchical or flat configuration dictionary.

    Returns
    -------
    dict
        Flattened configuration dictionary.
    """
    flat: dict[str, Any] = {}
    if not isinstance(cfg_in, dict):
        return flat

    # 1. Solver
    if "solver" in cfg_in and isinstance(cfg_in["solver"], dict):
        s = cfg_in["solver"]
        if "type" in s:
            flat["solver_type"] = s["type"]
        if "CFL" in s:
            flat["CFL"] = float(s["CFL"])
        if "time_factor" in s:
            flat["time_factor"] = float(s["time_factor"])
        if "beta" in s:
            flat["beta_val"] = float(s["beta"])
        if "gamma" in s:
            flat["gamma_val"] = float(s["gamma"])
        for key in ("dt", "duration"):
            if key in s:
                flat[key] = s[key]

    # 2. Geometry
    if "geometry" in cfg_in and isinstance(cfg_in["geometry"], dict):
        g = cfg_in["geometry"]
        if "sensing_length" in g:
            flat["sensing_length"] = float(g["sensing_length"])
        if "absorb_length" in g:
            flat["absorb_length"] = float(g["absorb_length"])

    # 3. Structure & Materials
    if "structure" in cfg_in and isinstance(cfg_in["structure"], dict):
        st = cfg_in["structure"]
        if "mode" in st:
            flat["structure_mode"] = str(st["mode"])
        if "single_layer" in st:
            flat["single_layer"] = st["single_layer"]
        if "layers" in st:
            flat["layers"] = st["layers"]
        if "halfspace" in st:
            flat["halfspace"] = st["halfspace"]
        if "substrate" in st:
            flat["substrate"] = st["substrate"]
        if "superstrate" in st:
            flat["superstrate"] = st["superstrate"]
        if "bottom_halfspace" in st:
            flat["bottom_halfspace"] = st["bottom_halfspace"]
        if "top_halfspace" in st:
            flat["top_halfspace"] = st["top_halfspace"]

    # 4. Absorbing Boundaries
    if "absorbing_boundaries" in cfg_in and isinstance(
        cfg_in["absorbing_boundaries"], dict
    ):
        ab = cfg_in["absorbing_boundaries"]
        if "alpha" in ab:
            flat["absorb_alpha"] = (
                float(ab["alpha"]) if ab["alpha"] is not None else None
            )
        if "beta" in ab:
            flat["absorb_beta"] = float(ab["beta"]) if ab["beta"] is not None else None

    # 5. Excitation
    if "excitation" in cfg_in and isinstance(cfg_in["excitation"], dict):
        ex = cfg_in["excitation"]
        if "f_center" in ex:
            flat["f_center"] = float(ex["f_center"])
        if "traction_max" in ex:
            flat["traction_max"] = float(ex["traction_max"])
        if "excitation_width" in ex:
            flat["excitation_width"] = float(ex["excitation_width"])
        if "source_x" in ex:
            flat["source_x"] = (
                float(ex["source_x"]) if ex["source_x"] is not None else None
            )
        if "source_fx" in ex:
            flat["source_fx"] = float(ex["source_fx"])
        if "source_fy" in ex:
            flat["source_fy"] = float(ex["source_fy"])

    # 6. Mesh
    if "mesh" in cfg_in and isinstance(cfg_in["mesh"], dict):
        m = cfg_in["mesh"]
        if "elems_per_wavelength" in m:
            flat["elems_per_wavelength"] = m["elems_per_wavelength"]
        if "min_elems_x" in m:
            flat["min_elems_x"] = m["min_elems_x"]
        if "min_elems_y" in m:
            flat["min_elems_y"] = m["min_elems_y"]
        if "degree" in m:
            flat["degree"] = m["degree"]
        if "max_frequency" in m:
            flat["max_frequency"] = m["max_frequency"]

    # 7. Sensors
    if "sensors" in cfg_in and isinstance(cfg_in["sensors"], dict):
        sn = cfg_in["sensors"]
        if "n_sensors" in sn:
            flat["n_sensors"] = sn["n_sensors"]
        if "sensor_offset" in sn:
            flat["sensor_offset"] = float(sn["sensor_offset"])
        if "sensor_pitch" in sn:
            flat["sensor_pitch"] = float(sn["sensor_pitch"])
        if "sensor_y" in sn and sn["sensor_y"] is not None:
            flat["sensor_y"] = float(sn["sensor_y"])

    # 8. Postprocess
    if "postprocess" in cfg_in and isinstance(cfg_in["postprocess"], dict):
        pp = cfg_in["postprocess"]
        if "run_postprocess" in pp:
            flat["run_postprocess"] = bool(pp["run_postprocess"])
        if "f_min_disp_MHz" in pp:
            flat["f_min_disp_MHz"] = float(pp["f_min_disp_MHz"])
        if "f_max_disp_MHz" in pp:
            flat["f_max_disp_MHz"] = float(pp["f_max_disp_MHz"])
        if "k_min_disp" in pp:
            flat["k_min_disp"] = float(pp["k_min_disp"])
        if "k_max_disp" in pp:
            flat["k_max_disp"] = float(pp["k_max_disp"])

    # 9. Output
    if "output" in cfg_in and isinstance(cfg_in["output"], dict):
        o = cfg_in["output"]
        if "output_dir" in o:
            flat["output_dir"] = o["output_dir"]
        if "target_snapshots" in o:
            flat["target_snapshots"] = o["target_snapshots"]
        if "write_vtx" in o:
            flat["write_vtx"] = bool(o["write_vtx"])
        if "quick_plot" in o:
            flat["quick_plot"] = bool(o["quick_plot"])

    # Also accept any direct top-level flat parameter keys
    for k, v in cfg_in.items():
        if k not in (
            "solver",
            "geometry",
            "structure",
            "absorbing_boundaries",
            "excitation",
            "mesh",
            "sensors",
            "postprocess",
            "output",
        ):
            flat[k] = v

    return flat


def parse_and_resolve_config(
    config_path: str | None = None,
    args: argparse.Namespace | None = None,
    custom_cfg: dict[str, Any] | None = None,
    validate_simulation: bool = True,
) -> dict[str, Any]:
    """Load, flatten, validate, and apply CLI overrides to the configuration.

    Parameters
    ----------
    config_path : str, optional
        Path to external configuration file (.yaml, .toml, or .json).
    args : argparse.Namespace, optional
        Command-line argument namespace containing overrides.
    custom_cfg : dict, optional
        In-memory configuration dictionary.
    validate_simulation : bool, default=True
        Whether to enforce mandatory simulation parameters (sensing_length, absorb_length, f_center).

    Returns
    -------
    dict
        Fully resolved and validated configuration dictionary.

    Raises
    ------
    ValueError
        If no configuration source is provided or required physical fields are missing.
    """
    if custom_cfg is not None:
        raw_cfg = custom_cfg
        config_source = custom_cfg.get("_config_source", "Supplied Dictionary")
    elif config_path is not None:
        raw_cfg = load_config(config_path)
        config_source = config_path
    elif args is not None and getattr(args, "config", None):
        config_path = args.config
        raw_cfg = load_config(config_path)
        config_source = config_path
    else:
        raise ValueError(
            "A configuration file path or configuration dictionary must be provided."
        )

    resolved = flatten_config(raw_cfg)

    # CLI overrides take highest precedence
    if args is not None:
        if getattr(args, "solver", None) is not None:
            resolved["solver_type"] = args.solver
        if getattr(args, "structure_mode", None) is not None:
            resolved["structure_mode"] = args.structure_mode
        if getattr(args, "output", None) is not None:
            resolved["output_dir"] = args.output
        if getattr(args, "f_min", None) is not None:
            resolved["f_min_disp_MHz"] = args.f_min
        if getattr(args, "f_max", None) is not None:
            resolved["f_max_disp_MHz"] = args.f_max
        if getattr(args, "k_min", None) is not None:
            resolved["k_min_disp"] = args.k_min
        if getattr(args, "k_max", None) is not None:
            resolved["k_max_disp"] = args.k_max
        if getattr(args, "max_frequency", None) is not None:
            resolved["max_frequency"] = args.max_frequency
        if getattr(args, "duration", None) is not None:
            resolved["duration"] = args.duration
        if getattr(args, "dt", None) is not None:
            resolved["dt"] = args.dt
        if getattr(args, "cfl", None) is not None:
            resolved["CFL"] = args.cfl
        if getattr(args, "no_postprocess", False):
            resolved["run_postprocess"] = False

    # Numerical solver fallbacks
    resolved.setdefault("solver_type", "explicit")
    resolved.setdefault("CFL", 0.5)
    resolved.setdefault("time_factor", 1.2)
    resolved.setdefault("beta_val", 0.25)
    resolved.setdefault("gamma_val", 0.5)
    resolved.setdefault("elems_per_wavelength", 10)
    resolved.setdefault("min_elems_x", 100)
    resolved.setdefault("min_elems_y", 4)
    resolved.setdefault("degree", 2)
    resolved.setdefault("n_sensors", 301)
    resolved.setdefault("sensor_offset", 0.05)
    resolved.setdefault("sensor_pitch", 0.5e-3)
    resolved.setdefault("traction_max", 1.0e6)
    resolved.setdefault("excitation_width", 2.0e-3)
    resolved.setdefault("source_fx", 0.0)
    resolved.setdefault("source_fy", 1.0)
    resolved.setdefault("target_snapshots", 200)
    resolved.setdefault("write_vtx", False)
    resolved.setdefault("run_postprocess", True)
    resolved.setdefault("quick_plot", True)
    resolved.setdefault("f_min_disp_MHz", 0.0)
    resolved.setdefault("f_max_disp_MHz", 2.0)
    resolved.setdefault("k_min_disp", 0.0)
    resolved.setdefault("k_max_disp", 3.0)

    # Required geometry & physical checks
    if validate_simulation:
        if "sensing_length" not in resolved:
            raise ValueError(
                f"Configuration '{config_source}' must specify 'geometry.sensing_length' (or 'sensing_length')."
            )
        if "absorb_length" not in resolved:
            raise ValueError(
                f"Configuration '{config_source}' must specify 'geometry.absorb_length' (or 'absorb_length')."
            )
        if "f_center" not in resolved:
            raise ValueError(
                f"Configuration '{config_source}' must specify 'excitation.f_center' (or 'f_center')."
            )

    if resolved.get("source_x") is None and "absorb_length" in resolved:
        resolved["source_x"] = float(resolved["absorb_length"]) + 0.005

    if resolved.get("bottom_halfspace") is None and "substrate" in resolved:
        resolved["bottom_halfspace"] = resolved["substrate"]
    if resolved.get("top_halfspace") is None and "superstrate" in resolved:
        resolved["top_halfspace"] = resolved["superstrate"]

    resolved["_config_source"] = config_source
    if validate_simulation:
        positive = ("sensing_length", "absorb_length", "f_center", "excitation_width",
                    "sensor_pitch", "CFL", "time_factor")
        for key in positive:
            value = float(resolved[key])
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("dt", "duration", "max_frequency"):
            if resolved.get(key) is not None:
                value = float(resolved[key])
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f"{key} must be finite and positive")
                resolved[key] = value
        for key, minimum in (("n_sensors", 2), ("elems_per_wavelength", 1),
                             ("min_elems_x", 1), ("min_elems_y", 1),
                             ("target_snapshots", 1), ("degree", 1)):
            value = resolved[key]
            if not np.isfinite(value) or int(value) != value or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        for key in ("source_x", "source_fx", "source_fy", "traction_max", "sensor_offset"):
            if not np.isfinite(float(resolved[key])):
                raise ValueError(f"{key} must be finite")
        if resolved["source_fx"] == resolved["source_fy"] == 0:
            raise ValueError("Source direction cannot be zero")
        for key in ("absorb_alpha", "absorb_beta"):
            if resolved.get(key) is not None:
                value = float(resolved[key])
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f"{key} must be finite and nonnegative")
        solver = str(resolved["solver_type"]).lower()
        if solver not in ("implicit", "explicit"):
            raise ValueError("solver_type must be explicit or implicit")
        resolved["solver_type"] = solver
        if solver == "explicit" and resolved["CFL"] > 1:
            raise ValueError("Explicit CFL must be in (0, 1]")
        beta, gamma = float(resolved["beta_val"]), float(resolved["gamma_val"])
        if solver == "implicit" and (not np.isfinite([beta, gamma]).all()
                                     or gamma < .5 or beta < .25*(gamma+.5)**2):
            raise ValueError("Newmark requires gamma >= 0.5 and beta >= (gamma+0.5)^2/4")
        length = 2*resolved["absorb_length"] + resolved["sensing_length"]
        if not 0 <= resolved["source_x"] <= length:
            raise ValueError("source_x must be inside the domain")
        sensor_start = resolved["source_x"] + resolved["sensor_offset"]
        sensor_end = sensor_start + (resolved["n_sensors"]-1)*resolved["sensor_pitch"]
        if sensor_start < 0 or sensor_end > length:
            raise ValueError("Sensor array must lie inside the domain")
        if resolved["run_postprocess"] and resolved["n_sensors"] < 3:
            raise ValueError("Postprocessing requires at least three sensors")
    for low, high in (("f_min_disp_MHz", "f_max_disp_MHz"), ("k_min_disp", "k_max_disp")):
        if not np.isfinite([resolved[low], resolved[high]]).all() or not 0 <= resolved[low] < resolved[high]:
            raise ValueError(f"Require 0 <= {low} < {high}, with finite limits")
    return resolved


# Backward-compatibility alias
resolve_and_apply_config = parse_and_resolve_config


def generate_template_config(output_path: str) -> None:
    """Generate an annotated template configuration file in YAML format.

    Parameters
    ----------
    output_path : str
        Target file path for the generated template configuration.
    """
    template_content = """# ==============================================================================
# Guided Wave FEM Simulation Configuration Template
# ==============================================================================

solver:
  type: "explicit"              # "explicit" (RK4) or "implicit" (Newmark-beta)
  CFL: 0.5                      # Courant-Friedrichs-Lewy stability factor
  time_factor: 1.2              # Total duration multiplier: T_total = time_factor * L / c_T
  beta: 0.25                    # Newmark-beta (implicit only)
  gamma: 0.5                    # Newmark-gamma (implicit only)
  dt: null                      # Optional requested maximum step [s]
  duration: null                # Optional duration [s]; overrides time_factor

geometry:
  sensing_length: 0.30          # [m] Pristine observation region length
  absorb_length: 0.15           # [m] 5th-order absorbing sponge width on lateral boundaries

structure:
  # Mode choices: "1_layer", "N_layers", "halfspace", "plate_on_halfspace", "plate_in_halfspaces"
  mode: "1_layer"

  # Single layer definition (mode: "1_layer")
  single_layer:
    name: "Aluminum Plate"
    thickness: 1.0e-3           # [m] Layer thickness (1 mm)
    material_model: "isotropic" # "isotropic", "anisotropic", or "viscoelastic"
    rho: 2700.0                 # [kg/m³] Density
    E: 70.0e9                   # [Pa] Young's modulus
    nu: 0.33                    # [-] Poisson's ratio
    min_elems_y: 6              # Minimum elements through thickness

absorbing_boundaries:
  alpha: null                   # [1/s] Peak mass-proportional damping (null = auto)
  beta: null                    # [s] Peak stiffness-proportional damping (null = auto)

excitation:
  f_center: 1.0e6               # [Hz] Center frequency of Ricker wavelet
  traction_max: 1.0e6           # [Pa] Surface load amplitude
  excitation_width: 2.0e-3      # [m] Spatial Gaussian width
  source_x: null                # [m] Excitation x-position (null = absorb_length + 5 mm)
  source_fx: 1.0                # Horizontal traction vector component
  source_fy: 1.0                # Vertical traction vector component

mesh:
  elems_per_wavelength: 10      # Spatial discretization density
  min_elems_x: 100              # Longitudinal element floor
  min_elems_y: 4                # Transverse element floor
  degree: 2                     # Polynomial order (Q2 quadrilateral elements)
  max_frequency: null           # [Hz] Mesh sizing frequency; default 3*f_center

sensors:
  n_sensors: 401                # Measurement points
  sensor_offset: 0.05           # [m] Distance from excitation to 1st sensor
  sensor_pitch: 0.5e-3          # [m] Distance between sensors

postprocess:
  run_postprocess: true         # Compute 2D FFT dispersion analysis
  f_min_disp_MHz: 0.0           # [MHz] Min frequency on dispersion axis
  f_max_disp_MHz: 2.0           # [MHz] Max frequency on dispersion axis
  k_min_disp: 0.0               # [rad/mm] Min wavenumber on dispersion axis
  k_max_disp: 3.0               # [rad/mm] Max wavenumber on dispersion axis

output:
  output_dir: null              # Result directory (null = auto: results_guided_waves_{solver})
  write_vtx: false              # ADIOS2 VTX (.bp) Paraview export
  target_snapshots: 200         # Target snapshot count
  quick_plot: true              # Disabled together with --no-postprocess
"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(template_content)


# ==============================================================================
# SECTION 2: Structural Geometry & Material Constitutive Modeling
# ==============================================================================


def _canonicalize_mode(mode_str: str | None) -> str:
    """Normalize user-provided structure mode string to a canonical identifier.

    Parameters
    ----------
    mode_str : str, optional
        User-provided structure mode name.

    Returns
    -------
    str
        Canonical mode string ('1_layer', 'N_layers', 'halfspace',
        'plate_on_halfspace', or 'plate_in_halfspaces').

    Raises
    ------
    ValueError
        If ``mode_str`` does not match any recognized structural category.
    """
    if mode_str is None:
        return "N_layers"
    s = str(mode_str).lower().replace("-", "_").replace(" ", "_").strip()
    if s in ("1_layer", "1layer", "single", "single_layer", "1", "one_layer"):
        return "1_layer"
    elif s in ("n_layers", "nlayers", "multilayer", "multi_layer", "n"):
        return "N_layers"
    elif s in (
        "halfspace",
        "half_space",
        "rayleigh",
        "single_halfspace",
        "halfspace_only",
    ):
        return "halfspace"
    elif any(
        k in s
        for k in (
            "on_halfspace",
            "over_halfspace",
            "over_a_halfspace",
            "on_a_halfspace",
        )
    ):
        return "plate_on_halfspace"
    elif any(
        k in s
        for k in (
            "between_2_halfspaces",
            "between_halfspaces",
            "in_halfspaces",
            "embedded",
        )
    ):
        return "plate_in_halfspaces"
    raise ValueError(
        f"Unknown structure_mode '{mode_str}'. "
        f"Choose from: '1_layer', 'N_layers', 'halfspace', 'plate_on_halfspace', 'plate_in_halfspaces'."
    )


def compute_rayleigh_speed(
    c_T: float, c_L: float | None = None, nu: float | None = None
) -> float:
    """Compute exact Rayleigh surface acoustic wave speed c_R for isotropic solid.

    Solves Rayleigh's cubic secular equation:
        xi^3 - 8*xi^2 + 8*(3 - 2*eta)*xi - 16*(1 - eta) = 0
    where xi = (c_R / c_T)^2 and eta = (c_T / c_L)^2 = (1 - 2*nu) / (2*(1 - nu)).

    Parameters
    ----------
    c_T : float
        Transverse / shear bulk wave speed [m/s].
    c_L : float, optional
        Longitudinal bulk wave speed [m/s].
    nu : float, optional
        Poisson's ratio [-].

    Returns
    -------
    float
        Exact Rayleigh surface wave speed [m/s].
    """
    if nu is None and c_L is not None:
        eta = (c_T / c_L) ** 2
    elif nu is not None:
        eta = (1.0 - 2.0 * nu) / (2.0 * (1.0 - nu))
    else:
        return 0.92 * c_T
    # Rayleigh's cubic equation: ξ³ - 8ξ² + 8(3 - 2η)ξ - 16(1 - η) = 0 where ξ = (c_R / c_T)²
    roots = np.roots([1.0, -8.0, 8.0 * (3.0 - 2.0 * eta), -16.0 * (1.0 - eta)])
    valid = [r.real for r in roots if np.isreal(r) and 0.0 < r.real < 1.0]
    if len(valid) > 0:
        return float(c_T * np.sqrt(valid[0]))
    return (
        float(c_T * (0.862 + 1.14 * nu) / (1.0 + nu)) if nu is not None else 0.92 * c_T
    )


def _normalize_region(
    item: dict[str, Any],
    default_name: str,
    role: str,
    is_halfspace: bool,
    halfspace_type: str | None,
    y_bot: float,
) -> dict[str, Any]:
    """Validate and precompute material parameters for a single subdomain region.

    Parameters
    ----------
    item : dict
        Raw configuration mapping for the layer/halfspace region.
    default_name : str
        Default region identifier if 'name' is not provided.
    role : str
        Subdomain description label for logging and table display.
    is_halfspace : bool
        Whether this region represents an infinite halfspace.
    halfspace_type : str, optional
        'bottom' for substrate, 'top' for superstrate, or None for plates.
    y_bot : float
        Bottom vertical coordinate [m] of the subdomain.

    Returns
    -------
    dict
        Normalized region dictionary containing thickness, wave speeds, and constitutive tensors.
    """
    name = str(item.get("name", default_name))
    h = float(item["thickness"])
    if not np.isfinite(h) or h <= 0:
        raise ValueError(f"Region '{name}' thickness must be positive, got {h}")
    model = str(item.get("material_model", "anisotropic")).lower().strip()
    if model in ("fluid", "acoustic_fluid", "water", "acoustic"):
        model = "fluid"
    if model not in ("isotropic", "anisotropic", "viscoelastic", "fluid"):
        raise ValueError(
            f"Region '{name}' has unknown material_model '{model}'. "
            f"Must be 'isotropic', 'anisotropic', 'viscoelastic', or 'fluid'."
        )
    rho = float(item["rho"])
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError(f"Region '{name}' density rho must be positive, got {rho}")

    entry = {
        "name": name,
        "role": role,
        "thickness": h,
        "material_model": model,
        "rho": rho,
        "y_bot": y_bot,
        "y_top": y_bot + h,
        "is_halfspace": is_halfspace,
        "halfspace_type": halfspace_type,
        "min_elems_y": item.get("min_elems_y"),
        "is_fluid": (model == "fluid"),
    }
    if entry["min_elems_y"] is not None:
        value = entry["min_elems_y"]
        if not np.isfinite(value) or int(value) != value or value < 1:
            raise ValueError(f"Region '{name}' min_elems_y must be a positive integer")
        entry["min_elems_y"] = int(value)

    if model == "fluid":
        c_sound = None
        for key in ("c", "c_sound", "c_fluid", "speed"):
            if key in item and item[key] is not None:
                c_sound = float(item[key])
                break

        K_val = None
        for key in ("K", "bulk_modulus", "bulk_mod"):
            if key in item and item[key] is not None:
                K_val = float(item[key])
                break

        if c_sound is not None:
            if not np.isfinite(c_sound) or c_sound <= 0:
                raise ValueError(
                    f"Fluid '{name}' sound speed must be positive, got {c_sound}"
                )
            K_val = rho * (c_sound**2)
        elif K_val is not None:
            if not np.isfinite(K_val) or K_val <= 0:
                raise ValueError(
                    f"Fluid '{name}' bulk modulus must be positive, got {K_val}"
                )
            c_sound = float(np.sqrt(K_val / rho))
        elif "E" in item and "nu" in item:
            E = float(item["E"])
            nu = float(item["nu"])
            if not np.isfinite([E, nu]).all() or E <= 0 or not -1 < nu < .5:
                raise ValueError(f"Fluid '{name}' requires E > 0 and -1 < nu < 0.5")
            K_val = E / (3.0 * (1.0 - 2.0 * nu))
            c_sound = float(np.sqrt(K_val / rho))
        else:
            raise ValueError(f"Fluid '{name}' requires a sound speed or bulk modulus")

        fluid_shear_reg = float(item.get("fluid_shear_reg", 1e-8))
        if not np.isfinite(fluid_shear_reg) or not 0 < fluid_shear_reg <= 1e-2:
            raise ValueError(
                f"Fluid '{name}' fluid_shear_reg must be in (0, 0.01], got {fluid_shear_reg}"
            )

        mu = fluid_shear_reg * K_val
        lmbda = K_val - (2.0 / 3.0) * mu
        c_L = float(np.sqrt((lmbda + 2*mu) / rho))
        c_T = float(np.sqrt(mu / rho))
        nu_eff = lmbda / (2.0 * (lmbda + mu))
        E_eff = 2.0 * mu * (1.0 + nu_eff)

        entry.update(
            c_fluid=c_sound,
            K=K_val,
            fluid_shear_reg=fluid_shear_reg,
            E=E_eff,
            nu=nu_eff,
            mu=mu,
            lmbda=lmbda,
            c_L=c_L,
            c_T=c_T,
            c_physical_min=c_sound,
        )
    elif model == "isotropic":
        E = float(item["E"])
        nu = float(item["nu"])
        if not np.isfinite([E, nu]).all() or E <= 0 or nu >= 0.5 or nu <= -1.0:
            raise ValueError(
                f"Region '{name}': require finite E > 0 and -1 < nu < 0.5"
            )
        mu = E / (2.0 * (1.0 + nu))
        lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        c_L = float(np.sqrt((lmbda + 2.0 * mu) / rho))
        c_T = float(np.sqrt(mu / rho))
        entry.update(
            E=E, nu=nu, mu=mu, lmbda=lmbda, c_L=c_L, c_T=c_T, c_physical_min=c_T
        )
    else:
        C = [float(v) for v in item["C"]]
        if len(C) != 6:
            raise ValueError(
                f"Region '{name}': C matrix must have 6 entries [C11, C12, C16, C22, C26, C66], got {len(C)}"
            )
        C_mat = np.array(
            [
                [C[0], C[1], C[2]],
                [C[1], C[3], C[4]],
                [C[2], C[4], C[5]],
            ]
        )
        C_mat = validate_tensor(C, f"Region '{name}' C")
        # Bounds using the orthonormal (Kelvin) strain representation.
        scale = np.diag([1., 1., np.sqrt(2.)])
        eig = np.linalg.eigvalsh(scale @ C_mat @ scale)
        c_L = float(np.sqrt(eig[-1] / rho))
        c_T = float(np.sqrt(eig[0] / (2*rho)))
        entry.update(C=C, C_mat=C_mat, c_L=c_L, c_T=c_T, c_physical_min=c_T)

        if model == "viscoelastic":
            Cv = [float(v) for v in item["Cv"]]
            if len(Cv) != 6:
                raise ValueError(
                    f"Region '{name}': Cv matrix must have 6 entries [Cv11, Cv12, Cv16, Cv22, Cv26, Cv66], got {len(Cv)}"
                )
            Cv_mat = np.array(
                [
                    [Cv[0], Cv[1], Cv[2]],
                    [Cv[1], Cv[3], Cv[4]],
                    [Cv[2], Cv[4], Cv[5]],
                ]
            )
            Cv_mat = validate_tensor(Cv, f"Region '{name}' Cv", positive=False)
            entry.update(Cv=Cv, Cv_mat=Cv_mat)

    return entry


def _normalize_structure(
    mode: str | None = None,
    single_layer_input: dict[str, Any] | None = None,
    layers_input: list[dict[str, Any]] | None = None,
    halfspace_input: dict[str, Any] | None = None,
    bottom_hs_input: dict[str, Any] | None = None,
    top_hs_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize full structure: plates, pristine halfspace, or plate-halfspace systems.

    Supports:
      - "1_layer"             : 1 layer in vacuum
      - "N_layers"            : N layers in vacuum
      - "halfspace"           : Pristine halfspace (Rayleigh surface waves)
      - "plate_on_halfspace"  : Plate on bottom substrate halfspace
      - "plate_in_halfspaces" : Plate embedded between bottom substrate & top superstrate

    Parameters
    ----------
    mode : str, optional
        Structure configuration mode. Defaults to "N_layers".
    single_layer_input : dict, optional
        Single layer definition mapping.
    layers_input : list of dict, optional
        List of layer mappings for multilayer laminate.
    halfspace_input : dict, optional
        Halfspace definition mapping for pristine halfspace or substrate.
    bottom_hs_input : dict, optional
        Bottom substrate halfspace mapping.
    top_hs_input : dict, optional
        Top superstrate halfspace mapping.

    Returns
    -------
    dict
        Comprehensive structure summary dictionary containing all normalized regions,
        bounding coordinates (y_min, y_max), plate thickness, and extremal wave speeds.
    """
    mode = _canonicalize_mode(mode or "N_layers")

    # Case 1: Pristine Halfspace alone (Rayleigh surface waves)
    if mode == "halfspace":
        hs_cfg = halfspace_input or bottom_hs_input
        if hs_cfg is None:
            raise ValueError(
                "structure.mode is 'halfspace', but 'halfspace' (or 'substrate') was not defined in configuration."
            )
        h_hs = float(hs_cfg["thickness"])
        norm_hs = _normalize_region(
            hs_cfg,
            default_name="Halfspace",
            role="Halfspace",
            is_halfspace=True,
            halfspace_type="bottom",
            y_bot=-h_hs,
        )
        absorb_depth = float(hs_cfg.get("absorb_depth", 0.6 * h_hs))
        if not np.isfinite(absorb_depth) or absorb_depth <= 0 or absorb_depth > h_hs:
            raise ValueError(
                f"Halfspace absorb_depth must be in (0, {h_hs}], got {absorb_depth}"
            )
        norm_hs["absorb_depth"] = absorb_depth
        norm_hs["tag"] = 1

        # Compute exact theoretical Rayleigh wave speed for isotropic solid halfspace
        if norm_hs["material_model"] == "isotropic" and not norm_hs.get(
            "is_fluid", False
        ):
            c_R = compute_rayleigh_speed(
                norm_hs["c_T"], c_L=norm_hs["c_L"], nu=norm_hs["nu"]
            )
        else:
            c_R = None
        norm_hs["c_R"] = c_R

        all_regions = [norm_hs]
        norm_plate_layers = []
        H_plate = 0.0
        y_min = -h_hs
        y_max = 0.0
        H_domain = h_hs

        c_T_min = norm_hs["c_T"]
        c_T_max = norm_hs["c_T"]
        c_L_min = norm_hs["c_L"]
        c_L_max = norm_hs["c_L"]
        c_wave_min = norm_hs["c_physical_min"]

        c_T_plate_min = norm_hs["c_T"]
        c_T_plate_max = norm_hs["c_T"]
        c_L_plate_min = norm_hs["c_L"]
        c_L_plate_max = norm_hs["c_L"]

        c_L_plate_top = norm_hs["c_L"]
        c_T_plate_top = norm_hs["c_T"]
        rho_plate_top = norm_hs["rho"]

        return {
            "structure_mode": mode,
            "all_regions": all_regions,
            "plate_layers": norm_plate_layers,
            "bottom_halfspace": norm_hs,
            "top_halfspace": None,
            "has_bottom_halfspace": True,
            "has_top_halfspace": False,
            "H_plate": H_plate,
            "y_min": y_min,
            "y_max": y_max,
            "H_domain": H_domain,
            "c_T_min": c_T_min,
            "c_T_max": c_T_max,
            "c_L_min": c_L_min,
            "c_L_max": c_L_max,
            "c_wave_min": c_wave_min,
            "has_fluid": norm_hs.get("is_fluid", False),
            "c_fluid_min": norm_hs["c_L"] if norm_hs.get("is_fluid", False) else None,
            "c_R": c_R,
            "c_T_plate_min": c_T_plate_min,
            "c_T_plate_max": c_T_plate_max,
            "c_L_plate_min": c_L_plate_min,
            "c_L_plate_max": c_L_plate_max,
            "c_L_plate_top": c_L_plate_top,
            "c_T_plate_top": c_T_plate_top,
            "rho_plate_top": rho_plate_top,
        }

    # Case 2: Plate-based systems (1_layer, N_layers, plate_on_halfspace, plate_in_halfspaces)
    if mode == "1_layer":
        if single_layer_input is not None:
            plate_cfg = [single_layer_input]
        elif layers_input is not None and len(layers_input) > 0:
            plate_cfg = [layers_input[0]]
        else:
            raise ValueError(
                "structure.mode is '1_layer', but 'single_layer' (or 'layers') was not defined in configuration."
            )
    else:
        if layers_input is not None and len(layers_input) > 0:
            plate_cfg = layers_input
        elif single_layer_input is not None:
            plate_cfg = [single_layer_input]
        else:
            raise ValueError(
                f"structure.mode is '{mode}', but 'layers' list was not defined in configuration."
            )

    # Resolve halfspace subdomains based on mode
    if mode in ("1_layer", "N_layers"):
        bot_hs_dict = None
        top_hs_dict = None
    elif mode == "plate_on_halfspace":
        bot_hs_dict = bottom_hs_input or halfspace_input
        if bot_hs_dict is None:
            raise ValueError(
                "structure.mode is 'plate_on_halfspace', but 'substrate' (or 'bottom_halfspace') was not defined in configuration."
            )
        top_hs_dict = None
    elif mode == "plate_in_halfspaces":
        bot_hs_dict = bottom_hs_input or halfspace_input
        if bot_hs_dict is None:
            raise ValueError(
                "structure.mode is 'plate_in_halfspaces', but 'substrate' (or 'bottom_halfspace') was not defined in configuration."
            )
        top_hs_dict = top_hs_input
        if top_hs_dict is None:
            raise ValueError(
                "structure.mode is 'plate_in_halfspaces', but 'superstrate' (or 'top_halfspace') was not defined in configuration."
            )

    # 1. Normalize plate layers (y starts at 0.0)
    norm_plate_layers = []
    y_curr = 0.0
    for idx, lyr in enumerate(plate_cfg):
        role_label = f"Plate L{idx + 1}" if len(plate_cfg) > 1 else "Plate (Single)"
        entry = _normalize_region(
            lyr,
            default_name=f"Layer {idx + 1}",
            role=role_label,
            is_halfspace=False,
            halfspace_type=None,
            y_bot=y_curr,
        )
        y_curr += entry["thickness"]
        norm_plate_layers.append(entry)
    H_plate = y_curr

    # 2. Normalize bottom halfspace (substrate at y <= 0)
    if bot_hs_dict is not None:
        h_sub = float(bot_hs_dict["thickness"])
        norm_bot_hs = _normalize_region(
            bot_hs_dict,
            default_name="Bottom Substrate",
            role="Substrate",
            is_halfspace=True,
            halfspace_type="bottom",
            y_bot=-h_sub,
        )
        absorb_depth = float(bot_hs_dict.get("absorb_depth", 0.75 * h_sub))
        if not np.isfinite(absorb_depth) or absorb_depth <= 0 or absorb_depth > h_sub:
            raise ValueError(
                f"Bottom halfspace absorb_depth must be in (0, {h_sub}], got {absorb_depth}"
            )
        norm_bot_hs["absorb_depth"] = absorb_depth
    else:
        norm_bot_hs = None

    # 3. Normalize top halfspace (superstrate at y >= H_plate)
    if top_hs_dict is not None:
        h_sup = float(top_hs_dict["thickness"])
        norm_top_hs = _normalize_region(
            top_hs_dict,
            default_name="Top Superstrate",
            role="Superstrate",
            is_halfspace=True,
            halfspace_type="top",
            y_bot=H_plate,
        )
        absorb_depth = float(top_hs_dict.get("absorb_depth", 0.75 * h_sup))
        if not np.isfinite(absorb_depth) or absorb_depth <= 0 or absorb_depth > h_sup:
            raise ValueError(
                f"Top halfspace absorb_depth must be in (0, {h_sup}], got {absorb_depth}"
            )
        norm_top_hs["absorb_depth"] = absorb_depth
    else:
        norm_top_hs = None

    # 4. Assemble unified all_regions list from bottom to top
    all_regions = []
    if norm_bot_hs is not None:
        all_regions.append(norm_bot_hs)
    for lyr in norm_plate_layers:
        all_regions.append(lyr)
    if norm_top_hs is not None:
        all_regions.append(norm_top_hs)

    for idx, reg in enumerate(all_regions):
        reg["tag"] = idx + 1

    y_min = all_regions[0]["y_bot"]
    y_max = all_regions[-1]["y_top"]
    H_domain = y_max - y_min

    solid_regions = [r for r in all_regions if not r.get("is_fluid", False)]
    if solid_regions:
        c_T_min = min(r["c_T"] for r in solid_regions)
        c_T_max = max(r["c_T"] for r in solid_regions)
    else:
        c_T_min = min(r["c_T"] for r in all_regions)
        c_T_max = max(r["c_T"] for r in all_regions)
    c_L_min = min(r["c_L"] for r in all_regions)
    c_L_max = max(r["c_L"] for r in all_regions)
    c_wave_min = min(r["c_physical_min"] for r in all_regions)
    has_fluid = any(r.get("is_fluid", False) for r in all_regions)
    c_fluid_min = min(
        (r["c_L"] for r in all_regions if r.get("is_fluid", False)), default=None
    )

    solid_plate_layers = [
        lyr for lyr in norm_plate_layers if not lyr.get("is_fluid", False)
    ]
    if solid_plate_layers:
        c_T_plate_min = min(r["c_T"] for r in solid_plate_layers)
        c_T_plate_max = max(r["c_T"] for r in solid_plate_layers)
    else:
        c_T_plate_min = min(r["c_T"] for r in norm_plate_layers)
        c_T_plate_max = max(r["c_T"] for r in norm_plate_layers)
    c_L_plate_min = min(r["c_L"] for r in norm_plate_layers)
    c_L_plate_max = max(r["c_L"] for r in norm_plate_layers)

    top_plate_lyr = norm_plate_layers[-1]
    c_L_plate_top = top_plate_lyr["c_L"]
    c_T_plate_top = top_plate_lyr["c_T"]
    rho_plate_top = top_plate_lyr["rho"]

    return {
        "structure_mode": mode,
        "all_regions": all_regions,
        "plate_layers": norm_plate_layers,
        "bottom_halfspace": norm_bot_hs,
        "top_halfspace": norm_top_hs,
        "has_bottom_halfspace": norm_bot_hs is not None,
        "has_top_halfspace": norm_top_hs is not None,
        "H_plate": H_plate,
        "y_min": y_min,
        "y_max": y_max,
        "H_domain": H_domain,
        "c_T_min": c_T_min,
        "c_T_max": c_T_max,
        "c_L_min": c_L_min,
        "c_L_max": c_L_max,
        "c_wave_min": c_wave_min,
        "has_fluid": has_fluid,
        "c_fluid_min": c_fluid_min,
        "c_T_plate_min": c_T_plate_min,
        "c_T_plate_max": c_T_plate_max,
        "c_L_plate_min": c_L_plate_min,
        "c_L_plate_max": c_L_plate_max,
        "c_L_plate_top": c_L_plate_top,
        "c_T_plate_top": c_T_plate_top,
        "rho_plate_top": rho_plate_top,
    }


def _format_banner_box(lines: Sequence[str], min_width: int = 60) -> str:
    """Format lines inside a double-bordered Unicode box with pixel-perfect right-wall alignment.

    Parameters
    ----------
    lines : sequence of str
        Text lines to display inside the banner box.
    min_width : int, default=60
        Minimum interior width of the banner box in characters.

    Returns
    -------
    str
        Formatted multi-line box string.
    """
    max_len = max(len(line) for line in lines)
    inner_width = max(max_len + 4, min_width)
    top = "╔" + "═" * inner_width + "╗"
    bot = "╚" + "═" * inner_width + "╝"
    box_lines = [top]
    for line in lines:
        padded = f"  {line}".ljust(inner_width)
        box_lines.append(f"║{padded}║")
    box_lines.append(bot)
    return "\n" + "\n".join(box_lines)


def _print_structure_table(
    summary: dict[str, Any], rank: int = 0, f0: float | None = None
) -> None:
    """Format and print a professional structure summary table on rank 0.

    Parameters
    ----------
    summary : dict
        Structure summary dictionary returned by ``_normalize_structure``.
    rank : int, default=0
        MPI process rank. Output is only printed on rank 0.
    f0 : float, optional
        Center frequency [Hz] of excitation for wavelength calculations.
    """
    if rank != 0:
        return
    regions = summary["all_regions"]
    w_no, w_name, w_th, w_mod, w_rho, w_cl, w_ct, w_role = 5, 27, 12, 15, 11, 11, 11, 14
    cols = [w_no, w_name, w_th, w_mod, w_rho, w_cl, w_ct, w_role]
    top = "┌" + "┬".join("─" * c for c in cols) + "┐"
    mid = "├" + "┼".join("─" * c for c in cols) + "┤"
    bot = "└" + "┴".join("─" * c for c in cols) + "┘"
    log("\n" + top)
    log(
        f"│ {'No.':<{w_no - 2}} │ {'Region / Layer Name':<{w_name - 2}} │ {'Thick [mm]':<{w_th - 2}} │ "
        f"{'Model':<{w_mod - 2}} │ {'ρ [kg/m³]':<{w_rho - 2}} │ {'c_L [m/s]':<{w_cl - 2}} │ "
        f"{'c_T [m/s]':<{w_ct - 2}} │ {'Domain Role':<{w_role - 2}} │"
    )
    log(mid)
    for i, r in enumerate(regions):
        ct_display = (
            f"{r['c_T']:<{w_ct - 2}.1f}"
            if not r.get("is_fluid", False)
            else f"{'— (fluid)':<{w_ct - 2}}"
        )
        log(
            f"│ {i + 1:<{w_no - 2}} │ {r['name'][: w_name - 2]:<{w_name - 2}} │ {r['thickness'] * 1e3:<{w_th - 2}.3f} │ "
            f"{r['material_model'][: w_mod - 2]:<{w_mod - 2}} │ {r['rho']:<{w_rho - 2}.1f} │ {r['c_L']:<{w_cl - 2}.1f} │ "
            f"{ct_display} │ {r['role'][: w_role - 2]:<{w_role - 2}} │"
        )
    log(bot)

    mode_labels = {
        "1_layer": "1 layer (Free plate in vacuum)",
        "N_layers": f"N layers ({len(summary['plate_layers'])} layers in vacuum)",
        "halfspace": "Halfspace (Semi-infinite solid, surface at y = 0)",
        "plate_on_halfspace": f"N layers over a halfspace ({len(summary['plate_layers'])} plate layers on substrate)",
        "plate_in_halfspaces": f"N layers between 2 halfspaces ({len(summary['plate_layers'])} plate layers between substrate & superstrate)",
    }
    mode_str = mode_labels.get(summary["structure_mode"], summary["structure_mode"])
    log(f"  Configuration mode: {mode_str}")
    if summary["structure_mode"] == "halfspace":
        c_R = summary.get("c_R")
        c_r_msg = f", c_R = {c_R:.1f} m/s (Rayleigh surface wave)" if c_R else ""
        h_clean = summary["H_domain"] - summary["bottom_halfspace"]["absorb_depth"]
        h_sponge = summary["bottom_halfspace"]["absorb_depth"]
        f0 = f0 or summary.get("f_center", 1.0e6)
        lambda_ref = (c_R / f0) if c_R else (summary["c_wave_min"] / f0)
        is_fluid_hs = summary["bottom_halfspace"].get("is_fluid", False)

        log(
            f"  Halfspace free surface: y = 0.000 mm "
            f"(total depth: {summary['H_domain'] * 1e3:.1f} mm, y ∈ [{summary['y_min'] * 1e3:.1f}, 0.0] mm)"
        )
        if is_fluid_hs:
            log(
                f"  Fluid halfspace acoustic sound speed: c_f = {summary['c_L_min']:.1f} m/s"
            )
            log(
                f"  Acoustic wavelength:    λ_f ≈ {lambda_ref * 1e3:.2f} mm (at f₀ = {f0 / 1e3:.0f} kHz)"
            )
        else:
            log(
                f"  Wave speeds: c_L = {summary['c_L_min']:.1f} m/s, "
                f"c_T = {summary['c_T_min']:.1f} m/s{c_r_msg}"
            )
            log(
                f"  Rayleigh wavelength:    λ_R ≈ {lambda_ref * 1e3:.2f} mm (at f₀ = {f0 / 1e3:.0f} kHz)"
            )
        wav_lbl = "λ_f" if is_fluid_hs else "λ_R"
        log(
            f"  Clean halfspace depth:  {h_clean * 1e3:.1f} mm ({h_clean / lambda_ref:.1f} {wav_lbl})"
        )
        log(
            f"  Bottom absorbing sponge:{h_sponge * 1e3:.1f} mm ({h_sponge / lambda_ref:.1f} {wav_lbl}), "
            f"y ∈ [{summary['y_min'] * 1e3:.1f}, {(summary['y_min'] + h_sponge) * 1e3:.1f}] mm"
        )
        log("  Sensors and excitation surface: y = 0.000 mm\n")
    else:
        log(
            f"  Plate thickness H_plate = {summary['H_plate'] * 1e3:.3f} mm (y ∈ [0, {summary['H_plate'] * 1e3:.3f}] mm)"
        )
        log(
            f"  Computational domain y  = [{summary['y_min'] * 1e3:.3f}, {summary['y_max'] * 1e3:.3f}] mm "
            f"(total height: {summary['H_domain'] * 1e3:.3f} mm)"
        )
        solid_layers = [
            lyr for lyr in summary["plate_layers"] if not lyr.get("is_fluid", False)
        ]
        fluid_layers = [
            lyr for lyr in summary["plate_layers"] if lyr.get("is_fluid", False)
        ]
        if solid_layers:
            log(
                f"  Plate wave speeds: c_L = [{summary['c_L_plate_min']:.1f}, {summary['c_L_plate_max']:.1f}] m/s, "
                f"c_T = [{summary['c_T_plate_min']:.1f}, {summary['c_T_plate_max']:.1f}] m/s"
            )
        if fluid_layers:
            c_fl_min = min(r["c_L"] for r in fluid_layers)
            c_fl_max = max(r["c_L"] for r in fluid_layers)
            log(
                f"  Plate fluid acoustic speed: c_f = [{c_fl_min:.1f}, {c_fl_max:.1f}] m/s"
            )
        if summary["has_bottom_halfspace"]:
            hs_b = summary["bottom_halfspace"]
            type_str = (
                f"fluid (c_f = {hs_b['c_L']:.1f} m/s)"
                if hs_b.get("is_fluid", False)
                else f"solid (c_L = {hs_b['c_L']:.1f}, c_T = {hs_b['c_T']:.1f} m/s)"
            )
            log(
                f"  Bottom substrate:        {hs_b['name']} [{type_str}], sponge y ∈ [{summary['y_min'] * 1e3:.2f}, "
                f"{(summary['y_min'] + hs_b['absorb_depth']) * 1e3:.2f}] mm "
                f"(depth: {hs_b['absorb_depth'] * 1e3:.2f} mm)"
            )
        if summary["has_top_halfspace"]:
            hs_t = summary["top_halfspace"]
            type_str = (
                f"fluid (c_f = {hs_t['c_L']:.1f} m/s)"
                if hs_t.get("is_fluid", False)
                else f"solid (c_L = {hs_t['c_L']:.1f}, c_T = {hs_t['c_T']:.1f} m/s)"
            )
            log(
                f"  Top superstrate:         {hs_t['name']} [{type_str}], sponge y ∈ [{(summary['y_max'] - hs_t['absorb_depth']) * 1e3:.2f}, "
                f"{summary['y_max'] * 1e3:.2f}] mm "
                f"(depth: {hs_t['absorb_depth'] * 1e3:.2f} mm)"
            )
        top_reg = summary["all_regions"][-1]
        top_type = "fluid" if top_reg.get("is_fluid", False) else "solid"
        if summary["has_top_halfspace"]:
            loc_str = (
                f"plate-superstrate interface at y = {summary['H_plate'] * 1e3:.3f} mm"
            )
        else:
            loc_str = f"top surface of '{top_reg['name']}' [{top_type}] at y = {summary['y_max'] * 1e3:.3f} mm"
        log(f"  Sensors and excitation boundary: {loc_str}\n")


def preflight_estimate(cfg: dict[str, Any], rank: int = 0) -> dict[str, Any]:
    """Compute lightweight preflight estimates for mesh sizing, DOFs, time steps, and memory.

    Parameters
    ----------
    cfg : dict
        Fully resolved configuration dictionary.
    rank : int, default=0
        MPI rank. Output is only logged on rank 0.

    Returns
    -------
    dict
        Preflight estimation dictionary containing elements, DOFs, time steps, and RAM footprints.
    """
    mode = cfg.get("structure_mode", "N_layers")
    struct_summary = _normalize_structure(
        mode=mode,
        single_layer_input=cfg.get("single_layer"),
        layers_input=cfg.get("layers"),
        halfspace_input=cfg.get("halfspace"),
        bottom_hs_input=cfg.get("bottom_halfspace"),
        top_hs_input=cfg.get("top_halfspace"),
    )
    all_regions = struct_summary["all_regions"]
    c_wave_min = struct_summary["c_wave_min"]
    c_L_max = struct_summary["c_L_max"]
    f_center = float(cfg["f_center"])
    mesh_frequency = float(cfg.get("max_frequency") or 3.0 * f_center)
    lambda_min = c_wave_min / mesh_frequency
    elems_per_wavelength = int(cfg.get("elems_per_wavelength", 10))
    h_target = lambda_min / elems_per_wavelength

    plate_length = 2.0 * float(cfg["absorb_length"]) + float(cfg["sensing_length"])
    nx = max(int(np.ceil(plate_length / h_target)), int(cfg.get("min_elems_x", 100)))
    ny_per_region = [
        max(
            int(np.ceil(reg["thickness"] / h_target)),
            reg.get("min_elems_y") or int(cfg.get("min_elems_y", 4)),
        )
        for reg in all_regions
    ]
    total_ny = sum(ny_per_region)
    total_elements = nx * total_ny
    degree = int(cfg.get("degree", 2))
    dim = 2
    # Q_p quadrilateral element global DOF estimate:
    ndofs_est = dim * (degree * nx + 1) * (degree * total_ny + 1)

    speed_for_T = struct_summary.get("c_R") or c_wave_min
    time_factor = float(cfg.get("time_factor", 1.2))
    T_total = float(cfg.get("duration") or (float(cfg["sensing_length"]) / speed_for_T * time_factor))

    h_min = min(
        plate_length / nx,
        min(r["thickness"] / ny_per_region[i] for i, r in enumerate(all_regions)),
    )
    cfl = float(cfg.get("CFL", 0.5))
    stype = str(cfg.get("solver_type", "explicit")).lower()

    # Heuristic estimate of critical time step prior to matrix assembly:
    dt_heuristic = cfl * h_min / (degree * c_L_max)
    dt_est = float(cfg.get("dt") or dt_heuristic)
    dt_val, nsteps_est = time_grid(T_total, dt_est)

    # Memory estimates:
    vec_mem_mb = (ndofs_est * 8) / (1024 ** 2)
    # Sparse matrix CSR: nonzeros per row for Q_degree in 2D is approximately (2*degree+1)^2 * dim
    nnz_per_row = ((2 * degree + 1) ** 2) * dim
    mat_mem_mb = (ndofs_est * nnz_per_row * (8 + 4)) / (1024 ** 2)
    total_mem_est_mb = (vec_mem_mb * 10) + (mat_mem_mb * 3)

    estimates = {
        "mesh_frequency": mesh_frequency,
        "lambda_min": lambda_min,
        "h_target": h_target,
        "h_min": h_min,
        "nx": nx,
        "ny_per_region": ny_per_region,
        "total_elements": total_elements,
        "ndofs_est": ndofs_est,
        "duration": T_total,
        "dt_est": dt_val,
        "nsteps_est": nsteps_est,
        "vec_mem_mb": vec_mem_mb,
        "mat_mem_mb": mat_mem_mb,
        "total_mem_est_mb": total_mem_est_mb,
    }

    if rank == 0:
        mem_str = (
            f"{total_mem_est_mb / 1024:.2f} GB"
            if total_mem_est_mb >= 1024
            else f"{total_mem_est_mb:.1f} MB"
        )
        lines = [
            f"PREFLIGHT NUMERICAL ESTIMATE ({stype.upper()})",
            "─" * 58,
            f"Mesh frequency:        {mesh_frequency / 1e6:.2f} MHz (λ_min = {lambda_min * 1e3:.3f} mm)",
            f"Target element size h: {h_target * 1e3:.3f} mm (actual min h = {h_min * 1e3:.3f} mm)",
            f"Grid resolution:       {nx} × {total_ny} Q{degree} elements ({total_elements:,} total)",
            f"Estimated global DOFs: {ndofs_est:,}",
            f"Simulation duration:   {T_total * 1e6:.1f} μs",
            f"Estimated step limit:  ~{dt_val * 1e9:.2f} ns (heuristic CFL={cfl})",
            f"Estimated step count:  ~{nsteps_est:,} steps",
            f"Estimated memory RAM:  ~{mem_str} (single state vector: {vec_mem_mb:.1f} MB)",
        ]
        log(_format_banner_box(lines))
    return estimates


# ==============================================================================
# SECTION 3: Signal Processing & 2D FFT Dispersion Analysis
# ==============================================================================


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
        if not np.isfinite(axis).all() or np.any(delta <= 0) or not np.allclose(delta, delta[0], rtol=1e-7, atol=0):
            raise ValueError("Postprocessing requires uniform increasing coordinates")
    for values in (sensor_data_y, sensor_data_x):
        if values is not None and (values.shape != (len(times), len(sensor_x)) or not np.isfinite(values).all()):
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
    df, dk = freqs_fk_MHz[1]-freqs_fk_MHz[0], k_fk_radmm[1]-k_fk_radmm[0]
    extent_disp = [selected_k[0]-dk/2, selected_k[-1]+dk/2,
                   selected_f[0]-df/2, selected_f[-1]+df/2]

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


# ==============================================================================
# SECTION 4: Finite Element Wave Propagation Simulation
# ==============================================================================


@mpi_fail_fast
def run_simulation(
    args: argparse.Namespace | None = None,
    cfg: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Execute guided wave finite element simulation using implicit or explicit time integration.

    Parameters
    ----------
    args : argparse.Namespace, optional
        Command-line argument namespace containing runtime overrides.
    cfg : dict, optional
        Simulation configuration dictionary. If None, loaded from ``args.config``.

    Returns
    -------
    tuple of (str, str)
        Tuple containing ``(sensor_data_path, output_dir)``:
        - ``sensor_data_path``: Path to saved ``sensor_data.npz``.
        - ``output_dir``: Directory containing simulation outputs and figures.

    Raises
    ------
    ValueError
        If no configuration is provided or an unsupported solver type is specified.
    RuntimeError
        If numerical instability occurs (e.g. non-positive lumped mass entries).
    """
    if cfg is None:
        if args is None or not getattr(args, "config", None):
            raise ValueError(
                "A configuration file (-c/--config) or config dictionary must be provided to run_simulation."
            )
        cfg = parse_and_resolve_config(config_path=args.config, args=args)
    else:
        cfg = parse_and_resolve_config(args=args, custom_cfg=cfg)

    stype = (getattr(args, "solver", None) if args is not None else None) or cfg.get(
        "solver_type", "explicit"
    )
    stype = str(stype).lower().strip()
    if stype not in ("implicit", "explicit"):
        raise ValueError(f"Unknown solver '{stype}'. Must be 'implicit' or 'explicit'.")

    # Set output directory based on solver
    out_dir = (
        (getattr(args, "output", None) if args is not None else None)
        or cfg.get("output_dir")
        or f"results_guided_waves_{stype}"
    )
    cfg["output_dir"] = out_dir

    sensing_length = float(cfg["sensing_length"])
    absorb_length = float(cfg["absorb_length"])
    f_center = float(cfg["f_center"])
    source_x = float(cfg["source_x"])
    source_fx = float(cfg.get("source_fx", 0.0))
    source_fy = float(cfg.get("source_fy", 1.0))
    traction_max = float(cfg.get("traction_max", 1.0e6))
    excitation_width = float(cfg.get("excitation_width", 2.0e-3))
    elems_per_wavelength = int(cfg.get("elems_per_wavelength", 10))
    min_elems_x = int(cfg.get("min_elems_x", 100))
    min_elems_y = int(cfg.get("min_elems_y", 4))
    degree = int(cfg.get("degree", 2))
    CFL = float(cfg.get("CFL", 0.5))
    time_factor = float(cfg.get("time_factor", 1.2))
    beta_val = float(cfg.get("beta_val", 0.25))
    gamma_val = float(cfg.get("gamma_val", 0.5))
    n_sensors = int(cfg.get("n_sensors", 301))
    sensor_offset = float(cfg.get("sensor_offset", 0.05))
    sensor_pitch = float(cfg.get("sensor_pitch", 0.5e-3))
    write_vtx = bool(cfg.get("write_vtx", False))
    target_snapshots = int(cfg.get("target_snapshots", 200))
    f_min_disp_MHz = float(cfg.get("f_min_disp_MHz", 0.0))
    f_max_disp_MHz = float(cfg.get("f_max_disp_MHz", 2.0))
    k_min_disp = float(cfg.get("k_min_disp", 0.0))
    k_max_disp = float(cfg.get("k_max_disp", 3.0))

    # Adaptive Rayleigh damping defaults
    absorb_alpha = cfg.get("absorb_alpha")
    absorb_beta = cfg.get("absorb_beta")
    if absorb_alpha is None:
        absorb_alpha = 1e7
    else:
        absorb_alpha = float(absorb_alpha)
    if absorb_beta is None:
        absorb_beta = 1e-8
    else:
        absorb_beta = float(absorb_beta)

    wall_t0 = pytime.perf_counter()

    # Total plate length = left absorber + sensing region + right absorber
    plate_length = absorb_length + sensing_length + absorb_length

    # Multilayer / Halfspace structure normalization & wave speed analysis
    mode_arg = getattr(args, "structure_mode", None) if args is not None else None
    struct_summary = _normalize_structure(
        mode=mode_arg or cfg.get("structure_mode", "N_layers"),
        single_layer_input=cfg.get("single_layer"),
        layers_input=cfg.get("layers"),
        halfspace_input=cfg.get("halfspace"),
        bottom_hs_input=cfg.get("bottom_halfspace"),
        top_hs_input=cfg.get("top_halfspace"),
    )
    all_regions = struct_summary["all_regions"]
    norm_plate_layers = struct_summary["plate_layers"]
    n_regions = len(all_regions)
    n_plate_layers = len(norm_plate_layers)
    H_plate = struct_summary["H_plate"]
    y_min = struct_summary["y_min"]
    y_max = struct_summary["y_max"]
    H_domain = struct_summary["H_domain"]
    has_bottom_halfspace = struct_summary["has_bottom_halfspace"]
    has_top_halfspace = struct_summary["has_top_halfspace"]
    active_structure_mode = struct_summary["structure_mode"]

    c_T_min = struct_summary["c_T_min"]
    c_L_max = struct_summary["c_L_max"]
    c_wave_min = struct_summary.get("c_wave_min", c_T_min)
    c_L_top = struct_summary["c_L_plate_top"]
    c_T_top = struct_summary["c_T_plate_top"]
    rho_top = struct_summary["rho_plate_top"]
    has_fluid = struct_summary.get("has_fluid", False)
    c_fluid_min = struct_summary.get("c_fluid_min")
    any_viscoelastic = any(r["material_model"] == "viscoelastic" for r in all_regions)

    if rank == 0:
        cfg_source_str = str(cfg.get("_config_source", "Defaults"))
        banner_lines = [
            f"Guided Waves FEM Solver: {stype.upper()}",
            f"MPI ranks: {nprocs}",
            f"Mode: {active_structure_mode}",
            f"Config: {cfg_source_str}",
        ]
        log(_format_banner_box(banner_lines))
        _print_structure_table(struct_summary, rank=rank, f0=f_center)
        if has_fluid:
            log("  Fluid model: regularized elasticity with continuous tangential displacement; "
                "not a mixed inviscid acoustic formulation.")

    # Preflight resource and discretization estimation
    preflight_estimate(cfg, rank=rank)

    # Mesh sizing based on shortest wavelength across all active regions
    mesh_frequency = float(cfg.get("max_frequency") or 3*f_center)
    lambda_min = c_wave_min / mesh_frequency
    h_target = lambda_min / elems_per_wavelength
    nx = max(int(np.ceil(plate_length / h_target)), min_elems_x)

    ny_per_region = [
        max(
            int(np.ceil(reg["thickness"] / h_target)),
            reg.get("min_elems_y") or min_elems_y,
        )
        for reg in all_regions
    ]
    ny = sum(ny_per_region)

    if rank == 0:
        log(f"  Shortest wavelength λ_min = {lambda_min * 1e3:.2f} mm")
        log(f"  Target element size h     = {h_target * 1e3:.3f} mm")
        log(
            f"  Mesh: {nx} × {ny} elements ({nx * ny} total), "
            f"ny_per_region = {ny_per_region}"
        )

    h_actual = min(
        plate_length / nx,
        min(r["thickness"] / ny_per_region[i] for i, r in enumerate(all_regions)),
    )
    speed_for_T = struct_summary.get("c_R") or c_wave_min
    T_total = float(cfg.get("duration") or sensing_length / speed_for_T * time_factor)

    # Sensor positions along top surface (or interface if top halfspace is present)
    sensor_x_arr = source_x + sensor_offset + np.arange(n_sensors) * sensor_pitch

    top_reg = all_regions[-1]
    is_top_fluid = top_reg.get("is_fluid", False)
    top_fluid_label = "fluid" if is_top_fluid else "solid"

    cfg_sensor_y = cfg.get("sensor_y", None)
    if cfg_sensor_y is not None:
        sensor_y_coord = float(cfg_sensor_y)
    elif has_top_halfspace:
        sensor_y_coord = H_plate
    else:
        # Default: strictly on the top surface of the topmost layer (even if fluid)
        sensor_y_coord = y_max

    if not np.isfinite(sensor_y_coord) or not y_min <= sensor_y_coord <= y_max:
        raise ValueError("sensor_y must lie inside the domain")

    if rank == 0:
        if active_structure_mode == "halfspace":
            interface_desc = "halfspace free surface (y = 0.00 mm)"
        elif has_top_halfspace:
            interface_desc = f"plate-superstrate interface (y = {H_plate * 1e3:.2f} mm)"
        else:
            interface_desc = f"top surface of '{top_reg['name']}' [{top_fluid_label}] (y = {sensor_y_coord * 1e3:.2f} mm)"
        log(
            f"  Sensor array: {n_sensors} sensors on {interface_desc}, "
            f"x = [{sensor_x_arr[0] * 1e3:.1f}, {sensor_x_arr[-1] * 1e3:.1f}] mm, "
            f"pitch = {sensor_pitch * 1e3:.1f} mm"
        )

    # Create conforming mesh spanning [0, y_min] to [plate_length, y_max]
    domain = create_rectangle(
        comm,
        [np.array([0.0, y_min]), np.array([plate_length, y_max])],
        [nx, ny],
        cell_type=CellType.quadrilateral,
    )
    dim = domain.topology.dim
    tdim = dim

    # In-place coordinate stretching: strictly align horizontal element lines with all region interfaces
    thicknesses = [r["thickness"] for r in all_regions]
    frac_grid = np.concatenate([[0.0], np.cumsum(ny_per_region) / ny])
    geom_interfaces = np.concatenate([[y_min], y_min + np.cumsum(thicknesses)])

    y_frac = np.clip((domain.geometry.x[:, 1] - y_min) / H_domain, 0.0, 1.0)
    domain.geometry.x[:, 1] = np.interp(y_frac, frac_grid, geom_interfaces)

    # Tag subdomain cells for each region
    tol_layer = min(thicknesses) * 1e-4
    num_local_cells = domain.topology.index_map(tdim).size_local

    all_cell_indices = []
    all_cell_values = []
    for r_idx, reg in enumerate(all_regions):
        tag_val = r_idx + 1
        y_bot = reg["y_bot"]
        y_top = reg["y_top"]
        cells = locate_entities(
            domain,
            tdim,
            lambda x_c, yb=y_bot, yt=y_top: (
                (x_c[1] >= yb - tol_layer) & (x_c[1] <= yt + tol_layer)
            ),
        )
        cells_loc = cells[cells < num_local_cells]
        all_cell_indices.append(cells_loc)
        all_cell_values.append(np.full(len(cells_loc), tag_val, dtype=np.int32))

    all_cell_indices = np.concatenate(all_cell_indices).astype(np.int32)
    all_cell_values = np.concatenate(all_cell_values).astype(np.int32)
    sort_idx = np.argsort(all_cell_indices)
    cell_tags = meshtags(
        domain, tdim, all_cell_indices[sort_idx], all_cell_values[sort_idx]
    )
    dx_layers = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)

    # Function space
    V = fem.functionspace(domain, ("Q", degree, (dim,)))
    u_ = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)

    ndofs_local = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    ndofs_global = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    if rank == 0:
        log(f"  DOFs: {ndofs_global} global ({ndofs_local} local per rank)")

    # Excitation facet tagging:
    # If top halfspace is present, excitation is on the internal interface at y = H_plate (dS).
    # If top halfspace is absent, excitation is on the exterior top boundary at y = H_plate (ds).
    tol_geom = min(thicknesses) * 1e-4
    fdim = dim - 1

    # Surface traction excitation formulation
    x = ufl.SpatialCoordinate(domain)
    traction_amplitude = fem.Constant(domain, default_scalar_type(0.0))
    excite_profile = ufl.exp(
        -((x[0] - source_x) ** 2) / (2.0 * (excitation_width / 3.0) ** 2)
    )
    norm_dir = np.sqrt(source_fx**2 + source_fy**2)
    if norm_dir == 0:
        raise ValueError("source_fx and source_fy cannot both be zero.")
    exc_dir = ufl.as_vector([source_fx / norm_dir, source_fy / norm_dir])
    traction_vector = traction_amplitude * excite_profile * exc_dir

    if has_top_halfspace:
        domain.topology.create_connectivity(fdim, tdim)
        domain.topology.create_connectivity(tdim, fdim)
        interface_facets = locate_entities(
            domain, fdim, lambda x_c: np.isclose(x_c[1], H_plate, atol=tol_geom)
        )
        interface_facet_tags = meshtags(
            domain,
            fdim,
            interface_facets,
            np.full(len(interface_facets), 1, dtype=np.int32),
        )
        dS = ufl.Measure("dS", domain=domain, subdomain_data=interface_facet_tags)
        F_ext_form_expr = ufl.dot(traction_vector, u_("+")) * dS(1)
    else:
        top_facets = locate_entities_boundary(
            domain, fdim, lambda x_c: np.isclose(x_c[1], y_max, atol=tol_geom)
        )
        top_facet_tags = meshtags(
            domain,
            fdim,
            top_facets,
            np.full(len(top_facets), 1, dtype=np.int32),
        )
        ds = ufl.Measure("ds", domain=domain, subdomain_data=top_facet_tags)
        F_ext_form_expr = ufl.dot(traction_vector, u_) * ds(1)

    if rank == 0:
        if has_top_halfspace:
            load_desc = f"plate-superstrate interface (y = {H_plate * 1e3:.2f} mm)"
        else:
            load_desc = (
                f"top surface of top layer '{top_reg['name']}' [{top_fluid_label}] "
                f"(y = {y_max * 1e3:.2f} mm)"
            )
        log(
            f"  Excitation source: applied on {load_desc} (x = {source_x * 1e3:.1f} mm)"
        )

    def epsilon(w):
        return ufl.sym(ufl.grad(w))

    def _voigt_matrix(c11, c12, c16, c22, c26, c66):
        return ufl.as_tensor(
            [
                [c11, c12, c16],
                [c12, c22, c26],
                [c16, c26, c66],
            ]
        )

    def _sigma_from_C(w, C_mat):
        e = epsilon(w)
        e_voigt = ufl.as_vector([e[0, 0], e[1, 1], 2.0 * e[0, 1]])
        s_voigt = C_mat * e_voigt
        return ufl.as_tensor([[s_voigt[0], s_voigt[2]], [s_voigt[2], s_voigt[1]]])

    rho_c_list = []
    sigma_list = []
    sigma_visco_list = []

    for reg in all_regions:
        rho_c_list.append(fem.Constant(domain, default_scalar_type(reg["rho"])))
        model = reg["material_model"]
        if model in ("isotropic", "fluid"):
            lmbda_c = fem.Constant(domain, default_scalar_type(reg["lmbda"]))
            mu_c = fem.Constant(domain, default_scalar_type(reg["mu"]))

            def make_iso_sigma(lmbda_val, mu_val):
                return lambda w: (
                    lmbda_val * ufl.tr(epsilon(w)) * ufl.Identity(dim)
                    + 2.0 * mu_val * epsilon(w)
                )

            sigma_list.append(make_iso_sigma(lmbda_c, mu_c))
            sigma_visco_list.append(None)
        else:
            C = reg["C"]
            C_c = _voigt_matrix(
                fem.Constant(domain, default_scalar_type(C[0])),
                fem.Constant(domain, default_scalar_type(C[1])),
                fem.Constant(domain, default_scalar_type(C[2])),
                fem.Constant(domain, default_scalar_type(C[3])),
                fem.Constant(domain, default_scalar_type(C[4])),
                fem.Constant(domain, default_scalar_type(C[5])),
            )

            def make_aniso_sigma(C_tensor):
                return lambda w: _sigma_from_C(w, C_tensor)

            sigma_list.append(make_aniso_sigma(C_c))

            if model == "viscoelastic":
                Cv = reg["Cv"]
                Cv_c = _voigt_matrix(
                    fem.Constant(domain, default_scalar_type(Cv[0])),
                    fem.Constant(domain, default_scalar_type(Cv[1])),
                    fem.Constant(domain, default_scalar_type(Cv[2])),
                    fem.Constant(domain, default_scalar_type(Cv[3])),
                    fem.Constant(domain, default_scalar_type(Cv[4])),
                    fem.Constant(domain, default_scalar_type(Cv[5])),
                )

                def make_visco_sigma(Cv_tensor):
                    return lambda w_vel: _sigma_from_C(w_vel, Cv_tensor)

                sigma_visco_list.append(make_visco_sigma(Cv_c))
            else:
                sigma_visco_list.append(None)

    # 2D Absorbing layers (Rayleigh damping sponge)
    # Left and right sponges:
    x_right_start = plate_length - absorb_length
    absorb_right = ufl.conditional(
        ufl.gt(x[0], x_right_start),
        ((x[0] - x_right_start) / absorb_length) ** 5,
        0.0,
    )
    absorb_left = ufl.conditional(
        ufl.lt(x[0], absorb_length),
        ((absorb_length - x[0]) / absorb_length) ** 5,
        0.0,
    )
    absorb_indicator = absorb_left + absorb_right

    # Vertical sponges for halfspaces (bottom substrate and top superstrate):
    if has_bottom_halfspace:
        L_sponge_sub = struct_summary["bottom_halfspace"]["absorb_depth"]
        y_sub_sponge_end = y_min + L_sponge_sub
        absorb_sub = ufl.conditional(
            ufl.lt(x[1], y_sub_sponge_end),
            ((y_sub_sponge_end - x[1]) / L_sponge_sub) ** 5,
            0.0,
        )
        absorb_indicator = absorb_indicator + absorb_sub

    if has_top_halfspace:
        L_sponge_sup = struct_summary["top_halfspace"]["absorb_depth"]
        y_sup_sponge_start = y_max - L_sponge_sup
        absorb_sup = ufl.conditional(
            ufl.gt(x[1], y_sup_sponge_start),
            ((x[1] - y_sup_sponge_start) / L_sponge_sup) ** 5,
            0.0,
        )
        absorb_indicator = absorb_indicator + absorb_sup

    alpha_absorb = fem.Constant(domain, default_scalar_type(absorb_alpha))
    beta_absorb = fem.Constant(domain, default_scalar_type(absorb_beta))

    # Unified bilinear/linear weak forms across subdomains
    def mass(w1, w2):
        return sum(
            rho_c_list[idx] * ufl.dot(w1, w2) * dx_layers(idx + 1)
            for idx in range(n_regions)
        )

    def stiffness(w1, w2):
        return sum(
            ufl.inner(sigma_list[idx](w1), epsilon(w2)) * dx_layers(idx + 1)
            for idx in range(n_regions)
        )

    def damping(w1, w2):
        return sum(
            (
                alpha_absorb * absorb_indicator * rho_c_list[idx] * ufl.dot(w1, w2)
                + beta_absorb
                * absorb_indicator
                * ufl.inner(sigma_list[idx](w1), epsilon(w2))
            )
            * dx_layers(idx + 1)
            for idx in range(n_regions)
        )

    def damping_visco(w_vel, w2):
        terms = [
            ufl.inner(sigma_visco_list[idx](w_vel), epsilon(w2)) * dx_layers(idx + 1)
            for idx in range(n_regions)
            if sigma_visco_list[idx] is not None
        ]
        return sum(terms) if len(terms) > 0 else None

    t0_ricker = 1.5 / f_center

    def ricker(t, f0, t0):
        tau = t - t0
        a_param = (np.pi * f0 * tau) ** 2
        return (1.0 - 2.0 * a_param) * np.exp(-a_param)

    probes = PointSensors(domain, np.column_stack([
        sensor_x_arr, np.full(n_sensors, sensor_y_coord), np.zeros(n_sensors)]))
    n_owned = len(probes.indices)

    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    comm.Barrier()

    # ==========================================================================
    # BRANCH A: IMPLICIT NEWMARK-β SOLVER
    # ==========================================================================
    if stype == "implicit":
        dt_val, Nsteps = time_grid(T_total, float(cfg.get("dt") or CFL*h_actual/c_L_max))
        if cfg["run_postprocess"] and Nsteps < 31:
            raise ValueError("Postprocessing requires at least 32 time samples; reduce dt or disable it")

        if rank == 0:
            log(f"  Time step Δt   = {dt_val * 1e9:.2f} ns")
            log(f"  Total time T   = {T_total * 1e6:.1f} μs")
            log(f"  Number of steps: {Nsteps}")

        # State fields
        u = fem.Function(V, name="Displacement")
        u_old = fem.Function(V, name="Displacement_old")
        v_old = fem.Function(V, name="Velocity_old")
        a_old = fem.Function(V, name="Acceleration_old")
        v_new = fem.Function(V, name="Velocity_new")
        a_new = fem.Function(V, name="Acceleration_new")

        beta = fem.Constant(domain, default_scalar_type(beta_val))
        gamma = fem.Constant(domain, default_scalar_type(gamma_val))
        dt = fem.Constant(domain, default_scalar_type(dt_val))

        a_newmark = 1.0 / beta / dt**2 * (u - u_old - dt * v_old) + a_old * (
            1.0 - 1.0 / (2.0 * beta)
        )
        v_newmark = v_old + dt * ((1.0 - gamma) * a_old + gamma * a_newmark)

        Residual = (
            mass(a_newmark, u_)
            + damping(v_newmark, u_)
            + stiffness(u, u_)
            - F_ext_form_expr
        )
        dv_term = damping_visco(v_newmark, u_)
        if dv_term is not None:
            Residual += dv_term

        Residual_du = ufl.replace(Residual, {u: du})
        a_form = ufl.lhs(Residual_du)
        L_form = ufl.rhs(Residual_du)

        t_setup_start = pytime.perf_counter()
        a_compiled = fem.form(a_form)
        L_compiled = fem.form(L_form)

        A = fem.petsc.assemble_matrix(a_compiled, bcs=[])
        A.assemble()
        b = fem.petsc.create_vector(V)

        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        use_iterative = ndofs_global >= 500_000

        if use_iterative:
            solver.setType(PETSc.KSP.Type.CG)
            pc = solver.getPC()
            pc.setType(PETSc.PC.Type.GAMG)
            solver.setTolerances(rtol=1e-8, atol=1e-14, max_it=2000)
            solver_desc = f"CG + GAMG (iterative, {ndofs_global} DOFs)"
        else:
            solver.setType(PETSc.KSP.Type.PREONLY)
            pc = solver.getPC()
            if nprocs > 1:
                pc.setType(PETSc.PC.Type.LU)
                try:
                    pc.setFactorSolverType("mumps")
                    solver_desc = f"MUMPS LU (parallel direct, {ndofs_global} DOFs)"
                except (PETSc.Error, RuntimeError):
                    pc.setType(PETSc.PC.Type.BJACOBI)
                    solver.setType(PETSc.KSP.Type.GMRES)
                    solver.setTolerances(rtol=1e-10, atol=1e-14, max_it=2000)
                    solver_desc = f"GMRES + BJacobi (parallel, {ndofs_global} DOFs)"
            else:
                pc.setType(PETSc.PC.Type.LU)
                solver_desc = f"LU (serial direct, {ndofs_global} DOFs)"

        solver.setErrorIfNotConverged(True)
        try:
            solver.setUp()
        except PETSc.Error:
            if nprocs == 1 or use_iterative:
                raise
            pc.setType(PETSc.PC.Type.BJACOBI)
            solver.setType(PETSc.KSP.Type.GMRES)
            solver.setTolerances(rtol=1e-10, atol=1e-14, max_it=2000)
            solver.setUp()
            solver_desc = "GMRES + BJacobi (MUMPS unavailable)"
        t_setup_end = pytime.perf_counter()

        if rank == 0:
            log(f"  Solver: {solver_desc}")
            log(
                f"  Matrix assembled & factorised in {t_setup_end - t_setup_start:.2f} s"
            )

        nm_c1 = 1.0 / (beta_val * dt_val**2)
        nm_c2 = 1.0 / (beta_val * dt_val)
        nm_c3 = 1.0 - 1.0 / (2.0 * beta_val)
        nm_c4 = dt_val * (1.0 - gamma_val)
        nm_c5 = dt_val * gamma_val

        sensor_data_x = np.zeros((Nsteps + 1, n_owned))
        sensor_data_y = np.zeros((Nsteps + 1, n_owned))
        times = np.zeros(Nsteps + 1)
        t_current = 0.0

        if write_vtx:
            vtx = io.VTXWriter(comm, f"{out_dir}/guided_wave.bp", [u], engine="BP4")
        save_freq = max(Nsteps // target_snapshots, 1)

        time_assemble = 0.0
        time_solve = 0.0
        time_update = 0.0
        time_io = 0.0
        u_petsc = u.x.petsc_vec
        report_interval = max(Nsteps // 20, 1)

        if rank == 0:
            log(f"\n{'═' * 58}")
            log(f"  Starting IMPLICIT time-stepping  ({Nsteps} steps)")
            log(f"{'═' * 58}")

        t_loop_start = pytime.perf_counter()

        for i in range(Nsteps):
            t_current = (i + 1) * dt_val
            times[i + 1] = t_current

            traction_amplitude.value = traction_max * ricker(
                t_current, f_center, t0_ricker
            )

            t0 = pytime.perf_counter()
            with b.localForm() as b_local:
                b_local.set(0.0)
            fem.petsc.assemble_vector(b, L_compiled)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            t1 = pytime.perf_counter()
            time_assemble += t1 - t0

            solver.solve(b, u_petsc)
            u.x.scatter_forward()
            t2 = pytime.perf_counter()
            time_solve += t2 - t1

            a_new.x.array[:] = (
                nm_c1 * (u.x.array - u_old.x.array)
                - nm_c2 * v_old.x.array
                + nm_c3 * a_old.x.array
            )
            v_new.x.array[:] = (
                v_old.x.array + nm_c4 * a_old.x.array + nm_c5 * a_new.x.array
            )

            u_old.x.array[:] = u.x.array
            v_old.x.array[:] = v_new.x.array
            a_old.x.array[:] = a_new.x.array
            t3 = pytime.perf_counter()
            time_update += t3 - t2

            if n_owned > 0:
                values = probes.sample(u)
                sensor_data_x[i + 1] = values[:, 0]
                sensor_data_y[i + 1] = values[:, 1]

            if write_vtx and i % save_freq == 0:
                t4 = pytime.perf_counter()
                vtx.write(t_current)
                time_io += pytime.perf_counter() - t4

            if rank == 0 and (i + 1) % report_interval == 0:
                pct = 100.0 * (i + 1) / Nsteps
                elapsed = pytime.perf_counter() - t_loop_start
                eta = elapsed / (i + 1) * (Nsteps - i - 1)
                log(
                    f"  [{pct:5.1f}%]  step {i + 1}/{Nsteps}  elapsed {elapsed:.1f}s  ETA {eta:.1f}s"
                )

        t_loop_end = pytime.perf_counter()
        if write_vtx:
            vtx.close()

        A.destroy()
        b.destroy()
        solver.destroy()

    # ==========================================================================
    # BRANCH B: EXPLICIT LUMPED-MASS RK4 SOLVER
    # ==========================================================================
    elif stype == "explicit":
        u = fem.Function(V, name="Displacement")
        v = fem.Function(V, name="Velocity")
        v_half = fem.Function(V, name="v_half")

        t_setup_start = pytime.perf_counter()

        K_form = fem.form(stiffness(du, u_))
        K_mat = fem.petsc.assemble_matrix(K_form, bcs=[])
        K_mat.assemble()

        M_form = fem.form(mass(du, u_))
        M_mat = fem.petsc.assemble_matrix(M_form, bcs=[])
        M_mat.assemble()

        ones = M_mat.createVecRight()
        ones.set(1.0)
        M_lumped = M_mat.createVecLeft()
        M_mat.mult(ones, M_lumped)
        n_owned_dofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
        M_lumped_np = M_lumped.array_r[:n_owned_dofs].copy()

        ones.destroy()
        M_mat.destroy()

        if comm.allreduce(bool(np.any(~np.isfinite(M_lumped_np) | (M_lumped_np <= 0))), op=MPI.LOR):
            raise RuntimeError("Mass lumping produced non-positive entries!")
        M_inv_np = 1.0 / M_lumped_np

        # Conservative spectral bounds; no rank-dependent random iteration and
        # no silent modification of the requested physical damping.
        omega2_max = operator_bound(K_mat, M_lumped_np)
        omega_max = np.sqrt(omega2_max)
        C_form = fem.form(damping(du, u_))
        C_mat = fem.petsc.assemble_matrix(C_form, bcs=[])
        C_mat.assemble()

        if any_viscoelastic:
            Kv_form = fem.form(damping_visco(du, u_))
            Kv_mat = fem.petsc.assemble_matrix(Kv_form, bcs=[])
            Kv_mat.assemble()

            C_mat.axpy(1., Kv_mat, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
            Kv_mat.destroy()
        damping_bound = operator_bound(C_mat, M_lumped_np)
        # For symmetric PSD damping/stiffness, first-order-system eigenvalues
        # lie in Re(z)<=0 and |z|<=max(sqrt(k_bound), damping_bound).
        # The left-half unit disk lies within classical RK4's stability region.
        dt_crit = 1.0 / max(omega_max, damping_bound)
        requested_dt = float(cfg.get("dt") or CFL*dt_crit)
        if requested_dt > dt_crit*(1+1e-12):
            raise ValueError(f"Requested dt={requested_dt:g} exceeds conservative RK4 limit {dt_crit:g}")
        dt_val, Nsteps = time_grid(T_total, requested_dt)
        if cfg["run_postprocess"] and Nsteps < 31:
            raise ValueError("Postprocessing requires at least 32 time samples; reduce dt or disable it")

        if rank == 0:
            log(f"  Frequency upper bound = {omega_max:.2e} rad/s")
            log(f"  Damping spectral bound = {damping_bound:.2e} 1/s")
            log(f"  Conservative step limit = {dt_crit * 1e9:.2f} ns")
            log(f"  Δt (CFL={CFL}) = {dt_val * 1e9:.2f} ns")
            log(f"  Number of steps: {Nsteps}")
            log(f"  Total time T   = {T_total * 1e6:.1f} μs")

        F_ext_form = fem.form(F_ext_form_expr)
        f_vec = fem.petsc.create_vector(V)
        r_vec = fem.petsc.create_vector(V)

        # Pre-assemble unit spatial load vector once for high explicit loop throughput
        traction_amplitude.value = 1.0
        f_spatial = fem.petsc.create_vector(V)
        fem.petsc.assemble_vector(f_spatial, F_ext_form)
        f_spatial.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        t_setup_end = pytime.perf_counter()

        sensor_data_x = np.zeros((Nsteps + 1, n_owned))
        sensor_data_y = np.zeros((Nsteps + 1, n_owned))
        times = np.zeros(Nsteps + 1)
        t_current = 0.0

        if write_vtx:
            vtx = io.VTXWriter(comm, f"{out_dir}/guided_wave.bp", [u], engine="BP4")
        save_freq = max(Nsteps // target_snapshots, 1)

        u_arr = u.x.array
        v_arr = v.x.array
        vh_arr = v_half.x.array

        u_petsc = u.x.petsc_vec
        v_half_petsc = v_half.x.petsc_vec

        report_interval = max(Nsteps // 20, 1)

        def acceleration(displacement, velocity, t):
            u_arr[:n_owned_dofs] = displacement
            vh_arr[:n_owned_dofs] = velocity
            K_mat.mult(u_petsc, r_vec)
            C_mat.mult(v_half_petsc, f_vec)
            r_vec.axpy(1., f_vec)
            force = traction_max*ricker(t, f_center, t0_ricker)
            return (force*f_spatial.array_r[:n_owned_dofs] - r_vec.array_r[:n_owned_dofs])*M_inv_np

        time_update = 0.0
        time_io = 0.0

        if rank == 0:
            log(f"\n{'═' * 58}")
            log(f"  Starting EXPLICIT time-stepping  ({Nsteps} steps)")
            log(f"{'═' * 58}")

        t_loop_start = pytime.perf_counter()

        for i in range(Nsteps):
            t_previous = i*dt_val
            t_current = (i+1)*dt_val
            times[i + 1] = t_current

            t0 = pytime.perf_counter()
            displacement, velocity = rk4_step(
                u_arr[:n_owned_dofs].copy(), v_arr[:n_owned_dofs].copy(),
                t_previous, dt_val, acceleration)
            u_arr[:n_owned_dofs] = displacement
            v_arr[:n_owned_dofs] = velocity
            u.x.scatter_forward()
            t1 = pytime.perf_counter()
            time_update += t1 - t0

            if n_owned > 0:
                values = probes.sample(u)
                sensor_data_x[i + 1] = values[:, 0]
                sensor_data_y[i + 1] = values[:, 1]

            if write_vtx and i % save_freq == 0:
                t5 = pytime.perf_counter()
                vtx.write(t_current)
                time_io += pytime.perf_counter() - t5

            if rank == 0 and (i + 1) % report_interval == 0:
                pct = 100.0 * (i + 1) / Nsteps
                elapsed = pytime.perf_counter() - t_loop_start
                eta = elapsed / (i + 1) * (Nsteps - i - 1)
                log(
                    f"  [{pct:5.1f}%]  step {i + 1}/{Nsteps}  elapsed {elapsed:.1f}s  ETA {eta:.1f}s"
                )

        t_loop_end = pytime.perf_counter()
        if write_vtx:
            vtx.close()

        K_mat.destroy()
        C_mat.destroy()
        M_lumped.destroy()
        f_spatial.destroy()
        f_vec.destroy()
        r_vec.destroy()

    # ── Gather sensor data across all MPI ranks ────────────────
    if comm.allreduce(not (np.isfinite(sensor_data_x).all() and np.isfinite(sensor_data_y).all()), op=MPI.LOR):
        raise RuntimeError("Nonfinite sensor data: simulation failed")
    sensor_data_x_all = probes.gather(sensor_data_x)
    sensor_data_y_all = probes.gather(sensor_data_y)
    cfg.update(absorb_alpha=absorb_alpha, absorb_beta=absorb_beta)
    cfg["runtime"] = {
        "dt": dt_val, "steps": Nsteps, "duration": Nsteps*dt_val,
        "nx": nx, "ny_per_region": ny_per_region, "ndofs": ndofs_global,
        "mpi_ranks": nprocs, "mesh_frequency": mesh_frequency,
        "integrator": "RK4" if stype == "explicit" else "Newmark",
        "python": platform.python_version(), "numpy": np.__version__,
        "dolfinx": dolfinx.__version__, "petsc": list(PETSc.Sys.getVersion()),
        "source_sha256": {p: hashlib.sha256((SOURCE_DIR / p).read_bytes()).hexdigest()
                          for p in ("guided_waves.py", "wave_numerics.py")},
    }
    if stype == "explicit":
        cfg["runtime"].update(stiffness_bound=omega2_max, damping_bound=damping_bound,
                              dt_limit=dt_crit)

    # ── Performance summary & data save (rank 0 only) ──────────
    saved_npz = os.path.join(out_dir, "sensor_data.npz")
    if rank == 0:
        wall_total = pytime.perf_counter() - wall_t0
        loop_total = t_loop_end - t_loop_start
        setup_total = t_setup_end - t_setup_start

        log(f"\n{'═' * 58}")
        log(f"  Performance Summary ({stype.upper()})")
        log(f"{'═' * 58}")
        log(f"  Total wall time:     {wall_total:.2f} s")
        log(f"  ├─ Setup / factorise: {setup_total:.2f} s")
        log(f"  ├─ Time-step loop:    {loop_total:.2f} s")
        log(f"  └─ Per-step average:  {loop_total / Nsteps * 1e3:.2f} ms/step")
        log(f"  Throughput:           {Nsteps / loop_total:.0f} steps/s")
        log(f"{'═' * 58}")

        top_lyr_or_hs = (
            norm_plate_layers[-1]
            if norm_plate_layers
            else struct_summary["bottom_halfspace"]
        )
        is_fluid_top = bool(top_lyr_or_hs.get("is_fluid", False))

        # Save sensor data .npz with comprehensive multilayer & halfspace metadata
        save_kwargs = {
            "times": times,
            "sensor_x": sensor_x_arr,
            "sensor_y": sensor_y_coord,
            "sensor_coords": np.column_stack(
                [sensor_x_arr, np.full_like(sensor_x_arr, sensor_y_coord)]
            ),
            "load_y": H_plate if has_top_halfspace else y_max,
            "sensor_data_x": sensor_data_x_all,
            "sensor_data_y": sensor_data_y_all,
            "f_center": f_center,
            "f_centers": np.array([f_center]),
            "plate_length": plate_length,
            "plate_thickness": H_plate,
            "rho": rho_top,
            "c_L": c_L_top,
            "c_T": c_T_top,
            "has_fluid": has_fluid,
            "is_fluid_top": is_fluid_top,
            "structure_mode": active_structure_mode,
            "material_model": (
                f"{active_structure_mode} ({n_plate_layers} plate layers)"
                if n_plate_layers > 1 or active_structure_mode != "1_layer"
                else norm_plate_layers[0]["material_model"]
            ),
            "n_layers": n_plate_layers,
            "layers_names": np.array(
                [layer["name"] for layer in norm_plate_layers], dtype="U"
            ),
            "layers_thicknesses": np.array(
                [layer["thickness"] for layer in norm_plate_layers]
            ),
            "layers_rhos": np.array([layer["rho"] for layer in norm_plate_layers]),
            "layers_models": np.array(
                [layer["material_model"] for layer in norm_plate_layers], dtype="U"
            ),
            "all_regions_names": np.array([r["name"] for r in all_regions], dtype="U"),
            "all_regions_thicknesses": np.array([r["thickness"] for r in all_regions]),
            "all_regions_roles": np.array([r["role"] for r in all_regions], dtype="U"),
            "all_regions_models": np.array(
                [r["material_model"] for r in all_regions], dtype="U"
            ),
            "has_bottom_halfspace": has_bottom_halfspace,
            "has_top_halfspace": has_top_halfspace,
            "y_min": y_min,
            "y_max": y_max,
            "source_x": source_x,
            "sensor_offset": sensor_offset,
            "sensor_pitch": sensor_pitch,
            "f_min_disp_MHz": f_min_disp_MHz,
            "f_max_disp_MHz": f_max_disp_MHz,
            "k_min_disp": k_min_disp,
            "k_max_disp": k_max_disp,
        }
        if c_fluid_min is not None:
            save_kwargs["c_fluid"] = c_fluid_min

        if n_plate_layers == 1:
            if norm_plate_layers[0]["material_model"] == "isotropic":
                save_kwargs.update(
                    E=norm_plate_layers[0]["E"], nu=norm_plate_layers[0]["nu"]
                )
            elif norm_plate_layers[0]["material_model"] == "fluid":
                save_kwargs.update(
                    c_fluid=norm_plate_layers[0]["c_fluid"],
                    K=norm_plate_layers[0]["K"],
                )
            else:
                save_kwargs.update(C_ij=norm_plate_layers[0]["C_mat"])
            if norm_plate_layers[0]["material_model"] == "viscoelastic":
                save_kwargs.update(Cv_ij=norm_plate_layers[0]["Cv_mat"])
        elif active_structure_mode == "halfspace":
            hs = struct_summary["bottom_halfspace"]
            if hs.get("c_R") is not None:
                save_kwargs["c_R"] = hs["c_R"]
            if hs["material_model"] == "isotropic":
                save_kwargs.update(E=hs["E"], nu=hs["nu"])
            elif hs["material_model"] == "fluid":
                save_kwargs.update(
                    c_fluid=hs["c_fluid"],
                    K=hs["K"],
                )
            else:
                save_kwargs.update(C_ij=hs["C_mat"])
            if hs["material_model"] == "viscoelastic":
                save_kwargs.update(Cv_ij=hs["Cv_mat"])
            # Compatibility keys with halfspace_rayleigh_explicit.py:
            save_kwargs.update(
                total_depth=H_domain,
                halfspace_depth=H_domain - hs["absorb_depth"],
                absorb_depth_bottom=hs["absorb_depth"],
            )

        # Save exact resolved simulation configuration alongside sensor_data.npz
        cfg["sensor_y"] = float(sensor_y_coord)
        cfg["load_y"] = float(H_plate if has_top_halfspace else y_max)
        saved_cfg_path = os.path.join(out_dir, "simulation_config.yaml")
        dump_config(cfg, saved_cfg_path)
        log(f"  Saved simulation configuration: {saved_cfg_path}")

        save_kwargs["config_source"] = str(cfg.get("_config_source", ""))
        np.savez(saved_npz, **save_kwargs)
        log(f"\n  Saved simulation sensor data: {saved_npz}")

    if rank == 0 and cfg.get("quick_plot", True) and cfg.get("run_postprocess", True):
        # Quick summary visualization
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        ax = axes[0]
        t_exc = np.linspace(0, 4.0 / f_center, 1000)
        ax.plot(
            t_exc * 1e6, [ricker(ti, f_center, t0_ricker) for ti in t_exc], "b-", lw=1.5
        )
        ax.set_xlabel("Time [μs]")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Excitation: Ricker wavelet, f₀ = {f_center / 1e3:.0f} kHz")
        ax.grid(True, alpha=0.3)

        # Detrend for display
        _fs = 1.0 / dt_val
        _f_lo = f_center / 5.0
        _f_hi = min(f_center * 3.0, _fs / 2.0 * 0.9)
        _sos = butter(4, [_f_lo, _f_hi], btype="bandpass", fs=_fs, output="sos")
        sensor_y_detrend = np.zeros_like(sensor_data_y_all)
        for j in range(n_sensors):
            sensor_y_detrend[:, j] = sosfiltfilt(_sos, sensor_data_y_all[:, j])

        ax = axes[1]
        global_peak = np.max(np.abs(sensor_y_detrend))
        pitch_mm = (sensor_x_arr[1] - sensor_x_arr[0]) * 1e3
        n_show = min(n_sensors, 30)
        step = max(n_sensors // n_show, 1)
        trace_spacing = pitch_mm * step
        global_scale = 0.8 * trace_spacing / max(global_peak, 1e-30)

        for j in range(0, n_sensors, step):
            offset = sensor_x_arr[j] * 1e3
            ax.plot(
                times * 1e6,
                sensor_y_detrend[:, j] * global_scale + offset,
                "b-",
                lw=0.5,
            )
        ax.set_xlabel("Time [μs]")
        ax.set_ylabel("Position [mm]")
        ax.set_title("Out-of-plane displacement (u_y) — Waterfall plot")
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        _dt_us = (times[1] - times[0]) * 1e6
        _dx_mm = (sensor_x_arr[1] - sensor_x_arr[0]) * 1e3
        extent = [
            times[0] * 1e6 - _dt_us / 2.0,
            times[-1] * 1e6 + _dt_us / 2.0,
            sensor_x_arr[0] * 1e3 - _dx_mm / 2.0,
            sensor_x_arr[-1] * 1e3 + _dx_mm / 2.0,
        ]
        vmax = np.max(np.abs(sensor_y_detrend)) * 0.5
        if vmax > 0:
            ax.imshow(
                sensor_y_detrend.T,
                aspect="auto",
                extent=extent,
                origin="lower",
                cmap="seismic",
                vmin=-vmax,
                vmax=vmax,
            )
        ax.set_xlim(times[0] * 1e6, times[-1] * 1e6)
        ax.set_ylim(sensor_x_arr[0] * 1e3, sensor_x_arr[-1] * 1e3)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_xlabel("Time [μs]")
        ax.set_ylabel("Position [mm]")
        ax.set_title("B-scan (out-of-plane displacement)")

        plt.tight_layout()
        quick_plot_path = os.path.join(out_dir, "quick_results.png")
        fig.savefig(quick_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"  Quick visualisation saved: {quick_plot_path}")

    comm.Barrier()
    return saved_npz, out_dir


# ==============================================================================
# SECTION 5: Command-Line Interface & Application Orchestration
# ==============================================================================


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
                "    python guided_waves.py -c configs/01_single_layer_aluminum.yaml\n\n"
                "Or generate a template configuration with:\n"
                "    python guided_waves.py --generate-config my_config.yaml\n"
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
