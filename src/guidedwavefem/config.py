"""Multi-format configuration loading, validation, resolution, and serialization.

Supports:
  - YAML (.yaml, .yml), TOML (.toml), and JSON (.json)
  - Hierarchical and flat parameter configurations
  - Interactive CLI overrides
  - Commented YAML template generation
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np

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
    """Normalize nested hierarchical configuration dictionary into a flat parameter mapping."""
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
    """Load, flatten, validate, and apply CLI overrides to the configuration."""
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
        positive = (
            "sensing_length",
            "absorb_length",
            "f_center",
            "excitation_width",
            "sensor_pitch",
            "CFL",
            "time_factor",
        )
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
        for key, minimum in (
            ("n_sensors", 2),
            ("elems_per_wavelength", 1),
            ("min_elems_x", 1),
            ("min_elems_y", 1),
            ("target_snapshots", 1),
            ("degree", 1),
        ):
            value = resolved[key]
            if not np.isfinite(value) or int(value) != value or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        for key in (
            "source_x",
            "source_fx",
            "source_fy",
            "traction_max",
            "sensor_offset",
        ):
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
        if solver == "implicit" and (
            not np.isfinite([beta, gamma]).all()
            or gamma < 0.5
            or beta < 0.25 * (gamma + 0.5) ** 2
        ):
            raise ValueError(
                "Newmark requires gamma >= 0.5 and beta >= (gamma+0.5)^2/4"
            )
        length = 2 * resolved["absorb_length"] + resolved["sensing_length"]
        if not 0 <= resolved["source_x"] <= length:
            raise ValueError("source_x must be inside the domain")
        sensor_start = resolved["source_x"] + resolved["sensor_offset"]
        sensor_end = (
            sensor_start + (resolved["n_sensors"] - 1) * resolved["sensor_pitch"]
        )
        if sensor_start < 0 or sensor_end > length:
            raise ValueError("Sensor array must lie inside the domain")
        if resolved["run_postprocess"] and resolved["n_sensors"] < 3:
            raise ValueError("Postprocessing requires at least three sensors")
    for low, high in (
        ("f_min_disp_MHz", "f_max_disp_MHz"),
        ("k_min_disp", "k_max_disp"),
    ):
        if (
            not np.isfinite([resolved[low], resolved[high]]).all()
            or not 0 <= resolved[low] < resolved[high]
        ):
            raise ValueError(f"Require 0 <= {low} < {high}, with finite limits")
    return resolved


# Backward-compatibility alias
resolve_and_apply_config = parse_and_resolve_config


def generate_template_config(output_path: str) -> None:
    """Generate an annotated template configuration file in YAML format."""
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
