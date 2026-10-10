"""Configuration loading, schema validation, resolution, and serialization.

Supports:
  - YAML (.yaml, .yml), TOML (.toml), and JSON (.json)
  - Hierarchical (sectioned) and flat parameter layouts
  - Strict schema: unknown keys are rejected with "did you mean" suggestions
  - CLI overrides (highest precedence) and a single table of defaults
  - Commented YAML template generation
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import tomllib
from collections.abc import Mapping
from typing import Any

import numpy as np
import yaml

from wavesimfem.device import DEVICE_CHOICES_HELP, normalize_device_request
from wavesimfem.materials import (
    DEFAULT_STEPS_PER_PERIOD,
    canonicalize_mode,
    mesh_resolution,
    sizing_wavelength,
    structure_from_config,
)
from wavesimfem.sources import DEFAULT_N_CYCLES, SOURCE_TYPES
from wavesimfem.storage import atomic_output
from wavesimfem.units import parse_quantity

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
#: Section name -> {section key: flat key}
SECTION_KEYS: dict[str, dict[str, str]] = {
    "solver": {
        "type": "solver_type",
        "CFL": "CFL",
        "steps_per_period": "steps_per_period",
        "time_factor": "time_factor",
        "beta": "beta_val",
        "gamma": "gamma_val",
        "dt": "dt",
        "duration": "duration",
        "device": "device",
        "precision": "precision",
        "min_group_velocity": "min_group_velocity",
    },
    "geometry": {"sensing_length": "sensing_length", "absorb_length": "absorb_length"},
    "structure": {
        "mode": "structure_mode",
        "single_layer": "single_layer",
        "layers": "layers",
        "halfspace": "halfspace",
        "substrate": "substrate",
        "superstrate": "superstrate",
        "bottom_halfspace": "bottom_halfspace",
        "top_halfspace": "top_halfspace",
    },
    "absorbing_boundaries": {"alpha": "absorb_alpha", "beta": "absorb_beta"},
    "excitation": {
        k: k
        for k in (
            "source_type",
            "f_center",
            "n_cycles",
            "traction_max",
            "excitation_width",
            "source_x",
            "source_fx",
            "source_fy",
            "source_kind",
            "source_amplitude",
            "source_boundary",
            "source_y",
            "source_region",
        )
    },
    "mesh": {
        k: k
        for k in (
            "nx",
            "ny_per_region",
            "degree",
            "max_frequency",
            "min_frequency",
            "min_wavelength",
        )
    },
    "sensors": {
        **{k: k for k in ("n_sensors", "sensor_offset", "sensor_pitch", "sensor_y")},
        "sample_rate": "sensor_sample_rate",
        "observable": "sensor_observable",
        "region": "sensor_region",
    },
    "postprocess": {
        "wave_characterisation": "wave_characterisation",
        "structure_analysis": "structure_analysis",
        "lamb_modes": "lamb_modes",
        "run_postprocess": "run_postprocess",
        # SI numbers (Hz, rad/m) or unit strings ("2 MHz", "3 rad/mm"); null = automatic
        "f_min": "disp_f_min",
        "f_max": "disp_f_max",
        "k_min": "disp_k_min",
        "k_max": "disp_k_max",
        # Legacy (<= 0.2) keys in MHz and rad/mm, converted on load
        "f_min_disp_MHz": "f_min_disp_MHz",
        "f_max_disp_MHz": "f_max_disp_MHz",
        "k_min_disp": "k_min_disp",
        "k_max_disp": "k_max_disp",
    },
    "output": {
        k: k
        for k in (
            "output_dir",
            "target_snapshots",
            "write_vtx",
            "quick_plot",
            "compress_output",
            "history_storage",
            "figure_formats",
            "usetex",
        )
    },
}

#: Keys written by the solver into ``simulation_config.yaml`` (accepted on reload).
PROVENANCE_KEYS = frozenset({"runtime", "load_y", "source_units"})
FLAT_KEYS = (
    frozenset({flat for section in SECTION_KEYS.values() for flat in section.values()})
    | PROVENANCE_KEYS
    | {"acoustic_boundaries"}
)

FIGURE_FORMATS = ("png", "pdf", "svg", "eps")

#: Single source of truth for defaults.
DEFAULTS: dict[str, Any] = {
    "solver_type": "explicit",
    "device": "auto",
    "precision": "double",
    "CFL": None,  # required for explicit: user-selected fraction of the stability certificate
    "steps_per_period": DEFAULT_STEPS_PER_PERIOD,  # implicit: steps per max_frequency period
    "time_factor": 1.2,
    "beta_val": 0.25,
    "gamma_val": 0.5,
    "dt": None,
    "duration": None,
    "structure_mode": "N_layers",
    "absorb_alpha": None,  # auto: ABSORB_ALPHA_REF * f_center / 1 MHz
    "absorb_beta": None,  # auto: ABSORB_BETA_REF * 1 MHz / f_center
    "source_type": "ricker",
    "source_kind": "traction",
    "source_amplitude": None,  # SI: Pa, m/s² or s^-2 according to source_kind
    "source_boundary": "auto",
    "source_y": None,
    "source_region": None,
    "acoustic_boundaries": {side: "rigid" for side in ("top", "bottom", "left", "right")},
    "n_cycles": None,  # tone_burst only (default DEFAULT_N_CYCLES)
    "traction_max": 1.0e6,
    # λ_ref is the center-frequency sizing wavelength, including A0 for free isotropic
    # plates and any supplied min_wavelength bound (see _resolve_layout_defaults).
    "excitation_width": None,  # auto: min(λ_ref / 2, sensing_length / 10)
    "source_x": None,  # auto: absorb_length + max(min(max(λ_ref, width), L / 8), width / 2)
    "source_fx": 0.0,
    "source_fy": 1.0,
    "nx": None,  # required exact longitudinal element count
    "ny_per_region": None,  # required exact counts in bottom-to-top region order
    "degree": 2,
    "max_frequency": None,
    "min_frequency": None,  # arrival-sizing lower edge (default f_center / 5; no filtering)
    "min_wavelength": None,  # conservative wavelength bound across retained band [m]
    "min_group_velocity": None,  # conservative packet travel speed [m/s]
    "n_sensors": None,  # auto: fill the sensing region up to MAX_AUTO_SENSORS
    "sensor_offset": None,  # auto: min(2 λ_ref, sensing_length / 4)
    "sensor_pitch": None,  # auto: min(λ_ref / 8, λ_mesh / 4, sensing_length / 100)
    "sensor_y": None,
    "sensor_observable": "auto",
    "sensor_region": None,
    "sensor_sample_rate": None,  # auto: 20 × mesh frequency (solvers.AUTO_SAMPLES_PER_MESH_PERIOD)
    "run_postprocess": True,
    "structure_analysis": True,  # structure-aware dispersion, polarization and propagation report
    "wave_characterisation": None,
    "lamb_modes": False,
    "disp_f_min": None,  # auto: 0
    "disp_f_max": None,  # auto: 2 * f_center
    "disp_k_min": None,  # auto: 0
    "disp_k_max": None,  # auto: sizing wavenumber at disp_f_max, limited by spatial Nyquist
    "output_dir": None,
    "target_snapshots": 200,
    "write_vtx": False,
    "quick_plot": True,
    "compress_output": False,
    "history_storage": "memory",  # disk: scratch-backed histories with bounded collection RAM
    "figure_formats": list(FIGURE_FORMATS),
    "usetex": False,
}

#: Upper limit for the automatic sensor count (``sensors.n_sensors: null``).
MAX_AUTO_SENSORS = 1001
#: Smallest accepted ``solver.steps_per_period`` (period error ≈ 20 % at 4).
MIN_STEPS_PER_PERIOD = 4.0
#: Sponge damping peaks at the 1 MHz reference; auto values scale so that
#: ``alpha / f_center`` and ``beta * f_center`` stay constant across frequency scales.
#: Mass-proportional ``alpha`` provides the absorption (damping ratio ≈ 0.8 at f0);
#: ``beta`` is kept small because stiffness-proportional damping grows with the
#: mesh frequency and would otherwise limit the explicit time step.
ABSORB_REFERENCE_FREQUENCY = 1.0e6
ABSORB_ALPHA_REF = 1.0e7
ABSORB_BETA_REF = 1.0e-9

#: Canonical dispersion-limit keys -> (quantity kind, user-facing name).
DISPERSION_KEYS: dict[str, tuple[str, str]] = {
    "disp_f_min": ("frequency", "postprocess.f_min"),
    "disp_f_max": ("frequency", "postprocess.f_max"),
    "disp_k_min": ("wavenumber", "postprocess.k_min"),
    "disp_k_max": ("wavenumber", "postprocess.k_max"),
}
#: Legacy dispersion keys -> (canonical key, factor to SI).
LEGACY_DISPERSION_KEYS: dict[str, tuple[str, float]] = {
    "f_min_disp_MHz": ("disp_f_min", 1e6),
    "f_max_disp_MHz": ("disp_f_max", 1e6),
    "k_min_disp": ("disp_k_min", 1e3),
    "k_max_disp": ("disp_k_max", 1e3),
}


def _suggest(key: str, candidates: set[str] | frozenset[str]) -> str:
    match = difflib.get_close_matches(key, sorted(candidates), n=1)
    return f" Did you mean '{match[0]}'?" if match else ""


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------
class _UniqueSafeLoader(yaml.SafeLoader):
    """Validate original mapping keys before SafeLoader expands YAML merges."""

    def construct_document(self, node: Any) -> Any:
        # flatten_mapping mutates merge-source nodes, including aliased nodes.
        # Inspect the entire original graph first so inherited overrides remain
        # valid and duplicate keys cannot disappear during merge expansion.
        pending = [node]
        visited = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            if isinstance(current, yaml.MappingNode):
                self._validate_mapping_keys(current)
                for key_node, value_node in current.value:
                    pending.extend((key_node, value_node))
            elif isinstance(current, yaml.SequenceNode):
                pending.extend(current.value)
        return super().construct_document(node)

    def _validate_mapping_keys(self, node: yaml.MappingNode) -> None:
        seen = {}
        merge_key = object()
        for key_node, _ in node.value:
            if not isinstance(key_node, yaml.ScalarNode):
                raise ValueError(f"Invalid mapping key at {key_node.start_mark}")
            if key_node.tag == "tag:yaml.org,2002:merge":
                # A merge directive and a quoted literal "<<" key are distinct.
                key = merge_key
            elif key_node.tag == "tag:yaml.org,2002:value":
                # SafeLoader normalizes the plain '=' key to a string on flattening.
                key = self.construct_scalar(key_node)
            else:
                key = self.construct_object(key_node, deep=True)
            try:
                if key in seen:
                    label = "<<" if key is merge_key else key
                    raise ValueError(
                        f"Duplicate key {label!r} at {key_node.start_mark}; "
                        f"first defined at {seen[key]}"
                    )
                seen[key] = key_node.start_mark
            except TypeError as exc:
                raise ValueError(f"Invalid mapping key at {key_node.start_mark}") from exc


def _unique_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r} in the same object")
        result[key] = value
    return result


def load_config(config_path: str) -> dict[str, Any]:
    """Load a configuration mapping from a YAML, TOML, or JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    ext = os.path.splitext(config_path)[1].lower()
    if ext in (".yaml", ".yml"):
        with open(config_path, encoding="utf-8") as f:
            try:
                data = yaml.load(f, Loader=_UniqueSafeLoader)
            except yaml.YAMLError as exc:
                raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc
    elif ext == ".toml":
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
    elif ext == ".json":
        with open(config_path, encoding="utf-8") as f:
            try:
                data = json.load(f, object_pairs_hook=_unique_json_pairs)
            except ValueError as exc:
                raise ValueError(f"Invalid JSON in {config_path}: {exc}") from exc
    else:
        raise ValueError(
            f"Unsupported configuration format '{ext}'. Supported: .yaml, .yml, .toml, .json"
        )
    if not isinstance(data, dict):
        raise TypeError(f"Configuration root in {config_path} must be a key-value mapping.")
    return data


def _sanitize(item: Any) -> Any:
    if isinstance(item, np.ndarray):
        return item.tolist()
    if isinstance(item, (bool, np.bool_)):
        return bool(item)
    if isinstance(item, (np.floating, float)):
        return float(item)
    if isinstance(item, (np.integer, int)):
        return int(item)
    if isinstance(item, Mapping):
        return {k: _sanitize(v) for k, v in item.items() if not str(k).startswith("_")}
    if isinstance(item, (list, tuple)):
        return [_sanitize(v) for v in item]
    return item


def dump_config(cfg_dict: dict[str, Any], out_path: str) -> None:
    """Serialize a configuration to YAML (``.yaml``/``.yml``) or JSON (otherwise).

    Keys starting with ``_`` are internal and are not written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cleaned = _sanitize(cfg_dict)
    with atomic_output(out_path) as f:
        if os.path.splitext(out_path)[1].lower() in (".yaml", ".yml"):
            yaml.safe_dump(cleaned, f, default_flow_style=False, sort_keys=False)
        else:
            json.dump(cleaned, f, indent=2)


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------
def flatten_config(cfg_in: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a sectioned configuration into a flat parameter mapping.

    Raises
    ------
    ValueError
        On unknown keys (with a suggestion), malformed sections, or a parameter
        defined both inside a section and at the top level.
    """
    if not isinstance(cfg_in, Mapping):
        raise TypeError("Configuration must be a mapping")
    flat: dict[str, Any] = {}
    origin: dict[str, str] = {}

    def _reject_removed_mesh_key(key: str, where: str) -> None:
        if key in {"elems_per_wavelength", "min_elems_x", "min_elems_y"}:
            raise ValueError(
                f"'{where}' is no longer supported: automatic mesh sizing has been removed. "
                "Specify exact counts with 'mesh.nx' and 'mesh.ny_per_region' "
                "(one count for each region, ordered bottom-to-top)."
            )

    def _put(flat_key: str, value: Any, where: str) -> None:
        if flat_key in flat:
            raise ValueError(
                f"Parameter '{flat_key}' is defined twice ({origin[flat_key]} and {where})"
            )
        flat[flat_key] = value
        origin[flat_key] = where

    for key, value in cfg_in.items():
        key = str(key)
        if key.startswith("_"):
            flat[key] = value  # internal metadata (e.g. _config_source)
            continue
        _reject_removed_mesh_key(key, key)
        if key in SECTION_KEYS:
            if not isinstance(value, Mapping):
                raise ValueError(f"Section '{key}' must be a mapping, got {type(value).__name__}")
            mapping = SECTION_KEYS[key]
            for sub_key, sub_value in value.items():
                if key == "mesh":
                    _reject_removed_mesh_key(str(sub_key), f"mesh.{sub_key}")
                if sub_key not in mapping:
                    raise ValueError(
                        f"Unknown key '{key}.{sub_key}'.{_suggest(str(sub_key), set(mapping))}"
                    )
                _put(mapping[sub_key], sub_value, f"'{key}.{sub_key}'")
        elif key in FLAT_KEYS:
            _put(key, value, f"top-level '{key}'")
        else:
            raise ValueError(
                f"Unknown configuration key '{key}'.{_suggest(key, FLAT_KEYS | set(SECTION_KEYS))}"
            )
    return flat


# ---------------------------------------------------------------------------
# Typed validation helpers
# ---------------------------------------------------------------------------
def _float(cfg: dict[str, Any], key: str, *, positive: bool = False, nonneg: bool = False) -> float:
    value = cfg[key]
    if isinstance(value, bool):
        raise ValueError(f"'{key}' must be a number, got a boolean")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' must be a number, got {value!r}") from None
    if not np.isfinite(out):
        raise ValueError(f"'{key}' must be finite, got {value!r}")
    if positive and out <= 0:
        raise ValueError(f"'{key}' must be positive, got {out}")
    if nonneg and out < 0:
        raise ValueError(f"'{key}' must be nonnegative, got {out}")
    cfg[key] = out
    return out


def _int(cfg: dict[str, Any], key: str, minimum: int) -> int:
    value = cfg[key]
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"'{key}' must be an integer >= {minimum}, got {value!r}")
    if not np.isfinite(value) or int(value) != value or value < minimum:
        raise ValueError(f"'{key}' must be an integer >= {minimum}, got {value!r}")
    cfg[key] = int(value)
    return int(value)


def _bool(cfg: dict[str, Any], key: str) -> bool:
    value = cfg[key]
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"'{key}' must be true or false, got {value!r}")
    cfg[key] = bool(value)
    return bool(value)


def _validate_output_options(cfg: dict[str, Any]) -> None:
    for key in (
        "run_postprocess",
        "structure_analysis",
        "write_vtx",
        "quick_plot",
        "compress_output",
    ):
        _bool(cfg, key)
    history_storage = cfg["history_storage"]
    if not isinstance(history_storage, str) or history_storage.strip().lower() not in (
        "memory",
        "disk",
    ):
        raise ValueError("'history_storage' must be 'memory' or 'disk'")
    cfg["history_storage"] = history_storage.strip().lower()
    formats = cfg["figure_formats"]
    if isinstance(formats, str):
        formats = [s.strip() for s in formats.split(",") if s.strip()]
    if not isinstance(formats, (list, tuple)) or not formats:
        raise ValueError(f"'figure_formats' must be a non-empty list from {FIGURE_FORMATS}")
    formats = [str(f).lower().lstrip(".") for f in formats]
    bad = [f for f in formats if f not in FIGURE_FORMATS]
    if bad:
        raise ValueError(f"Unsupported figure format(s) {bad}; choose from {FIGURE_FORMATS}")
    cfg["figure_formats"] = list(dict.fromkeys(formats))
    usetex = cfg["usetex"]
    if isinstance(usetex, str) and usetex.lower() == "auto":
        cfg["usetex"] = "auto"
    elif isinstance(usetex, (bool, np.bool_)):
        cfg["usetex"] = bool(usetex)
    else:
        raise ValueError(f"'usetex' must be 'auto', true or false, got {usetex!r}")
    if cfg["output_dir"] is not None and not isinstance(cfg["output_dir"], str):
        raise ValueError("'output_dir' must be a string or null")
    _resolve_dispersion_limits(cfg)


def _resolve_dispersion_limits(cfg: dict[str, Any]) -> None:
    """Convert legacy keys and unit strings to SI (Hz, rad/m); ``None`` stays automatic."""
    for legacy, (key, factor) in LEGACY_DISPERSION_KEYS.items():
        value = cfg.pop(legacy, None)
        if value is None:
            continue
        if cfg.get(key) is not None:
            name = DISPERSION_KEYS[key][1]
            raise ValueError(f"Specify only one of '{name}' and the legacy '{legacy}'")
        cfg[key] = _float({legacy: value}, legacy) * factor
    for key, (kind, name) in DISPERSION_KEYS.items():
        if cfg.get(key) is None:
            continue
        try:
            value = parse_quantity(cfg[key], kind)  # type: ignore[arg-type]
        except ValueError as exc:
            raise ValueError(f"'{name}': {exc}") from None
        if value < 0:
            raise ValueError(f"'{name}' must be nonnegative, got {cfg[key]!r}")
        cfg[key] = value
    for low, high in (("disp_f_min", "disp_f_max"), ("disp_k_min", "disp_k_max")):
        lo, hi = cfg.get(low), cfg.get(high)
        if lo is not None and hi is not None and not lo < hi:
            raise ValueError(
                f"Require {DISPERSION_KEYS[low][1]} < {DISPERSION_KEYS[high][1]}, got {lo:g} and {hi:g}"
            )


def _resolve_source(cfg: dict[str, Any]) -> None:
    source_type = str(cfg["source_type"]).lower().strip().replace("-", "_")
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"source_type must be one of {SOURCE_TYPES}, got {cfg['source_type']!r}")
    cfg["source_type"] = source_type
    if source_type == "tone_burst":
        if cfg.get("n_cycles") is None:
            cfg["n_cycles"] = DEFAULT_N_CYCLES
        _int(cfg, "n_cycles", 1)
    elif cfg.get("n_cycles") is not None:
        raise ValueError("'n_cycles' only applies to source_type 'tone_burst'")


def _validate_acoustic_options(cfg: dict[str, Any], summary: dict[str, Any]) -> None:
    """Validate source and observable semantics before assembly."""
    kinds = {
        "traction": "Pa",
        "acoustic_pressure": "Pa",
        "acoustic_normal_acceleration": "m/s^2",
        "acoustic_volume": "s^-2",
    }
    kind = str(cfg["source_kind"]).strip().lower()
    if kind not in kinds:
        raise ValueError(f"source_kind must be one of {tuple(kinds)}, got {kind!r}")
    cfg["source_kind"] = kind
    cfg["source_units"] = kinds[kind]
    if kind == "traction":
        if cfg.get("source_amplitude") is not None:
            raise ValueError(
                "source_amplitude applies to acoustic sources; use traction_max for traction"
            )
        if cfg["source_fx"] == 0 and cfg["source_fy"] == 0:
            raise ValueError("Source direction (source_fx, source_fy) cannot be zero")
    else:
        if not summary["has_acoustic"]:
            raise ValueError(f"source_kind '{kind}' requires an acoustic fluid region")
        if cfg.get("source_amplitude") is None:
            raise ValueError(
                f"source_kind '{kind}' requires excitation.source_amplitude in SI {kinds[kind]}"
            )
        _float(cfg, "source_amplitude")
    boundary = str(cfg["source_boundary"]).strip().lower()
    if boundary not in ("auto", "top", "bottom", "left", "right", "interface"):
        raise ValueError("source_boundary must be auto, top, bottom, left, right or interface")
    cfg["source_boundary"] = boundary
    if cfg.get("source_y") is not None:
        y = _float(cfg, "source_y")
        if not summary["y_min"] <= y <= summary["y_max"]:
            raise ValueError("source_y must lie inside the domain")
    if boundary == "interface" and cfg.get("source_y") is None:
        raise ValueError("source_boundary: interface requires excitation.source_y")
    names = {r["name"]: r for r in summary["all_regions"]}
    if len(names) != len(summary["all_regions"]) and (
        cfg.get("source_region") or cfg.get("sensor_region")
    ):
        raise ValueError(
            "Material region names must be unique when selecting a source or sensor region"
        )
    for key in ("source_region", "sensor_region"):
        name = cfg.get(key)
        if name is not None and (not isinstance(name, str) or name not in names):
            raise ValueError(f"{key} must name one material region from {tuple(names)}")
    if kind == "acoustic_volume":
        name = cfg.get("source_region")
        if name is None or not names[name]["is_acoustic"]:
            raise ValueError("acoustic_volume requires source_region naming an acoustic fluid")
        if boundary != "auto" or cfg.get("source_y") is not None:
            raise ValueError(
                "acoustic_volume is a body source; source_boundary/source_y do not apply"
            )
    observable = str(cfg["sensor_observable"]).strip().lower()
    observables = ("auto", "displacement", "velocity", "pressure", "particle_velocity")
    if observable not in observables:
        raise ValueError(f"sensor_observable must be one of {observables}, got {observable!r}")
    cfg["sensor_observable"] = observable
    if observable in ("pressure", "particle_velocity") and not summary["has_acoustic"]:
        raise ValueError(f"sensor_observable '{observable}' requires an acoustic fluid region")
    if cfg.get("sensor_region") is not None:
        region = names[cfg["sensor_region"]]
        if observable in ("pressure", "particle_velocity") and not region["is_acoustic"]:
            raise ValueError(f"sensor_observable '{observable}' requires an acoustic sensor_region")
        if observable in ("displacement", "velocity") and region["is_acoustic"]:
            raise ValueError(
                f"sensor_observable '{observable}' requires a solid/legacy sensor_region"
            )
    boundaries = cfg["acoustic_boundaries"]
    if not isinstance(boundaries, Mapping):
        raise ValueError(
            "acoustic_boundaries must map top, bottom, left and right to boundary types"
        )
    sides = ("top", "bottom", "left", "right")
    unknown = set(boundaries) - set(sides)
    if unknown:
        raise ValueError(f"Unknown acoustic boundary side(s): {sorted(unknown)}")
    resolved = {side: str(boundaries.get(side, "rigid")).strip().lower() for side in sides}
    if any(value not in ("rigid", "pressure_release", "radiation") for value in resolved.values()):
        raise ValueError("Acoustic boundary types must be rigid, pressure_release or radiation")
    cfg["acoustic_boundaries"] = resolved


def _resolve_automatic_values(cfg: dict[str, Any], summary: Mapping[str, Any]) -> None:
    """Fill frequency-scaled sponge damping and dispersion plot limits left as ``null``."""
    f0 = cfg["f_center"]
    scale = f0 / ABSORB_REFERENCE_FREQUENCY
    if cfg["absorb_alpha"] is None:
        cfg["absorb_alpha"] = ABSORB_ALPHA_REF * scale
    if cfg["absorb_beta"] is None:
        cfg["absorb_beta"] = ABSORB_BETA_REF / scale
    for key in ("absorb_alpha", "absorb_beta"):
        _float(cfg, key, nonneg=True)

    if cfg["disp_f_min"] is None:
        cfg["disp_f_min"] = 0.0
    if cfg["disp_f_max"] is None:
        cfg["disp_f_max"] = max(2.0 * f0, 1.5 * cfg["disp_f_min"])
    if cfg["disp_k_min"] is None:
        cfg["disp_k_min"] = 0.0
    if cfg["disp_k_max"] is None:
        k_nyquist = np.pi / cfg["sensor_pitch"]
        k_wave = 2.0 * np.pi / sizing_wavelength(cfg, summary, cfg["disp_f_max"])
        cfg["disp_k_max"] = float(max(min(k_wave, k_nyquist), 1.5 * cfg["disp_k_min"]))
    _resolve_dispersion_limits(cfg)  # re-check ordering with the automatic values


def _resolve_layout_defaults(cfg: dict[str, Any], summary: Mapping[str, Any]) -> None:
    """Fill source/sensor geometry using the center-frequency sizing wavelength.

    Sensor pitch also resolves the shortest retained-band wavelength. A0 is
    included for free isotropic plates; other guided structures need explicit
    conservative bounds. Domain-fraction caps keep the layout inside the mesh.
    """
    wavelength = sizing_wavelength(cfg, summary, cfg["f_center"])
    sensing = cfg["sensing_length"]
    if cfg["excitation_width"] is None:
        cfg["excitation_width"] = min(0.5 * wavelength, 0.1 * sensing)
    _float(cfg, "excitation_width", positive=True)
    length = 2 * cfg["absorb_length"] + sensing
    boundary = str(cfg["source_boundary"]).strip().lower()
    if boundary in ("left", "right"):
        boundary_x = 0.0 if boundary == "left" else length
        if cfg["source_x"] is not None:
            supplied_x = _float(cfg, "source_x")
            if not np.isclose(supplied_x, boundary_x, rtol=0.0, atol=1e-12 * length):
                raise ValueError(
                    f"source_boundary: {boundary} requires source_x={boundary_x:g} m; "
                    f"got {supplied_x:g} m"
                )
        cfg["source_x"] = boundary_x
    elif cfg["source_x"] is None:
        gap = min(max(wavelength, cfg["excitation_width"]), 0.125 * sensing)
        cfg["source_x"] = cfg["absorb_length"] + max(gap, 0.5 * cfg["excitation_width"])
    _float(cfg, "source_x")
    if not 0 <= cfg["source_x"] <= length:
        raise ValueError(f"source_x={cfg['source_x']} must lie inside [0, {length}]")
    if cfg["sensor_pitch"] is None:
        shortest = sizing_wavelength(cfg, summary, cfg.get("max_frequency") or 3 * cfg["f_center"])
        cfg["sensor_pitch"] = min(wavelength / 8.0, shortest / 4.0, 0.01 * sensing)
    _float(cfg, "sensor_pitch", positive=True)
    sensor_gap = min(2.0 * wavelength, 0.25 * sensing)
    if boundary == "right" and cfg["sensor_offset"] is None:
        # Receivers remain ordered in increasing x for FFT processing. Place the
        # last receiver to the left of a right-boundary source, then fill leftward.
        last = min(length - sensor_gap, cfg["absorb_length"] + sensing)
        if cfg["n_sensors"] is None:
            span = last - cfg["absorb_length"]
            count = int(np.floor(span / cfg["sensor_pitch"] * (1.0 + 1e-9))) + 1
            if count < 3:
                raise ValueError("The automatic sensor array does not fit in the sensing region")
            cfg["n_sensors"] = min(count, MAX_AUTO_SENSORS)
        count = _int(cfg, "n_sensors", 2)
        cfg["sensor_offset"] = last - (count - 1) * cfg["sensor_pitch"] - cfg["source_x"]
    elif cfg["sensor_offset"] is None:
        cfg["sensor_offset"] = sensor_gap
    _float(cfg, "sensor_offset")
    if cfg["n_sensors"] is None:
        first = cfg["source_x"] + cfg["sensor_offset"]
        span = cfg["absorb_length"] + sensing - first
        count = int(np.floor(span / cfg["sensor_pitch"] * (1.0 + 1e-9))) + 1 if span > 0 else 0
        if count < 3:
            raise ValueError(
                "The automatic sensor array does not fit in the sensing region; set "
                "sensors.n_sensors / sensor_offset / sensor_pitch or enlarge sensing_length"
            )
        cfg["n_sensors"] = min(count, MAX_AUTO_SENSORS)


def _validate_simulation(cfg: dict[str, Any], source: str) -> None:
    for key in ("sensing_length", "absorb_length", "f_center"):
        if cfg.get(key) is None:
            raise ValueError(f"Configuration '{source}' must specify '{key}'.")

    for key in ("nx", "ny_per_region"):
        if cfg.get(key) is None:
            raise ValueError(f"Configuration '{source}' must specify 'mesh.{key}'.")
    if cfg["solver_type"] == "explicit" and cfg.get("CFL") is None:
        raise ValueError(f"Configuration '{source}' must specify 'solver.CFL' for explicit runs.")
    if cfg.get("CFL") is not None and _float(cfg, "CFL", positive=True) > 1:
        raise ValueError("CFL must be in (0, 1] (fraction of the RK4 stability limit)")

    for key in ("sensing_length", "f_center", "time_factor"):
        _float(cfg, key, positive=True)
    _float(cfg, "absorb_length", nonneg=True)
    if _float(cfg, "steps_per_period", positive=True) < MIN_STEPS_PER_PERIOD:
        raise ValueError(
            f"'steps_per_period' must be >= {MIN_STEPS_PER_PERIOD:g}, got {cfg['steps_per_period']}"
        )
    for key in (
        "dt",
        "duration",
        "max_frequency",
        "min_frequency",
        "min_wavelength",
        "min_group_velocity",
    ):
        if cfg.get(key) is not None:
            _float(cfg, key, positive=True)
    if (cfg.get("min_frequency") or cfg["f_center"] / 5) >= (
        cfg.get("max_frequency") or 3 * cfg["f_center"]
    ):
        raise ValueError("min_frequency must be less than max_frequency")
    if cfg.get("sensor_sample_rate") is not None:
        try:
            rate = parse_quantity(cfg["sensor_sample_rate"], "frequency")
        except ValueError as exc:
            raise ValueError(f"'sensors.sample_rate': {exc}") from None
        if not rate > 0:
            raise ValueError(f"'sensors.sample_rate' must be positive, got {rate}")
        cfg["sensor_sample_rate"] = rate
    for key, minimum in (
        ("target_snapshots", 1),
        ("degree", 1),
    ):
        _int(cfg, key, minimum)

    summary = structure_from_config(cfg)
    if not summary["has_acoustic"] and cfg["absorb_length"] == 0:
        raise ValueError("absorb_length must be positive for the elastic solver")
    cfg["nx"], cfg["ny_per_region"] = mesh_resolution(cfg, summary)
    _resolve_layout_defaults(cfg, summary)
    _int(cfg, "n_sensors", 2)
    for key in ("source_fx", "source_fy", "traction_max"):
        _float(cfg, key)
    _resolve_source(cfg)
    _validate_acoustic_options(cfg, summary)
    for key in ("absorb_alpha", "absorb_beta"):
        if cfg[key] is not None:
            _float(cfg, key, nonneg=True)

    solver = cfg["solver_type"]
    beta, gamma = _float(cfg, "beta_val"), _float(cfg, "gamma_val")
    if solver == "implicit" and (gamma < 0.5 or beta < 0.25 * (gamma + 0.5) ** 2):
        raise ValueError("Newmark requires gamma >= 0.5 and beta >= (gamma+0.5)^2/4")

    length = 2 * cfg["absorb_length"] + cfg["sensing_length"]
    sensor_start = cfg["source_x"] + cfg["sensor_offset"]
    sensor_end = sensor_start + (cfg["n_sensors"] - 1) * cfg["sensor_pitch"]
    if sensor_start < 0 or sensor_end > length:
        raise ValueError(
            f"Sensor array [{sensor_start:.4g}, {sensor_end:.4g}] m must lie inside [0, {length:.4g}] m"
        )
    if cfg["run_postprocess"] and cfg["n_sensors"] < 3:
        raise ValueError("Postprocessing requires at least three sensors")

    if cfg.get("sensor_y") is not None:
        y = _float(cfg, "sensor_y")
        if not summary["y_min"] <= y <= summary["y_max"]:
            raise ValueError(
                f"sensor_y={y} must lie inside the domain [{summary['y_min']}, {summary['y_max']}]"
            )
    _resolve_automatic_values(cfg, summary)


def _apply_cli_overrides(resolved: dict[str, Any], args: argparse.Namespace) -> None:
    overrides = {
        "solver": "solver_type",
        "structure_mode": "structure_mode",
        "output": "output_dir",
        "f_min": "disp_f_min",
        "f_max": "disp_f_max",
        "k_min": "disp_k_min",
        "k_max": "disp_k_max",
        "min_frequency": "min_frequency",
        "max_frequency": "max_frequency",
        "duration": "duration",
        "dt": "dt",
        "cfl": "CFL",
        "device": "device",
        "precision": "precision",
    }
    legacy_for = {key: legacy for legacy, (key, _) in LEGACY_DISPERSION_KEYS.items()}
    for attr, key in overrides.items():
        value = getattr(args, attr, None)
        if value is not None:
            resolved[key] = value
            resolved.pop(legacy_for.get(key, ""), None)  # CLI wins over legacy file keys
    if (
        getattr(args, "max_frequency", None) is not None
        and getattr(args, "min_frequency", None) is None
    ):
        min_f = resolved.get("min_frequency")
        if min_f is not None:
            try:
                if float(min_f) >= resolved["max_frequency"]:
                    resolved["min_frequency"] = None
            except (ValueError, TypeError):
                pass
    if getattr(args, "no_postprocess", False):
        resolved["run_postprocess"] = False


def parse_and_resolve_config(
    config_path: str | None = None,
    args: argparse.Namespace | None = None,
    custom_cfg: dict[str, Any] | None = None,
    validate_simulation: bool = True,
) -> dict[str, Any]:
    """Load, flatten, apply defaults and CLI overrides, and validate a configuration.

    Precedence: CLI overrides > configuration file/dictionary > :data:`DEFAULTS`.
    """
    if custom_cfg is not None:
        raw_cfg = custom_cfg
        config_source = str(custom_cfg.get("_config_source", "Supplied Dictionary"))
    elif config_path is not None:
        raw_cfg = load_config(config_path)
        config_source = config_path
    elif args is not None and getattr(args, "config", None):
        config_path = args.config
        raw_cfg = load_config(config_path)
        config_source = config_path
    else:
        raise ValueError("A configuration file path or configuration dictionary must be provided.")

    resolved = flatten_config(raw_cfg)
    if args is not None:
        _apply_cli_overrides(resolved, args)
    for key, value in DEFAULTS.items():
        if resolved.get(key) is None:
            resolved[key] = (
                list(value)
                if isinstance(value, list)
                else dict(value)
                if isinstance(value, Mapping)
                else value
            )

    # Halfspace aliases
    for alias, canonical in (("substrate", "bottom_halfspace"), ("superstrate", "top_halfspace")):
        if resolved.get(alias) is not None:
            if resolved.get(canonical) is not None:
                raise ValueError(f"Specify only one of '{alias}' and '{canonical}'")
            resolved[canonical] = resolved.pop(alias)
        else:
            resolved.pop(alias, None)

    resolved["_config_source"] = config_source
    resolved["solver_type"] = str(resolved["solver_type"]).lower().strip()
    if resolved["solver_type"] not in ("implicit", "explicit"):
        raise ValueError("solver_type must be 'explicit' or 'implicit'")
    try:
        resolved["device"] = normalize_device_request(resolved["device"])
    except ValueError:
        raise ValueError(
            f"Invalid device '{resolved['device']}'. Choose from: {DEVICE_CHOICES_HELP}."
        ) from None
    resolved["precision"] = str(resolved["precision"]).lower().strip()
    if resolved["precision"] not in ("double", "single"):
        raise ValueError(f"precision must be 'double' or 'single', got '{resolved['precision']}'")
    resolved["structure_mode"] = canonicalize_mode(resolved["structure_mode"])

    _validate_output_options(resolved)
    from wavesimfem.characterisation import validate_options

    resolved["wave_characterisation"] = validate_options(resolved["wave_characterisation"])
    if not isinstance(resolved["lamb_modes"], bool):
        raise ValueError("postprocess.lamb_modes must be boolean")
    if validate_simulation:
        _validate_simulation(resolved, config_source)
    return resolved


# Backward-compatibility alias
resolve_and_apply_config = parse_and_resolve_config


TEMPLATE_CONFIG = """# ==============================================================================
# WaveSimFEM Simulation Configuration Template
# ==============================================================================

solver:
  type: "explicit"              # "explicit" (lumped-mass RK4) or "implicit" (Newmark-beta)
  CFL: 0.8                      # Required for explicit: fraction (0, 1] of the assembled stability certificate
  steps_per_period: 40          # Implicit only: Newmark steps per period at max_frequency
  time_factor: 1.2              # Travel-time margin plus source pulse length
  min_group_velocity: null     # [m/s] Independent bound for the modes/frequencies to retain
  # Automatic duration estimates A0 only for a free isotropic plate; it cannot
  # guarantee higher-mode arrivals near cutoffs. Set duration or this bound.
  beta: 0.25                    # Newmark-beta (implicit only)
  gamma: 0.5                    # Newmark-gamma (implicit only)
  dt: null                      # Optional maximum time step [s]; explicit stability checked after assembly
  duration: null                # Optional duration [s]; overrides time_factor
  device: "auto"                # Explicit, single-process only: auto | cpu | cuda | cuda:N
  precision: "double"           # Explicit device engine: "double" or "single"

geometry:
  sensing_length: 0.30          # [m] Pristine observation region length
  absorb_length: 0.15           # [m] Lateral sponge width; zero allowed for acoustic configurations

structure:
  # Modes: "1_layer", "N_layers", "halfspace", "plate_on_halfspace", "plate_in_halfspaces"
  mode: "1_layer"
  single_layer:
    name: "Aluminum Plate"
    thickness: 1.0e-3           # [m]
    material_model: "isotropic" # "isotropic", "anisotropic", "viscoelastic", or "fluid"
    rho: 2700.0                 # [kg/m^3]
    E: 70.0e9                   # [Pa]
    nu: 0.33                    # [-]
  # Fluid alternative (replace E/nu with c): material_model: fluid, rho: 1000, c: 1480.
  # fluid_formulation: acoustic is the default, with scalar pressure and zero shear.
  # To reproduce old runs, explicitly set fluid_formulation: regularized_elastic
  # and optionally fluid_shear_reg: 1e-8. Acoustic materials reject fluid_shear_reg.

absorbing_boundaries:
  alpha: null                   # [1/s] Peak mass-proportional damping (null = 1e7 * f_center / 1 MHz)
  beta: null                    # [s] Peak stiffness-proportional damping (null = 1e-9 * 1 MHz / f_center)

acoustic_boundaries:            # Exterior acoustic boundaries only
  top: "rigid"                 # rigid | pressure_release | radiation
  bottom: "rigid"
  left: "rigid"
  right: "rigid"

excitation:
  source_kind: "traction"       # traction | acoustic_pressure | acoustic_normal_acceleration | acoustic_volume
  source_amplitude: null        # Acoustic kinds require SI amplitude: Pa | m/s^2 | s^-2
  source_boundary: "auto"       # auto | top | bottom | left | right | interface
  source_y: null                # [m] Required with source_boundary: interface
  source_region: null           # Region name: required for acoustic_volume; selects an interface side
  source_type: "ricker"         # "ricker" (broadband) or "tone_burst" (Hann-windowed, narrowband)
  f_center: 1.0e6               # [Hz] Centre frequency
  n_cycles: null                # tone_burst only: number of cycles (null = 5)
  traction_max: 1.0e6           # [Pa] Peak traction amplitude
  # lambda_ref is the sizing wavelength at f_center: bulk/Rayleigh reference,
  # tightened by A0 for a free isotropic plate and by mesh.min_wavelength if set.
  excitation_width: null        # [m] sigma = width / 3; auto min(lambda_ref / 2, L_sensing / 10)
  source_x: null                # [m] Auto sponge clearance; left/right sources resolve to 0/Lx
  source_fx: 0.0                # Tangential traction direction component
  source_fy: 1.0                # Normal traction direction component (default: normal load)
  # Auto source_x = absorb_length + max(min(max(lambda_ref, width), L_sensing/8), width/2).

mesh:
  nx: 4000                     # Required exact longitudinal element count; verify mesh convergence
  ny_per_region: [8]           # Required exact counts, bottom-to-top (including any halfspaces)
  degree: 2                     # Lagrange polynomial order (Q2 quadrilaterals)
  # Frequency and wavelength controls below support time stepping, arrival
  # estimates, source/sensor defaults, and plots; they never change mesh counts.
  max_frequency: null           # [Hz] Analysis/time-step band edge (null = 3 * f_center)
  min_frequency: null           # [Hz] Arrival-sizing band edge (null = f_center / 5)
  min_wavelength: null          # [m] Conservative shortest retained-band wavelength

sensors:
  observable: "auto"            # auto | displacement | velocity | pressure | particle_velocity
  region: null                  # Material region name; selects a side for discontinuous particle velocity
  n_sensors: null               # Measurement points (null = fill the sensing region, max 1001)
  sensor_offset: null           # [m] Source to first sensor (null = min(2 lambda_ref, L_sensing / 4))
  # For a right-boundary source, automatic offsets instead place the array to its left.
  sensor_pitch: null            # [m] Auto min(lambda_ref/8, lambda_mesh/4, L_sensing/100)
  # lambda_mesh is the reference wavelength at mesh.max_frequency, not an element size.
  sensor_y: null                # [m] Sensor line height (null = loaded surface/interface)
  sample_rate: null             # [Hz] Stored sampling rate, e.g. "20 MHz" (null = 20 * max_frequency)

postprocess:
  structure_analysis: true     # Physics-aware multi-panel reports selected from archived structure
  run_postprocess: true         # 2D FFT dispersion analysis and figures
  # Plot limits: SI numbers (Hz, rad/m) or strings with units ("2 MHz", "3 rad/mm").
  f_min: null                   # null = 0
  f_max: null                   # null = 2 * f_center
  k_min: null                   # null = 0
  k_max: null                   # Auto sizing wavenumber at f_max, limited by spatial Nyquist

output:
  output_dir: null              # null = results_wavesim_{solver}
  write_vtx: false              # ADIOS2 VTX (.bp) export for ParaView
  target_snapshots: 200         # VTX snapshot count
  quick_plot: true              # Overview figure (disabled by --no-postprocess)
  compress_output: false        # Write sensor_data.npz with zlib compression
  history_storage: memory       # memory | disk (temporary maps; FFT postprocess still needs RAM)
  figure_formats: [png, pdf, svg, eps]
  usetex: false                 # true = external LaTeX; false = built-in serif/mathtext
"""


def generate_template_config(output_path: str) -> None:
    """Write the annotated YAML template configuration."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(TEMPLATE_CONFIG)
