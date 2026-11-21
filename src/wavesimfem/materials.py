"""Structural geometry, constitutive models, mesh layout, and preflight estimation.

Supports:
  - Isotropic, anisotropic, Kelvin-Voigt viscoelastic, and acoustic fluids
  - Explicitly selected legacy regularized-elastic fluids
  - Single plates, multilayer laminates, semi-infinite elastic halfspaces, and
    plate-substrate systems
  - A shared mesh/time layout used by both the preflight estimator and the solver
"""

from __future__ import annotations

import difflib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from wavesimfem.numerics import RK4_STABILITY_RADIUS, sample_stride, time_grid, validate_tensor
from wavesimfem.reporting import format_banner_box, log, warn
from wavesimfem.sources import describe_source, source_duration
from wavesimfem.storage import HISTORY_CHUNK_BYTES
from wavesimfem.units import fmt_frequency, fmt_length, fmt_time

STRUCTURE_MODES = (
    "1_layer",
    "N_layers",
    "halfspace",
    "plate_on_halfspace",
    "plate_in_halfspaces",
)
MATERIAL_MODELS = ("isotropic", "anisotropic", "viscoelastic", "fluid")
FLUID_SHEAR_RATIO = 0.01

_FLUID_SPEED_KEYS = ("c", "c_sound", "c_fluid", "speed")
_FLUID_BULK_KEYS = ("K", "bulk_modulus", "bulk_mod")
REGION_KEYS = frozenset(
    {
        "name",
        "thickness",
        "material_model",
        "rho",
        "E",
        "nu",
        "C",
        "Cv",
        "fluid_shear_reg",
        "fluid_formulation",
        "absorb_depth",
        *_FLUID_SPEED_KEYS,
        *_FLUID_BULK_KEYS,
    }
)

#: Constitutive fields accepted by each material model (others are rejected so
#: that parameters are never silently ignored).
_MODEL_FIELDS: dict[str, frozenset[str]] = {
    "isotropic": frozenset({"E", "nu"}),
    "anisotropic": frozenset({"C"}),
    "viscoelastic": frozenset({"C", "Cv"}),
    "fluid": frozenset(
        {"E", "nu", "fluid_formulation", "fluid_shear_reg", *_FLUID_SPEED_KEYS, *_FLUID_BULK_KEYS}
    ),
}
_CONSTITUTIVE_FIELDS = frozenset().union(*_MODEL_FIELDS.values())

DEFAULT_FLUID_SHEAR_REG = 1e-8

#: Default number of implicit (Newmark) steps per period at the analysis-band
#: frequency. Trapezoidal Newmark (beta = 1/4, gamma = 1/2) is unconditionally
#: stable; its relative period error is about (omega dt)^2 / 12, i.e. 0.2 % at
#: ``max_frequency`` and 0.05 % at half of it for 40 steps per period.
DEFAULT_STEPS_PER_PERIOD = 40


# ---------------------------------------------------------------------------
# Small validation helpers
# ---------------------------------------------------------------------------
def _suggest(key: str, allowed: set[str] | frozenset[str]) -> str:
    match = difflib.get_close_matches(key, sorted(allowed), n=1)
    return f" Did you mean '{match[0]}'?" if match else ""


def _region_float(item: Mapping[str, Any], key: str, name: str) -> float:
    if key not in item or item[key] is None:
        raise ValueError(f"Region '{name}' is missing required field '{key}'")
    raw = item[key]
    if isinstance(raw, (bool, np.bool_)):
        raise ValueError(f"Region '{name}' field '{key}' must be a number, got {raw!r}")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"Region '{name}' field '{key}' must be a number, got {raw!r}") from None
    if not np.isfinite(value):
        raise ValueError(f"Region '{name}' field '{key}' must be finite, got {value}")
    return value


def _region_voigt(item: Mapping[str, Any], key: str, name: str) -> list[float]:
    if key not in item or item[key] is None:
        raise ValueError(f"Region '{name}' is missing required field '{key}'")
    raw = item[key]
    if isinstance(raw, (str, bytes, Mapping)):
        raise ValueError(
            f"Region '{name}' field '{key}' must be a list of six numbers "
            "[X11, X12, X16, X22, X26, X66]"
        )
    try:
        values: list[float] = []
        for entry in raw:
            if isinstance(entry, (bool, np.bool_)):
                raise ValueError("boolean in Voigt tensor")
            val = float(entry)
            if not np.isfinite(val):
                raise ValueError("non-finite entry in Voigt tensor")
            values.append(val)
    except (TypeError, ValueError):
        raise ValueError(
            f"Region '{name}' field '{key}' must be a list of six finite numbers "
            "[X11, X12, X16, X22, X26, X66]"
        ) from None
    if len(values) != 6:
        raise ValueError(
            f"Region '{name}' field '{key}' must have 6 entries "
            f"[X11, X12, X16, X22, X26, X66], got {len(values)}"
        )
    return values


def canonicalize_mode(mode_str: str | None) -> str:
    """Normalize a user-provided structure mode to its canonical identifier.

    Raises
    ------
    ValueError
        If ``mode_str`` does not match a recognized structural category.
    """
    if mode_str is None:
        return "N_layers"
    s = str(mode_str).strip().lower().replace("-", "_").replace(" ", "_")
    if s in ("1_layer", "1layer", "single", "single_layer", "1", "one_layer"):
        return "1_layer"
    if s in ("n_layers", "nlayers", "multilayer", "multi_layer", "n"):
        return "N_layers"
    if s in ("halfspace", "half_space", "rayleigh", "single_halfspace", "halfspace_only"):
        return "halfspace"
    if any(
        k in s for k in ("on_halfspace", "over_halfspace", "over_a_halfspace", "on_a_halfspace")
    ):
        return "plate_on_halfspace"
    if any(
        k in s for k in ("between_2_halfspaces", "between_halfspaces", "in_halfspaces", "embedded")
    ):
        return "plate_in_halfspaces"
    raise ValueError(
        f"Unknown structure_mode '{mode_str}'. Choose from: {', '.join(STRUCTURE_MODES)}."
    )


def compute_rayleigh_speed(c_T: float, c_L: float | None = None, nu: float | None = None) -> float:
    """Rayleigh surface-wave speed of an isotropic solid.

    Solves Rayleigh's cubic ``ξ³ - 8ξ² + 8(3 - 2η)ξ - 16(1 - η) = 0`` with
    ``ξ = (c_R/c_T)²`` and ``η = (c_T/c_L)² = (1 - 2ν)/(2(1 - ν))``, returning the
    unique physical root in ``(0, 1)``. Falls back to Viktorov's approximation.
    """
    if nu is not None:
        eta = (1.0 - 2.0 * nu) / (2.0 * (1.0 - nu))
    elif c_L is not None:
        eta = (c_T / c_L) ** 2
    else:
        return 0.92 * c_T
    roots = np.roots([1.0, -8.0, 8.0 * (3.0 - 2.0 * eta), -16.0 * (1.0 - eta)])
    valid = [r.real for r in roots if abs(r.imag) < 1e-10 and 0.0 < r.real < 1.0]
    if valid:
        return float(c_T * np.sqrt(valid[0]))
    return float(c_T * (0.862 + 1.14 * nu) / (1.0 + nu)) if nu is not None else 0.92 * c_T


def _infer_model(item: Mapping[str, Any]) -> str:
    """Infer the constitutive model when ``material_model`` is omitted."""
    if "Cv" in item:
        return "viscoelastic"
    if "C" in item:
        return "anisotropic"
    if any(k in item for k in (*_FLUID_SPEED_KEYS, *_FLUID_BULK_KEYS)):
        return "fluid"
    if "E" in item and "nu" in item:
        return "isotropic"
    raise ValueError("cannot infer 'material_model'; please specify it explicitly")


def _normalize_region(
    item: Mapping[str, Any],
    default_name: str,
    role: str,
    is_halfspace: bool,
    halfspace_type: str | None,
    y_bot: float,
) -> dict[str, Any]:
    """Validate one subdomain region and precompute its material constants."""
    if not isinstance(item, Mapping):
        raise ValueError(f"{role} definition must be a mapping, got {type(item).__name__}")
    name = str(item.get("name", default_name))
    if "min_elems_y" in item:
        raise ValueError(
            f"Region '{name}': min_elems_y is no longer supported. Specify exact "
            "mesh.ny_per_region counts for all regions in bottom-to-top order."
        )
    unknown = set(item) - REGION_KEYS
    if unknown:
        key = sorted(unknown)[0]
        raise ValueError(
            f"Region '{name}' has unknown field(s) {sorted(unknown)}.{_suggest(key, REGION_KEYS)}"
        )
    if "absorb_depth" in item and not is_halfspace:
        raise ValueError(f"Region '{name}': 'absorb_depth' is only valid for halfspaces")

    h = _region_float(item, "thickness", name)
    if h <= 0:
        raise ValueError(f"Region '{name}' thickness must be positive, got {h}")
    rho = _region_float(item, "rho", name)
    if rho <= 0:
        raise ValueError(f"Region '{name}' density rho must be positive, got {rho}")

    raw_model = item.get("material_model")
    try:
        model = _infer_model(item) if raw_model is None else str(raw_model).lower().strip()
    except ValueError as exc:
        raise ValueError(f"Region '{name}': {exc}") from None
    if model in ("acoustic_fluid", "water", "acoustic"):
        model = "fluid"
    if model not in MATERIAL_MODELS:
        raise ValueError(
            f"Region '{name}' has unknown material_model '{raw_model}'. "
            f"Must be one of {', '.join(MATERIAL_MODELS)}."
        )
    ignored = sorted((set(item) & _CONSTITUTIVE_FIELDS) - _MODEL_FIELDS[model])
    if ignored:
        raise ValueError(
            f"Region '{name}': field(s) {ignored} are not used by material_model '{model}'"
            + ("; use 'viscoelastic' for Kelvin-Voigt damping (C + Cv)" if "Cv" in ignored else "")
        )

    entry: dict[str, Any] = {
        "name": name,
        "role": role,
        "thickness": h,
        "material_model": model,
        "rho": rho,
        "y_bot": y_bot,
        "y_top": y_bot + h,
        "is_halfspace": is_halfspace,
        "halfspace_type": halfspace_type,
        "is_fluid": model == "fluid",
        "is_acoustic": False,
        "is_legacy_fluid": False,
    }

    if model == "fluid":
        speed_keys = [k for k in _FLUID_SPEED_KEYS if k in item]
        bulk_keys = [k for k in _FLUID_BULK_KEYS if k in item]
        elastic_keys = [k for k in ("E", "nu") if k in item]
        if len(speed_keys) > 1 or len(bulk_keys) > 1:
            raise ValueError(f"Fluid '{name}': duplicate aliases {speed_keys + bulk_keys}")
        if sum(bool(keys) for keys in (speed_keys, bulk_keys, elastic_keys)) != 1:
            raise ValueError(
                f"Fluid '{name}': specify exactly one of sound speed, bulk modulus, or E and nu"
            )
        if elastic_keys and len(elastic_keys) != 2:
            raise ValueError(f"Fluid '{name}': E and nu must be supplied together")
        speed_key = next((k for k in _FLUID_SPEED_KEYS if item.get(k) is not None), None)
        bulk_key = next((k for k in _FLUID_BULK_KEYS if item.get(k) is not None), None)
        if speed_key is not None:
            c_sound = _region_float(item, speed_key, name)
            if c_sound <= 0:
                raise ValueError(f"Fluid '{name}' sound speed must be positive, got {c_sound}")
            K_val = rho * c_sound**2
        elif bulk_key is not None:
            K_val = _region_float(item, bulk_key, name)
            if K_val <= 0:
                raise ValueError(f"Fluid '{name}' bulk modulus must be positive, got {K_val}")
            c_sound = float(np.sqrt(K_val / rho))
        elif "E" in item and "nu" in item:
            E = _region_float(item, "E", name)
            nu = _region_float(item, "nu", name)
            if E <= 0 or not -1 < nu < 0.5:
                raise ValueError(f"Fluid '{name}' requires E > 0 and -1 < nu < 0.5")
            K_val = E / (3.0 * (1.0 - 2.0 * nu))
            c_sound = float(np.sqrt(K_val / rho))
        else:
            raise ValueError(f"Fluid '{name}' requires a sound speed 'c' or bulk modulus 'K'")

        formulation = str(item.get("fluid_formulation", "acoustic")).lower().strip()
        if formulation not in ("acoustic", "regularized_elastic"):
            raise ValueError(
                f"Fluid '{name}' fluid_formulation must be 'acoustic' or 'regularized_elastic'"
            )
        if formulation == "acoustic" and "fluid_shear_reg" in item:
            raise ValueError(
                f"Fluid '{name}': fluid_shear_reg is unused by acoustic pressure physics; "
                "remove it or explicitly set fluid_formulation: regularized_elastic"
            )
        entry.update(
            c_fluid=c_sound,
            c_L=c_sound,
            K=K_val,
            c_physical_min=c_sound,
            fluid_formulation=formulation,
            is_acoustic=formulation == "acoustic",
            is_legacy_fluid=formulation == "regularized_elastic",
        )
        if formulation == "regularized_elastic":
            fluid_shear_reg = _region_float(
                {"fluid_shear_reg": item.get("fluid_shear_reg", DEFAULT_FLUID_SHEAR_REG)},
                "fluid_shear_reg",
                name,
            )
            if not 0 < fluid_shear_reg <= 1e-2:
                raise ValueError(
                    f"Fluid '{name}' fluid_shear_reg must be in (0, 0.01], got {fluid_shear_reg}"
                )
            mu = fluid_shear_reg * K_val
            lmbda = K_val - (2.0 / 3.0) * mu
            nu_eff = lmbda / (2.0 * (lmbda + mu))
            entry.update(
                fluid_shear_reg=fluid_shear_reg,
                E=2.0 * mu * (1.0 + nu_eff),
                nu=nu_eff,
                mu=mu,
                lmbda=lmbda,
                c_L=float(np.sqrt((lmbda + 2 * mu) / rho)),
                c_T=float(np.sqrt(mu / rho)),
            )
    elif model == "isotropic":
        E = _region_float(item, "E", name)
        nu = _region_float(item, "nu", name)
        if E <= 0 or not -1.0 < nu < 0.5:
            raise ValueError(f"Region '{name}': require finite E > 0 and -1 < nu < 0.5")
        mu = E / (2.0 * (1.0 + nu))
        lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        c_T = float(np.sqrt(mu / rho))
        entry.update(
            E=E,
            nu=nu,
            mu=mu,
            lmbda=lmbda,
            c_L=float(np.sqrt((lmbda + 2.0 * mu) / rho)),
            c_T=c_T,
            c_physical_min=c_T,
        )
    else:
        C = _region_voigt(item, "C", name)
        C_mat = validate_tensor(C, f"Region '{name}' C")
        # Bounds from the orthonormal (Kelvin) strain representation.
        scale = np.diag([1.0, 1.0, np.sqrt(2.0)])
        eig = np.linalg.eigvalsh(scale @ C_mat @ scale)
        c_T = float(np.sqrt(eig[0] / (2 * rho)))
        entry.update(
            C=C,
            C_mat=C_mat,
            c_L=float(np.sqrt(eig[-1] / rho)),
            c_T=c_T,
            c_physical_min=c_T,
        )
        if model == "viscoelastic":
            Cv = _region_voigt(item, "Cv", name)
            entry.update(Cv=Cv, Cv_mat=validate_tensor(Cv, f"Region '{name}' Cv", positive=False))
    return entry


def _halfspace_absorb_depth(
    cfg: Mapping[str, Any], h: float, default_frac: float, label: str
) -> float:
    raw = cfg.get("absorb_depth")
    if raw is None:
        raw = default_frac * h
    if isinstance(raw, (bool, np.bool_)):
        raise ValueError(f"{label} absorb_depth must be a finite number in (0, {h}], got {raw!r}")
    try:
        absorb_depth = float(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"{label} absorb_depth must be a finite number in (0, {h}], got {raw!r}"
        ) from None
    if not np.isfinite(absorb_depth) or absorb_depth <= 0 or absorb_depth > h:
        raise ValueError(f"{label} absorb_depth must be in (0, {h}], got {absorb_depth}")
    return absorb_depth


def _speed_summary(regions: list[dict[str, Any]], prefix: str) -> dict[str, float | None]:
    solids = [r for r in regions if not r.get("is_fluid", False)]
    shear_regions = solids or [r for r in regions if r.get("is_legacy_fluid", False)]
    return {
        f"c_T_{prefix}min": min((r["c_T"] for r in shear_regions), default=None),
        f"c_T_{prefix}max": max((r["c_T"] for r in shear_regions), default=None),
        f"c_L_{prefix}min": min(r["c_L"] for r in regions),
        f"c_L_{prefix}max": max(r["c_L"] for r in regions),
    }


def normalize_structure(
    mode: str | None = None,
    single_layer_input: dict[str, Any] | None = None,
    layers_input: list[dict[str, Any]] | None = None,
    halfspace_input: dict[str, Any] | None = None,
    bottom_hs_input: dict[str, Any] | None = None,
    top_hs_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize plates, a pristine halfspace, or plate-halfspace systems.

    Returns a summary dictionary with all regions ordered bottom-to-top, the
    domain extents, and wave-speed bounds used for mesh and time-step sizing.
    """
    mode = canonicalize_mode(mode or "N_layers")

    if mode == "halfspace":
        hs_cfg = halfspace_input or bottom_hs_input
        if hs_cfg is None:
            raise ValueError(
                "structure.mode is 'halfspace', but 'halfspace' (or 'substrate') is not defined."
            )
        h_hs = _region_float(hs_cfg, "thickness", str(hs_cfg.get("name", "Halfspace")))
        norm_hs = _normalize_region(hs_cfg, "Halfspace", "Halfspace", True, "bottom", -h_hs)
        norm_hs["absorb_depth"] = _halfspace_absorb_depth(hs_cfg, h_hs, 0.6, "Halfspace")
        norm_hs["tag"] = 1
        is_solid_iso = norm_hs["material_model"] == "isotropic"
        c_R = (
            compute_rayleigh_speed(norm_hs["c_T"], c_L=norm_hs["c_L"], nu=norm_hs["nu"])
            if is_solid_iso
            else None
        )
        norm_hs["c_R"] = c_R
        speeds = _speed_summary([norm_hs], "")
        return {
            "structure_mode": mode,
            "all_regions": [norm_hs],
            "plate_layers": [],
            "bottom_halfspace": norm_hs,
            "top_halfspace": None,
            "has_bottom_halfspace": True,
            "has_top_halfspace": False,
            "H_plate": 0.0,
            "y_min": -h_hs,
            "y_max": 0.0,
            "H_domain": h_hs,
            **speeds,
            "c_wave_min": norm_hs["c_physical_min"],
            "has_fluid": norm_hs["is_fluid"],
            "has_acoustic": norm_hs["is_acoustic"],
            "has_legacy_fluid": norm_hs["is_legacy_fluid"],
            "c_fluid_min": norm_hs.get("c_fluid"),
            "c_R": c_R,
            "c_T_plate_min": norm_hs.get("c_T"),
            "c_T_plate_max": norm_hs.get("c_T"),
            "c_L_plate_min": norm_hs["c_L"],
            "c_L_plate_max": norm_hs["c_L"],
            "c_L_plate_top": norm_hs["c_L"],
            "c_T_plate_top": norm_hs.get("c_T"),
            "rho_plate_top": norm_hs["rho"],
        }

    if layers_input is not None and not isinstance(layers_input, list):
        raise ValueError("'layers' must be a list of region definitions")
    if mode == "1_layer":
        if single_layer_input is not None:
            plate_cfg = [single_layer_input]
        elif layers_input:
            if len(layers_input) != 1:
                raise ValueError(
                    f"structure.mode '1_layer' requires exactly one layer, got "
                    f"{len(layers_input)}; use mode 'N_layers' for laminates."
                )
            plate_cfg = list(layers_input)
        else:
            raise ValueError("structure.mode is '1_layer', but 'single_layer' is not defined.")
    elif layers_input:
        plate_cfg = list(layers_input)
    elif single_layer_input is not None:
        plate_cfg = [single_layer_input]
    else:
        raise ValueError(f"structure.mode is '{mode}', but the 'layers' list is not defined.")

    bot_hs_dict = top_hs_dict = None
    if mode in ("plate_on_halfspace", "plate_in_halfspaces"):
        bot_hs_dict = bottom_hs_input or halfspace_input
        if bot_hs_dict is None:
            raise ValueError(
                f"structure.mode is '{mode}', but 'substrate' (or 'bottom_halfspace') is not defined."
            )
    if mode == "plate_in_halfspaces":
        top_hs_dict = top_hs_input
        if top_hs_dict is None:
            raise ValueError(
                "structure.mode is 'plate_in_halfspaces', but 'superstrate' "
                "(or 'top_halfspace') is not defined."
            )

    norm_plate_layers = []
    y_curr = 0.0
    for idx, lyr in enumerate(plate_cfg):
        role = f"Plate L{idx + 1}" if len(plate_cfg) > 1 else "Plate (Single)"
        entry = _normalize_region(lyr, f"Layer {idx + 1}", role, False, None, y_curr)
        y_curr += entry["thickness"]
        norm_plate_layers.append(entry)
    H_plate = y_curr

    norm_bot_hs = norm_top_hs = None
    if bot_hs_dict is not None:
        h_sub = _region_float(
            bot_hs_dict, "thickness", str(bot_hs_dict.get("name", "Bottom Substrate"))
        )
        norm_bot_hs = _normalize_region(
            bot_hs_dict, "Bottom Substrate", "Substrate", True, "bottom", -h_sub
        )
        norm_bot_hs["absorb_depth"] = _halfspace_absorb_depth(
            bot_hs_dict, h_sub, 0.75, "Bottom halfspace"
        )
    if top_hs_dict is not None:
        h_sup = _region_float(
            top_hs_dict, "thickness", str(top_hs_dict.get("name", "Top Superstrate"))
        )
        norm_top_hs = _normalize_region(
            top_hs_dict, "Top Superstrate", "Superstrate", True, "top", H_plate
        )
        norm_top_hs["absorb_depth"] = _halfspace_absorb_depth(
            top_hs_dict, h_sup, 0.75, "Top halfspace"
        )

    all_regions = [r for r in (norm_bot_hs, *norm_plate_layers, norm_top_hs) if r is not None]
    for idx, reg in enumerate(all_regions):
        reg["tag"] = idx + 1

    y_min = all_regions[0]["y_bot"]
    y_max = all_regions[-1]["y_top"]
    top_plate = norm_plate_layers[-1]
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
        "H_domain": y_max - y_min,
        **_speed_summary(all_regions, ""),
        "c_wave_min": min(r["c_physical_min"] for r in all_regions),
        "has_fluid": any(r["is_fluid"] for r in all_regions),
        "has_acoustic": any(r["is_acoustic"] for r in all_regions),
        "has_legacy_fluid": any(r["is_legacy_fluid"] for r in all_regions),
        "c_fluid_min": min((r["c_fluid"] for r in all_regions if r["is_fluid"]), default=None),
        **_speed_summary(norm_plate_layers, "plate_"),
        "c_L_plate_top": top_plate["c_L"],
        "c_T_plate_top": top_plate.get("c_T"),
        "rho_plate_top": top_plate["rho"],
    }


def structure_from_config(cfg: Mapping[str, Any], mode: str | None = None) -> dict[str, Any]:
    """Normalize the structure described by a resolved (flat) configuration."""
    return normalize_structure(
        mode=mode or cfg.get("structure_mode", "N_layers"),
        single_layer_input=cfg.get("single_layer"),
        layers_input=cfg.get("layers"),
        halfspace_input=cfg.get("halfspace"),
        bottom_hs_input=cfg.get("bottom_halfspace"),
        top_hs_input=cfg.get("top_halfspace"),
    )


# ---------------------------------------------------------------------------
# Mesh / time layout shared by the solver and the preflight estimator
# ---------------------------------------------------------------------------
@lru_cache(maxsize=256)
def a0_wavenumber(frequency: float, thickness: float, c_L: float, c_T: float) -> float:
    """Fundamental antisymmetric Rayleigh–Lamb root for a free isotropic plate.

    Solve the real, sub-shear branch using half-thickness-scaled variables.
    At very small/large frequency-thickness use its thin-plate/Rayleigh limit,
    avoiding cancellation of the characteristic equation in those limits.
    """
    from scipy.optimize import brentq

    if not all(np.isfinite(v) and v > 0 for v in (frequency, thickness, c_L, c_T)) or c_L <= c_T:
        raise ValueError("A0 requires positive frequency/thickness and c_L > c_T > 0")
    half = thickness / 2
    omega = 2 * np.pi * frequency
    s = omega * half / c_T
    ratio = c_T / c_L
    rayleigh = compute_rayleigh_speed(c_T, c_L=c_L)
    if s > 30:
        return omega / rayleigh
    # omega = B k², B = h*c_T*sqrt((1-c_T²/c_L²)/3)
    bending = thickness * c_T * np.sqrt((1 - ratio**2) / 3)
    thin = np.sqrt(omega / bending)
    if s < 1e-5:
        return thin

    def residual(x: float) -> float:
        p = np.sqrt(x * x - (s * ratio) ** 2)
        q = np.sqrt(x * x - s * s)
        return np.tanh(q) / np.tanh(p) - (x * x + q * q) ** 2 / (4 * x * x * p * q)

    lower = s * (1 + 1e-9)
    upper = max(2 * thin * half, 2 * s * c_T / rayleigh)
    return brentq(residual, lower, upper, xtol=1e-13, rtol=1e-12) / half


def _free_isotropic_plate(summary: Mapping[str, Any]) -> Mapping[str, Any] | None:
    regions = summary["all_regions"]
    if (
        len(regions) == 1
        and not regions[0]["is_halfspace"]
        and regions[0]["material_model"] == "isotropic"
    ):
        return regions[0]
    return None


def sizing_wavelength(
    cfg: Mapping[str, Any], summary: Mapping[str, Any], frequency: float
) -> float:
    """Shortest sizing wavelength, including A0 for a free isotropic plate.

    ``min_wavelength`` is a user-supplied conservative bound across the entire
    retained band, also used for sensor sampling and dispersion display.
    """
    wavelength = reference_wave_speed(summary) / frequency
    plate = _free_isotropic_plate(summary)
    if plate is not None:
        wavelength = min(
            wavelength,
            2 * np.pi / a0_wavenumber(frequency, plate["thickness"], plate["c_L"], plate["c_T"]),
        )
    if cfg.get("min_wavelength") is not None:
        wavelength = min(wavelength, float(cfg["min_wavelength"]))
    return wavelength


def sizing_group_velocity(cfg: Mapping[str, Any], summary: Mapping[str, Any]) -> float:
    """A0/reference packet-speed estimate over the arrival-sizing band.

    This is not an all-mode lower bound: higher Lamb modes can have arbitrarily
    small group velocities near their cutoffs. ``min_group_velocity`` supplies
    an independent lower bound for the particular modes/frequencies of interest.
    """
    speed = reference_wave_speed(summary)
    plate = _free_isotropic_plate(summary)
    if plate is not None:
        high = float(cfg.get("max_frequency") or 3 * float(cfg["f_center"]))
        low = float(cfg.get("min_frequency") or min(float(cfg["f_center"]) / 5, 0.5 * high))
        for frequency in np.geomspace(low, high, 32):
            delta = frequency * 1e-3
            args = (plate["thickness"], plate["c_L"], plate["c_T"])
            derivative = (
                a0_wavenumber(frequency + delta, *args) - a0_wavenumber(frequency - delta, *args)
            ) / (2 * delta)
            speed = min(speed, 2 * np.pi / derivative)
    if cfg.get("min_group_velocity") is not None:
        speed = min(speed, float(cfg["min_group_velocity"]))
    return speed


def duration_sizing_notes(
    cfg: Mapping[str, Any], summary: Mapping[str, Any]
) -> tuple[str, list[str]]:
    """Describe the arrival-window contract and identify unsupported guarantees.

    A finite recording window cannot capture every packet at a mode cutoff.
    User-provided duration or group velocity expresses the selected recording
    target; an automatic A0 window is only an estimate for that branch.
    """
    if cfg.get("duration") is not None:
        return "user-specified duration", []
    if cfg.get("min_group_velocity") is not None:
        return "user-supplied group-velocity bound with A0/reference estimate", []
    plate = _free_isotropic_plate(summary)
    if plate is not None:
        # First nonzero thickness resonance: a shear half-wave across the full
        # plate thickness (the A1 cutoff); c_L > c_T for a stable isotropic solid.
        cutoff = plate["c_T"] / (2 * plate["thickness"])
        high = float(cfg.get("max_frequency") or 3 * float(cfg["f_center"]))
        warnings = []
        if high >= cutoff:
            warnings.append(
                "Automatic duration estimates A0 arrivals only. The arrival-sizing band "
                f"extends to {fmt_frequency(high)}, reaching higher Lamb modes "
                f"(first cutoff {fmt_frequency(cutoff)}). Their group velocities can "
                "approach zero near cutoffs, so this window may truncate their arrivals. "
                "Set solver.duration or solver.min_group_velocity from the modes and "
                "frequencies you intend to retain; no finite window covers every cutoff."
            )
        return "A0/reference estimate over the arrival-sizing band", warnings
    if summary["structure_mode"] != "halfspace":
        return "bulk/reference estimate", [
            "Automatic duration uses a bulk/reference speed and does not bound guided-mode "
            "arrivals for this structure. Set solver.duration or solver.min_group_velocity "
            "from an independent dispersion calculation for the modes and frequencies "
            "you intend to retain."
        ]
    return "halfspace reference-speed estimate", []


@dataclass(frozen=True)
class MeshLayout:
    """Exact configured mesh counts and their resulting physical cell dimensions.

    ``mesh_frequency`` is the analysis/time-sampling band retained for API
    compatibility; it never determines the mesh counts.
    """

    mesh_frequency: float
    h_x: float
    h_y_per_region: list[float]
    plate_length: float
    nx: int
    ny_per_region: list[int]
    h_min: float
    duration: float

    @property
    def ny(self) -> int:
        return sum(self.ny_per_region)


def mesh_resolution(cfg: Mapping[str, Any], summary: Mapping[str, Any]) -> tuple[int, list[int]]:
    """Validate explicit element counts without selecting or refining a mesh."""

    def count(value: Any, name: str) -> int:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
            or value < 1
            or int(value) != value
        ):
            raise ValueError(f"'{name}' must be a positive integer, got {value!r}")
        return int(value)

    if cfg.get("nx") is None or cfg.get("ny_per_region") is None:
        raise ValueError(
            "Specify mesh.nx and mesh.ny_per_region explicitly; automatic mesh sizing "
            "is no longer supported."
        )
    nx = count(cfg["nx"], "mesh.nx")
    values = cfg["ny_per_region"]
    if not isinstance(values, list):
        raise ValueError("'mesh.ny_per_region' must be a list of positive integers")
    regions = summary["all_regions"]
    if len(values) != len(regions):
        names = ", ".join(region["name"] for region in regions)
        raise ValueError(
            f"'mesh.ny_per_region' must have {len(regions)} entries in bottom-to-top "
            f"order ({names}); got {len(values)}"
        )
    return nx, [count(value, f"mesh.ny_per_region[{i}]") for i, value in enumerate(values)]


def mesh_vertical_tolerance(cfg: Mapping[str, Any], summary: Mapping[str, Any]) -> float:
    """Geometric tolerance shared by sensor selection and mesh assembly."""
    _, rows = mesh_resolution(cfg, summary)
    return min(r["thickness"] / n for r, n in zip(summary["all_regions"], rows, strict=True)) * 1e-6


def sensor_region_candidates(
    cfg: Mapping[str, Any], summary: Mapping[str, Any]
) -> tuple[float, list[dict[str, Any]]]:
    """Resolve receiver height and touching regions using the mesh tolerance."""
    y = cfg.get("sensor_y")
    y = (
        float(summary["H_plate"] if summary["has_top_halfspace"] else summary["y_max"])
        if y is None
        else float(y)
    )
    tol = mesh_vertical_tolerance(cfg, summary)
    candidates = [
        r
        for r in summary["all_regions"]
        if r["y_bot"] - tol <= y <= r["y_top"] + tol
        and (cfg.get("sensor_region") is None or r["name"] == cfg["sensor_region"])
    ]
    return y, candidates


def resolve_sensor_region(
    cfg: Mapping[str, Any], summary: Mapping[str, Any]
) -> tuple[float, str, dict[str, Any]]:
    """Validate the receiver observable and select its material side."""
    y, candidates = sensor_region_candidates(cfg, summary)
    if not candidates:
        raise ValueError(
            f"sensor_region {cfg['sensor_region']!r} does not contain or touch sensor_y={y:g}"
            if cfg.get("sensor_region") is not None
            else f"No region at sensor_y={y:g}"
        )
    observable = cfg.get("sensor_observable", "auto")
    if observable == "auto":
        observable = (
            "pressure" if all(r.get("is_acoustic", False) for r in candidates) else "displacement"
        )
    compatible = [
        r
        for r in candidates
        if r.get("is_acoustic", False) == (observable in ("pressure", "particle_velocity"))
    ]
    if not compatible:
        raise ValueError(f"No region at sensor_y={y:g} supports sensor_observable {observable!r}")
    if (
        len(compatible) > 1
        and observable == "particle_velocity"
        and cfg.get("sensor_region") is None
    ):
        raise ValueError(
            "Particle velocity at a fluid-fluid interface requires sensors.region to select a side"
        )
    return y, observable, compatible[0]


def compute_mesh_layout(cfg: Mapping[str, Any], summary: Mapping[str, Any]) -> MeshLayout:
    """Use the exact user-supplied mesh, preserving every material interface."""
    f_center = float(cfg["f_center"])
    mesh_frequency = float(cfg.get("max_frequency") or 3.0 * f_center)
    plate_length = 2.0 * float(cfg["absorb_length"]) + float(cfg["sensing_length"])
    nx, ny_per_region = mesh_resolution(cfg, summary)
    regions = summary["all_regions"]
    h_x = plate_length / nx
    h_y_per_region = [r["thickness"] / n for r, n in zip(regions, ny_per_region, strict=True)]
    h_min = min(h_x, *h_y_per_region)
    if cfg.get("duration") is not None:
        duration = float(cfg["duration"])
    else:
        speed_for_T = sizing_group_velocity(cfg, summary)
        first = float(cfg.get("sensor_offset") or 0)
        last = first + (int(cfg.get("n_sensors") or 1) - 1) * float(cfg.get("sensor_pitch") or 0)
        travel_distance = max(float(cfg["sensing_length"]), abs(first), abs(last))
        pulse = source_duration(
            str(cfg.get("source_type") or "ricker"), f_center, cfg.get("n_cycles")
        )
        duration = travel_distance / speed_for_T * float(cfg["time_factor"]) + pulse
    return MeshLayout(
        mesh_frequency=mesh_frequency,
        h_x=h_x,
        h_y_per_region=h_y_per_region,
        plate_length=plate_length,
        nx=nx,
        ny_per_region=ny_per_region,
        h_min=h_min,
        duration=duration,
    )


def reference_wave_speed(summary: Mapping[str, Any]) -> float:
    """Bulk reference speed, using ``c_R`` for a halfspace when available.

    Sizing helpers tighten this reference with A0 dispersion and user-supplied
    conservative bounds; the reference alone does not bound all guided modes.
    """
    return float(summary.get("c_R") or summary["c_wave_min"])


def implicit_time_step(mesh_frequency: float, steps_per_period: float) -> float:
    """Accuracy-driven Newmark step ``1 / (steps_per_period · max_frequency)``.

    Trapezoidal Newmark is unconditionally stable, so the step is set by temporal
    accuracy at the highest resolved frequency only. It is deliberately independent
    of the smallest element: thin layers (adhesives, coatings) no longer force tiny
    steps.
    """
    if not (mesh_frequency > 0 and steps_per_period > 0):
        raise ValueError("mesh_frequency and steps_per_period must be positive")
    return 1.0 / (steps_per_period * mesh_frequency)


def _voigt_stiffness(region: Mapping[str, Any]) -> np.ndarray:
    if region["material_model"] in ("isotropic", "fluid"):
        lam, mu = float(region["lmbda"]), float(region["mu"])
        return np.array([[lam + 2 * mu, lam, 0.0], [lam, lam + 2 * mu, 0.0], [0.0, 0.0, mu]])
    return np.asarray(region["C_mat"], dtype=float)


def _element_integrands(
    hx: float, hy: float, degree: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quadrature weights, scalar shape values and strain operators on an ``hx × hy`` cell.

    Uses the same Lagrange variant (GLL-warped) as the DOLFINx ``("Q", p)`` space.
    """
    import basix

    element = basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.quadrilateral,
        degree,
        basix.LagrangeVariant.gll_warped,
    )
    points, weights = basix.make_quadrature(basix.CellType.quadrilateral, 2 * degree + 1)
    table = element.tabulate(1, points)
    shape = table[0, :, :, 0]
    d_dx, d_dy = table[1, :, :, 0] / hx, table[2, :, :, 0] / hy
    n_q, n_nodes = shape.shape
    strain = np.zeros((n_q, 3, 2 * n_nodes))  # [e_xx, e_yy, 2 e_xy], interleaved (u_x, u_y)
    strain[:, 0, 0::2] = d_dx
    strain[:, 1, 1::2] = d_dy
    strain[:, 2, 0::2] = d_dy
    strain[:, 2, 1::2] = d_dx
    return weights * hx * hy, shape, strain


def _tensor_row_sums(
    weights: np.ndarray, strain: np.ndarray, tensor: np.ndarray, *, absolute_integrand: bool
) -> np.ndarray:
    """Absolute row sums of ``∫ Bᵀ D B``; optionally of ``∫ |Bᵀ D B|`` (entry-wise bound)."""
    per_q = np.einsum("qai,ab,qbj->qij", strain, tensor, strain)
    if absolute_integrand:
        return np.einsum("q,qij->i", weights, np.abs(per_q))
    return np.abs(np.einsum("q,qij->ij", weights, per_q)).sum(axis=1)


def element_operator_bounds(
    cfg: Mapping[str, Any], summary: Mapping[str, Any], layout: MeshLayout
) -> tuple[float, float]:
    """Uncoupled element bounds ``B_K`` and ``B_C`` before meshing.

    With acoustic regions these omit interface coupling and are advisory only.

    For row-sum lumping, ``Σ_j |Σ_e Aᵉ_ij| / Σ_e mᵉ_i ≤ max_e Σ_j |Aᵉ_ij| / mᵉ_i``,
    so element-level bounds of each region's (uniform) cell bound the solver's
    assembled values from above. The sponge indicator is bounded by 1, or by 2
    in halfspace regions where lateral and vertical sponges overlap; integrands
    are bounded entry-wise by their absolute values.

    Returns
    -------
    (B_K, B_C)
        Bounds of ``ρ(M_L⁻¹K)`` [rad²/s²] and ``ρ(M_L⁻¹C)`` [1/s].
    """
    degree = int(cfg.get("degree", 2))
    alpha = float(cfg.get("absorb_alpha") or 0.0)
    beta = float(cfg.get("absorb_beta") or 0.0)
    hx = layout.plate_length / layout.nx
    bound_k = bound_c = 0.0
    regions = summary["all_regions"]
    for region, ny in zip(regions, layout.ny_per_region, strict=True):
        weights, shape, strain = _element_integrands(hx, region["thickness"] / ny, degree)
        if region.get("is_acoustic", False):
            factor = 1.0 / float(region["K"])
            mass_rows = factor * np.einsum("q,qi,qj->i", weights, shape, shape)
            abs_mass_rows = factor * np.einsum("q,qi,qj->i", weights, np.abs(shape), np.abs(shape))
            gradient = np.stack((strain[:, 0, 0::2], strain[:, 1, 1::2]), axis=1)
            scalar_tensor = np.eye(2) / float(region["rho"])
            k_rows = _tensor_row_sums(weights, gradient, scalar_tensor, absolute_integrand=False)
            abs_k_rows = _tensor_row_sums(weights, gradient, scalar_tensor, absolute_integrand=True)
        else:
            rho = float(region["rho"])
            mass_rows = np.repeat(rho * np.einsum("q,qi,qj->i", weights, shape, shape), 2)
            abs_mass_rows = np.repeat(
                rho * np.einsum("q,qi,qj->i", weights, np.abs(shape), np.abs(shape)), 2
            )
            stiffness = _voigt_stiffness(region)
            k_rows = _tensor_row_sums(weights, strain, stiffness, absolute_integrand=False)
            abs_k_rows = _tensor_row_sums(weights, strain, stiffness, absolute_integrand=True)
        bound_k = max(bound_k, float(np.max(k_rows / mass_rows)))
        indicator = 2.0 if region.get("is_halfspace") else 1.0
        damping_rows = indicator * (alpha * abs_mass_rows + beta * abs_k_rows)
        if region["material_model"] == "viscoelastic":  # exact: Cv is not indicator-weighted
            damping_rows = damping_rows + _tensor_row_sums(
                weights, strain, np.asarray(region["Cv_mat"]), absolute_integrand=False
            )
        bound_c = max(bound_c, float(np.max(damping_rows / mass_rows)))
    return bound_k, bound_c


def explicit_time_step_estimate(
    cfg: Mapping[str, Any], summary: Mapping[str, Any], layout: MeshLayout
) -> float:
    """Pre-assembly explicit step ``CFL · R / max(√B_K, B_C)`` from element bounds.

    For elastic-only models this is a conservative certificate. Acoustic models
    require a coupled assembled check; this estimate has no stability guarantee.
    """
    bound_k, bound_c = element_operator_bounds(cfg, summary, layout)
    return float(cfg["CFL"]) * RK4_STABILITY_RADIUS / max(np.sqrt(bound_k), bound_c)


#: Minimum number of time samples required by the FFT post-processing.
MIN_POSTPROCESS_SAMPLES = 32
#: Default sensor sampling rate in multiples of the analysis maximum frequency.
AUTO_SAMPLES_PER_MESH_PERIOD = 20


def sampled_time_grid(
    cfg: Mapping[str, Any], layout: MeshLayout, requested_dt: float
) -> tuple[float, int, int]:
    """Shared solver/preflight grid with uniform samples ending at the duration.

    Round the step count up to a multiple of the recording stride. This can only
    reduce the requested maximum step, preserving its stability/accuracy limit.
    """
    dt, n_steps = time_grid(layout.duration, requested_dt)
    min_samples = MIN_POSTPROCESS_SAMPLES if cfg.get("run_postprocess", True) else 2
    if cfg.get("run_postprocess", True) and n_steps + 1 < min_samples:
        raise ValueError(
            f"Postprocessing requires at least {min_samples} time samples "
            f"(got {n_steps + 1}); reduce dt or disable postprocessing"
        )
    stride = sample_stride(
        dt,
        n_steps,
        float(
            cfg.get("sensor_sample_rate") or AUTO_SAMPLES_PER_MESH_PERIOD * layout.mesh_frequency
        ),
        # Keep the whole meshed band alias-free, not only the source band.
        f_max=max(
            float(cfg.get("disp_f_max") or 2 * cfg["f_center"]),
            3 * cfg["f_center"],
            float(layout.mesh_frequency),
        ),
        min_samples=min_samples,
    )
    if n_steps % stride:
        n_steps = -(-n_steps // stride) * stride
        dt = layout.duration / n_steps
    return dt, n_steps, stride


def restricted_field_dof_counts(
    summary: Mapping[str, Any], layout: MeshLayout, degree: int
) -> dict[str, int]:
    """Exact unconstrained Q-degree field counts on the layered conforming mesh.

    Adjacent regions with the same field share their interface nodes. Solid and
    pressure fields have separate interface nodes; separated components retain
    their own rows, without unused degrees of freedom in other material cells.
    """
    row_counts = {"displacement": 0, "pressure": 0}
    previous_field = None
    for region, ny in zip(summary["all_regions"], layout.ny_per_region, strict=True):
        field = "pressure" if region.get("is_acoustic", False) else "displacement"
        row_counts[field] += degree * ny + int(field != previous_field)
        previous_field = field
    nx_nodes = degree * layout.nx + 1
    return {
        "displacement": 2 * nx_nodes * row_counts["displacement"],
        "pressure": nx_nodes * row_counts["pressure"],
    }


def preflight_estimate(cfg: dict[str, Any], rank: int = 0) -> dict[str, Any]:
    """Report the configured mesh and estimate time steps, DOFs, and memory."""
    summary = structure_from_config(cfg)
    layout = compute_mesh_layout(cfg, summary)
    degree = int(cfg.get("degree", 2))
    dim = 2
    field_dofs = restricted_field_dof_counts(summary, layout, degree)
    ndofs_est = sum(field_dofs.values())

    stype = str(cfg.get("solver_type", "explicit")).lower()
    requested_dt = float(cfg["dt"]) if cfg.get("dt") is not None else None
    duration_basis, warnings = duration_sizing_notes(cfg, summary)
    element_dt_limit = None
    dt_status = "not_applicable"
    if stype == "implicit":
        steps_per_period = float(cfg.get("steps_per_period") or DEFAULT_STEPS_PER_PERIOD)
        if requested_dt is not None:
            dt_est, dt_note = requested_dt, "user-requested maximum"
        else:
            dt_est = implicit_time_step(layout.mesh_frequency, steps_per_period)
            dt_note = f"accuracy, {steps_per_period:g} steps/period"
    else:
        cfl = float(cfg["CFL"])
        conservative_dt = explicit_time_step_estimate(cfg, summary, layout)
        if summary["has_acoustic"]:
            dt_status = "advisory_uncoupled_requires_assembled_check"
            dt_est = requested_dt or conservative_dt
            dt_note = "advisory uncoupled estimate; coupled assembly required"
            warnings.append(
                "Acoustic preflight omits interface coupling and does not certify explicit "
                "stability or a conservative step count. The assembled coupled operator "
                "determines the actual timestep."
            )
        else:
            element_dt_limit = conservative_dt / cfl
            dt_status = "certified_by_element_bound"
            if requested_dt is not None and requested_dt > element_dt_limit * (1 + 1e-12):
                dt_status = "requires_assembled_check"
                dt_est = conservative_dt
                dt_note = f"conservative resource estimate, element bound, CFL={cfl}"
                warnings.append(
                    f"Requested explicit dt={fmt_time(requested_dt)} exceeds the conservative "
                    f"element stability certificate ({fmt_time(element_dt_limit)}). Preflight "
                    "cannot certify this request; the assembled-operator check may reject it. "
                    f"Resource estimates use {fmt_time(conservative_dt)} before grid rounding. "
                    "Exceeding the conservative certificate alone does not prove instability."
                )
            elif requested_dt is not None:
                dt_est, dt_note = requested_dt, "user maximum, certified by element bound"
            else:
                dt_est, dt_note = conservative_dt, f"element bound, CFL={cfl}"
    dt_val, nsteps_est, stride = sampled_time_grid(cfg, layout, dt_est)

    vec_mem_mb = ndofs_est * 8 / 1024**2
    nnz_per_row = ((2 * degree + 1) ** 2) * dim
    mat_mem_mb = ndofs_est * nnz_per_row * (8 + 4) / 1024**2
    n_mats = (
        (5 if stype == "implicit" else 6)
        if summary["has_acoustic"]
        else (1 if stype == "implicit" else 2)
    )
    total_mem_est_mb = vec_mem_mb * 10 + mat_mem_mb * (n_mats + 1)
    n_samples = nsteps_est // stride + 1
    # Float64 is conservative for single precision and matches PETSc histories.
    observable = cfg.get("sensor_observable", "auto")
    if observable == "auto":
        _, candidates = sensor_region_candidates(cfg, summary)
        observable = (
            "pressure"
            if candidates and all(r["is_acoustic"] for r in candidates)
            else "displacement"
        )
    n_components = 1 if observable == "pressure" else 2
    history_size_mb = n_components * int(cfg["n_sensors"]) * n_samples * 8 / 1024**2
    timestamp_mem_mb = n_samples * 8 / 1024**2
    history_storage = str(cfg.get("history_storage", "memory"))
    history_disk_mb = 0.0
    if history_storage == "disk":
        history_mem_mb = 0.0  # measurement payload is mapped; OS page cache is separate
        history_disk_mb = 2 * history_size_mb  # rank-local and root gathered scratch histories
        chunk_mb = min(
            history_size_mb,
            max(HISTORY_CHUNK_BYTES, n_components * int(cfg["n_sensors"]) * 8) / 1024**2,
        )
        transfer_mem_mb = 2 * chunk_mb  # device ring and temporary host copy
        collection_mem_mb = 2 * chunk_mb  # bounded send and receive buffers
    else:
        history_mem_mb = history_size_mb
        transfer_mem_mb = min(history_mem_mb, 64.0)
        collection_mem_mb = 2 * history_mem_mb  # component splitting and root gathering
    solver_overhead_mb = 2 * mat_mem_mb  # workspace; LU/GAMG fill is problem-dependent
    total_mem_est_mb += (
        history_mem_mb
        + transfer_mem_mb
        + collection_mem_mb
        + solver_overhead_mb
        + 2 * timestamp_mem_mb
    )

    estimates = {
        "mesh_frequency": layout.mesh_frequency,
        "h_x": layout.h_x,
        "h_y_per_region": layout.h_y_per_region,
        "h_min": layout.h_min,
        "nx": layout.nx,
        "ny_per_region": layout.ny_per_region,
        "total_elements": layout.nx * layout.ny,
        "ndofs_est": ndofs_est,
        "field_dofs_est": field_dofs,
        "sensor_components_est": n_components,
        "duration": layout.duration,
        "duration_basis": duration_basis,
        "warnings": warnings,
        "dt_requested": requested_dt,
        "dt_element_limit": element_dt_limit,
        "dt_stability_status": dt_status,
        "dt_est": dt_val,
        "nsteps_est": nsteps_est,
        "sample_stride_est": stride,
        "vec_mem_mb": vec_mem_mb,
        "mat_mem_mb": mat_mem_mb,
        "total_mem_est_mb": total_mem_est_mb,
        "n_samples_est": n_samples,
        "history_mem_mb": history_mem_mb,
        "history_size_mb": history_size_mb,
        "history_storage": history_storage,
        "history_disk_mb": history_disk_mb,
        "archive_payload_mb": history_size_mb + timestamp_mem_mb,
        "timestamp_mem_mb": timestamp_mem_mb,
        "transfer_mem_mb": transfer_mem_mb,
        "collection_mem_mb": collection_mem_mb,
        "solver_overhead_mb": solver_overhead_mb,
    }

    if rank == 0:
        mem_str = (
            f"{total_mem_est_mb / 1024:.2f} GB"
            if total_mem_est_mb >= 1024
            else f"{total_mem_est_mb:.1f} MB"
        )
        source_type = str(cfg.get("source_type") or "ricker")
        lines = [
            f"PREFLIGHT NUMERICAL ESTIMATE ({stype.upper()})",
            f"Excitation:            {describe_source(source_type, cfg.get('n_cycles'))}, "
            f"f₀ = {fmt_frequency(float(cfg['f_center']))}",
            f"Analysis max frequency: {fmt_frequency(layout.mesh_frequency)}",
            f"Element sizes:         dx = {fmt_length(layout.h_x)}, "
            f"dy by region = {', '.join(fmt_length(h) for h in layout.h_y_per_region)}",
            f"Grid resolution:       {layout.nx} × {layout.ny} Q{degree} elements "
            f"({layout.nx * layout.ny:,} total; user-supplied)",
            f"Estimated global DOFs: {ndofs_est:,}",
            f"Simulation duration:   {fmt_time(layout.duration)}",
            f"Duration basis:        {duration_basis}",
            f"Time step:             ~{fmt_time(dt_val)} ({dt_note})",
            f"Estimated step count:  ~{nsteps_est:,} steps",
            f"Estimated memory RAM:  ~{mem_str} (single state vector: {vec_mem_mb:.1f} MB)",
            f"Recorded histories:   ~{history_size_mb:.1f} MB ({n_samples:,} samples, {history_storage})",
            "RAM includes history buffers, timestamps and workspace; factorization fill may exceed it.",
        ]
        if history_storage == "disk":
            lines.extend(
                [
                    f"History scratch disk: ~{history_disk_mb:.1f} MB, plus the output archive.",
                    "Disk histories bound measurement buffers; OS page cache and FFT RAM are excluded.",
                ]
            )
        if stype == "explicit":
            lines.append("Explicit Δt is finalized from the assembled operator bound.")
        if _free_isotropic_plate(summary) is None and summary["structure_mode"] != "halfspace":
            lines.append(
                "Validate mesh convergence and arrival coverage for the guided modes of interest."
            )
        lines.insert(1, "─" * max(len(line) for line in lines))
        log(format_banner_box(lines, min_width=64))
        for message in warnings:
            warn(message)
    return estimates
