"""Structural geometry, constitutive models, mesh layout, and preflight estimation.

Supports:
  - Isotropic, anisotropic, Kelvin-Voigt viscoelastic, and regularized fluids
  - Single plates, multilayer laminates, semi-infinite elastic halfspaces, and
    plate-substrate systems
  - A shared mesh/time layout used by both the preflight estimator and the solver
"""

from __future__ import annotations

import difflib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from guidedwavefem.numerics import RK4_STABILITY_RADIUS, time_grid, validate_tensor
from guidedwavefem.reporting import format_banner_box, log
from guidedwavefem.sources import describe_source, source_duration
from guidedwavefem.units import fmt_frequency, fmt_length, fmt_time

STRUCTURE_MODES = (
    "1_layer",
    "N_layers",
    "halfspace",
    "plate_on_halfspace",
    "plate_in_halfspaces",
)
MATERIAL_MODELS = ("isotropic", "anisotropic", "viscoelastic", "fluid")

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
        "min_elems_y",
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
    "fluid": frozenset({"E", "nu", "fluid_shear_reg", *_FLUID_SPEED_KEYS, *_FLUID_BULK_KEYS}),
}
_CONSTITUTIVE_FIELDS = frozenset().union(*_MODEL_FIELDS.values())

DEFAULT_FLUID_SHEAR_REG = 1e-8

#: Default number of implicit (Newmark) steps per period at the mesh-sizing
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
    try:
        value = float(item[key])
    except (TypeError, ValueError):
        raise ValueError(
            f"Region '{name}' field '{key}' must be a number, got {item[key]!r}"
        ) from None
    if not np.isfinite(value):
        raise ValueError(f"Region '{name}' field '{key}' must be finite, got {value}")
    return value


def _region_voigt(item: Mapping[str, Any], key: str, name: str) -> list[float]:
    if key not in item or item[key] is None:
        raise ValueError(f"Region '{name}' is missing required field '{key}'")
    try:
        values = [float(v) for v in item[key]]
    except (TypeError, ValueError):
        raise ValueError(
            f"Region '{name}' field '{key}' must be a list of six numbers "
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
    s = str(mode_str).lower().replace("-", "_").replace(" ", "_").strip()
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
        "min_elems_y": None,
        "is_fluid": model == "fluid",
    }
    if item.get("min_elems_y") is not None:
        value = item["min_elems_y"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer))
            or (int(value) != value or value < 1)
        ):
            raise ValueError(f"Region '{name}' min_elems_y must be a positive integer")
        entry["min_elems_y"] = int(value)

    if model == "fluid":
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

        fluid_shear_reg = float(item.get("fluid_shear_reg", DEFAULT_FLUID_SHEAR_REG))
        if not np.isfinite(fluid_shear_reg) or not 0 < fluid_shear_reg <= 1e-2:
            raise ValueError(
                f"Fluid '{name}' fluid_shear_reg must be in (0, 0.01], got {fluid_shear_reg}"
            )
        mu = fluid_shear_reg * K_val
        lmbda = K_val - (2.0 / 3.0) * mu
        nu_eff = lmbda / (2.0 * (lmbda + mu))
        entry.update(
            c_fluid=c_sound,
            K=K_val,
            fluid_shear_reg=fluid_shear_reg,
            E=2.0 * mu * (1.0 + nu_eff),
            nu=nu_eff,
            mu=mu,
            lmbda=lmbda,
            c_L=float(np.sqrt((lmbda + 2 * mu) / rho)),
            c_T=float(np.sqrt(mu / rho)),
            c_physical_min=c_sound,
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
    absorb_depth = float(cfg.get("absorb_depth", default_frac * h))
    if not np.isfinite(absorb_depth) or absorb_depth <= 0 or absorb_depth > h:
        raise ValueError(f"{label} absorb_depth must be in (0, {h}], got {absorb_depth}")
    return absorb_depth


def _speed_summary(regions: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    solids = [r for r in regions if not r.get("is_fluid", False)] or regions
    return {
        f"c_T_{prefix}min": min(r["c_T"] for r in solids),
        f"c_T_{prefix}max": max(r["c_T"] for r in solids),
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
            "c_fluid_min": norm_hs["c_L"] if norm_hs["is_fluid"] else None,
            "c_R": c_R,
            "c_T_plate_min": norm_hs["c_T"],
            "c_T_plate_max": norm_hs["c_T"],
            "c_L_plate_min": norm_hs["c_L"],
            "c_L_plate_max": norm_hs["c_L"],
            "c_L_plate_top": norm_hs["c_L"],
            "c_T_plate_top": norm_hs["c_T"],
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
        "c_fluid_min": min((r["c_L"] for r in all_regions if r["is_fluid"]), default=None),
        **_speed_summary(norm_plate_layers, "plate_"),
        "c_L_plate_top": top_plate["c_L"],
        "c_T_plate_top": top_plate["c_T"],
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
@dataclass(frozen=True)
class MeshLayout:
    """Structured-mesh sizing derived from a configuration and structure."""

    mesh_frequency: float
    lambda_min: float
    h_target: float
    plate_length: float
    nx: int
    ny_per_region: list[int]
    h_min: float
    duration: float

    @property
    def ny(self) -> int:
        return sum(self.ny_per_region)


def compute_mesh_layout(cfg: Mapping[str, Any], summary: Mapping[str, Any]) -> MeshLayout:
    """Mesh resolution from the shortest physical wavelength at ``max_frequency``.

    ``elems_per_wavelength`` counts *elements*, independent of polynomial degree
    (a Q2 element carries two node spacings).
    """
    f_center = float(cfg["f_center"])
    mesh_frequency = float(cfg.get("max_frequency") or 3.0 * f_center)
    lambda_min = summary["c_wave_min"] / mesh_frequency
    h_target = lambda_min / int(cfg["elems_per_wavelength"])
    plate_length = 2.0 * float(cfg["absorb_length"]) + float(cfg["sensing_length"])
    nx = max(int(np.ceil(plate_length / h_target)), int(cfg["min_elems_x"]))
    regions = summary["all_regions"]
    ny_per_region = [
        max(
            int(np.ceil(r["thickness"] / h_target)), r.get("min_elems_y") or int(cfg["min_elems_y"])
        )
        for r in regions
    ]
    h_min = min(
        plate_length / nx,
        min(r["thickness"] / n for r, n in zip(regions, ny_per_region, strict=True)),
    )
    speed_for_T = reference_wave_speed(summary)
    pulse = source_duration(str(cfg.get("source_type") or "ricker"), f_center, cfg.get("n_cycles"))
    duration = float(
        cfg.get("duration")
        or float(cfg["sensing_length"]) / speed_for_T * float(cfg["time_factor"]) + pulse
    )
    return MeshLayout(
        mesh_frequency=mesh_frequency,
        lambda_min=lambda_min,
        h_target=h_target,
        plate_length=plate_length,
        nx=nx,
        ny_per_region=ny_per_region,
        h_min=h_min,
        duration=duration,
    )


def reference_wave_speed(summary: Mapping[str, Any]) -> float:
    """Slowest relevant wave speed: ``c_R`` for a halfspace, else the slowest physical speed.

    Sets the default duration and the scale of the automatic source/sensor layout.
    """
    return float(summary.get("c_R") or summary["c_wave_min"])


def implicit_time_step(mesh_frequency: float, steps_per_period: float) -> float:
    """Accuracy-driven Newmark step ``1 / (steps_per_period · f_mesh)`` (shared with preflight).

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
    """Upper bounds of the assembled Gershgorin bounds ``B_K`` and ``B_C`` before meshing.

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
        rho = float(region["rho"])
        mass_rows = np.repeat(rho * np.einsum("q,qi,qj->i", weights, shape, shape), 2)
        abs_mass_rows = np.repeat(
            rho * np.einsum("q,qi,qj->i", weights, np.abs(shape), np.abs(shape)), 2
        )
        stiffness = _voigt_stiffness(region)
        k_rows = _tensor_row_sums(weights, strain, stiffness, absolute_integrand=False)
        bound_k = max(bound_k, float(np.max(k_rows / mass_rows)))
        indicator = 2.0 if region.get("is_halfspace") else 1.0
        abs_k_rows = _tensor_row_sums(weights, strain, stiffness, absolute_integrand=True)
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

    Never larger than the solver's step (the solver uses the sharper assembled
    bounds), so the preflight step count is an upper estimate.
    """
    bound_k, bound_c = element_operator_bounds(cfg, summary, layout)
    return float(cfg.get("CFL", 0.5)) * RK4_STABILITY_RADIUS / max(np.sqrt(bound_k), bound_c)


def preflight_estimate(cfg: dict[str, Any], rank: int = 0) -> dict[str, Any]:
    """Lightweight estimates for mesh sizing, DOFs, time steps, and memory."""
    summary = structure_from_config(cfg)
    layout = compute_mesh_layout(cfg, summary)
    degree = int(cfg.get("degree", 2))
    dim = 2
    ndofs_est = dim * (degree * layout.nx + 1) * (degree * layout.ny + 1)

    cfl = float(cfg.get("CFL", 0.5))
    stype = str(cfg.get("solver_type", "explicit")).lower()
    if cfg.get("dt"):
        dt_est, dt_note = float(cfg["dt"]), "user-requested maximum"
    elif stype == "implicit":
        steps_per_period = float(cfg.get("steps_per_period") or DEFAULT_STEPS_PER_PERIOD)
        dt_est = implicit_time_step(layout.mesh_frequency, steps_per_period)
        dt_note = f"accuracy, {steps_per_period:g} steps/period"
    else:
        dt_est = explicit_time_step_estimate(cfg, summary, layout)
        dt_note = f"element bound, CFL={cfl}"
    dt_val, nsteps_est = time_grid(layout.duration, dt_est)

    vec_mem_mb = ndofs_est * 8 / 1024**2
    nnz_per_row = ((2 * degree + 1) ** 2) * dim
    mat_mem_mb = ndofs_est * nnz_per_row * (8 + 4) / 1024**2
    n_mats = 1 if stype == "implicit" else 2
    total_mem_est_mb = vec_mem_mb * 10 + mat_mem_mb * (n_mats + 1)

    estimates = {
        "mesh_frequency": layout.mesh_frequency,
        "lambda_min": layout.lambda_min,
        "h_target": layout.h_target,
        "h_min": layout.h_min,
        "nx": layout.nx,
        "ny_per_region": layout.ny_per_region,
        "total_elements": layout.nx * layout.ny,
        "ndofs_est": ndofs_est,
        "duration": layout.duration,
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
        source_type = str(cfg.get("source_type") or "ricker")
        lines = [
            f"PREFLIGHT NUMERICAL ESTIMATE ({stype.upper()})",
            f"Excitation:            {describe_source(source_type, cfg.get('n_cycles'))}, "
            f"f₀ = {fmt_frequency(float(cfg['f_center']))}",
            f"Mesh frequency:        {fmt_frequency(layout.mesh_frequency)} "
            f"(λ_min = {fmt_length(layout.lambda_min)})",
            f"Target element size h: {fmt_length(layout.h_target)} "
            f"(actual min h = {fmt_length(layout.h_min)})",
            f"Grid resolution:       {layout.nx} × {layout.ny} Q{degree} elements "
            f"({layout.nx * layout.ny:,} total)",
            f"Estimated global DOFs: {ndofs_est:,}",
            f"Simulation duration:   {fmt_time(layout.duration)}",
            f"Time step:             ~{fmt_time(dt_val)} ({dt_note})",
            f"Estimated step count:  ~{nsteps_est:,} steps",
            f"Estimated memory RAM:  ~{mem_str} (single state vector: {vec_mem_mb:.1f} MB)",
        ]
        if stype == "explicit":
            lines.append("Explicit Δt is finalized from the assembled operator bound.")
        lines.insert(1, "─" * max(len(line) for line in lines))
        log(format_banner_box(lines, min_width=64))
    return estimates
