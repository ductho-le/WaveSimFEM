"""Structural geometry, material constitutive modeling, and preflight estimation.

Supports:
  - Isotropic, anisotropic, Kelvin-Voigt viscoelastic, and regularized acoustic fluids
  - Single plates, multilayer laminates, semi-infinite elastic halfspaces, and plate-substrate systems
  - Preflight estimation for computational sizing and memory footprints
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from guidedwavefem.numerics import time_grid, validate_tensor


def log(msg: str = "") -> None:
    """Print message with immediate flush for real-time progress logging."""
    print(msg, flush=True)


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
    valid = [r.real for r in roots if abs(r.imag) < 1e-10 and 0.0 < r.real < 1.0]
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
    """Validate and precompute material parameters for a single subdomain region."""
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
            if not np.isfinite([E, nu]).all() or E <= 0 or not -1 < nu < 0.5:
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
        c_L = float(np.sqrt((lmbda + 2 * mu) / rho))
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
        scale = np.diag([1.0, 1.0, np.sqrt(2.0)])
        eig = np.linalg.eigvalsh(scale @ C_mat @ scale)
        c_L = float(np.sqrt(eig[-1] / rho))
        c_T = float(np.sqrt(eig[0] / (2 * rho)))
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
    """Normalize full structure: plates, pristine halfspace, or plate-halfspace systems."""
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
    """Format lines inside a double-bordered Unicode box with pixel-perfect right-wall alignment."""
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
    """Format and print a professional structure summary table on rank 0."""
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
    """Compute lightweight preflight estimates for mesh sizing, DOFs, time steps, and memory."""
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
    T_total = float(
        cfg.get("duration")
        or (float(cfg["sensing_length"]) / speed_for_T * time_factor)
    )

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
    vec_mem_mb = (ndofs_est * 8) / (1024**2)
    # Sparse matrix CSR: nonzeros per row for Q_degree in 2D is approximately (2*degree+1)^2 * dim
    nnz_per_row = ((2 * degree + 1) ** 2) * dim
    mat_mem_mb = (ndofs_est * nnz_per_row * (8 + 4)) / (1024**2)
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
        log(_format_banner_box(lines, min_width=64))

    return estimates
