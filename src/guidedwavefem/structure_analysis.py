"""Structure-specific interpretation and reference curves for receiver reports.

References are physical context, not automatic identifications of measured peaks.
Free-plate Rayleigh--Lamb curves reuse :mod:`guidedwavefem.lamb`; layered and
fluid-loaded structures require independent dispersion solutions. Source URLs are
included in each serializable profile so exported reports retain their provenance.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import yaml

from guidedwavefem.materials import (
    a0_wavenumber,
    canonicalize_mode,
    compute_rayleigh_speed,
    structure_from_config,
)

_SOURCES = {
    "lamb": {
        "title": "Lamb (1917), On waves in an elastic plate",
        "url": "https://doi.org/10.1098/rspa.1917.0008",
    },
    "surface": {
        "title": "USGS: Body waves and surface waves",
        "url": "https://earthquake.usgs.gov/earthquakes/events/1906calif/18april/earthwaves.php",
    },
    "fluid": {
        "title": "Carcione et al. (2018), Waves at fluid-solid interfaces",
        "url": "https://doi.org/10.1093/gji/ggy262",
    },
    "anisotropic": {
        "title": "COMSOL: Christoffel equation and anisotropic wave speeds",
        "url": "https://doc.comsol.com/6.4/doc/com.comsol.help.sme/sme_ug_theory.06.026.html",
    },
}


def _scalar(data: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if key not in data:
        return default
    value = np.asarray(data[key])
    if value.size != 1:
        return default
    value = value.item()
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _number(data: Mapping[str, Any], key: str, *, positive: bool = False) -> float | None:
    value = _scalar(data, key)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) and (not positive or result > 0) else None


def _configuration(data: Mapping[str, Any]) -> dict[str, Any] | None:
    """Read only structure fields, accepting resolved and original sectioned YAML.

    Unrelated schema additions in future archives must not break interpretation.
    An invalid present configuration raises; silently guessing its physics from
    legacy scalar metadata would conceal a corrupt or incompatible provenance.
    """
    raw = _scalar(data, "simulation_config_yaml")
    if raw is None or raw == "":
        return None
    try:
        cfg = yaml.safe_load(raw)
    except (yaml.YAMLError, TypeError) as exc:
        raise ValueError("Invalid archived simulation_config_yaml") from exc
    if not isinstance(cfg, Mapping):
        raise ValueError("Archived simulation_config_yaml must contain a mapping")
    section = cfg.get("structure", cfg)
    if not isinstance(section, Mapping):
        raise ValueError("Archived structure configuration must contain a mapping")
    flat = dict(section)
    if "mode" in flat:
        flat["structure_mode"] = flat["mode"]
    for alias, key in (("substrate", "bottom_halfspace"), ("superstrate", "top_halfspace")):
        if alias in flat and key not in flat:
            flat[key] = flat[alias]
    return flat


def _region(region: Mapping[str, Any]) -> dict[str, Any]:
    model = region["material_model"]
    result = {
        "name": str(region["name"]),
        "role": str(region["role"]),
        "material_model": model,
        "thickness_m": float(region["thickness"]),
        "y_bot_m": float(region["y_bot"]),
        "y_top_m": float(region["y_top"]),
        "density_kg_m3": float(region["rho"]),
        "is_halfspace": bool(region["is_halfspace"]),
        "halfspace_type": region["halfspace_type"],
        "is_fluid": bool(region["is_fluid"]),
        "absorb_depth_m": _number(region, "absorb_depth", positive=True),
        "c_L_m_s": float(region["c_L"]) if model == "isotropic" else None,
        "c_T_m_s": float(region["c_T"]) if model == "isotropic" else None,
        "c_fluid_m_s": _number(region, "c_fluid", positive=True),
        "c_R_m_s": None,
    }
    if model == "isotropic":
        result["c_R_m_s"] = compute_rayleigh_speed(region["c_T"], c_L=region["c_L"])
    elif model == "fluid":
        # The artificial shear branch is not an acoustic propagation reference.
        result["numerical_shear_speed_m_s"] = float(region["c_T"])
    else:
        # In the solver these are Kelvin-tensor sizing bounds, not P/S speeds.
        result["sizing_c_L_m_s"] = float(region["c_L"])
        result["sizing_c_T_m_s"] = float(region["c_T"])
        c11, _, c16, _, _, c66 = region["C"]
        speeds = np.sqrt(np.linalg.eigvalsh([[c11, c16], [c16, c66]]) / region["rho"])
        result["axial_qSV_m_s"], result["axial_qP_m_s"] = map(float, speeds)
    return result


def _legacy_regions(data: Mapping[str, Any], mode: str) -> list[dict[str, Any]]:
    """Recover geometry without assigning the top layer's speeds to every layer."""

    def vector(key: str) -> list[Any]:
        return np.asarray(data[key]).reshape(-1).tolist() if key in data else []

    names, thicknesses, models, roles = [
        vector("all_regions_" + key) for key in ("names", "thicknesses", "models", "roles")
    ]
    if not names:
        names, thicknesses, models = [
            vector("layers_" + key) for key in ("names", "thicknesses", "models")
        ]
    if not names:
        depth = _number(data, "total_depth", positive=True)
        names = ["Archived medium" if mode == "halfspace" else "Archived plate"]
        thicknesses = [depth if mode == "halfspace" else _number(data, "plate_thickness")]
        models = [str(_scalar(data, "material_model", "unknown"))]
    if len(thicknesses) != len(names):
        thicknesses = [None] * len(names)
    y = _number(data, "y_min")
    if y is None and mode in ("1_layer", "N_layers"):
        y = 0.0
    regions = []
    for i, name in enumerate(names):
        model = str(models[i]) if i < len(models) else "unknown"
        role = str(roles[i]) if i < len(roles) else "Unknown"
        halfspace = mode == "halfspace" or role in ("Halfspace", "Substrate", "Superstrate")
        h = _number({"h": thicknesses[i]}, "h", positive=True)
        top = y + h if y is not None and h is not None else None
        regions.append(
            {
                "name": str(name),
                "role": role,
                "material_model": model,
                "thickness_m": h,
                "y_bot_m": y,
                "y_top_m": top,
                "density_kg_m3": None,
                "is_halfspace": halfspace,
                "halfspace_type": "top"
                if role == "Superstrate"
                else "bottom"
                if halfspace
                else None,
                "is_fluid": model == "fluid",
                "absorb_depth_m": None,
                "c_L_m_s": None,
                "c_T_m_s": None,
                "c_fluid_m_s": None,
                "c_R_m_s": None,
            }
        )
        y = top
    # Scalar speeds refer to the top finite layer, or the sole halfspace.
    target = next((r for r in reversed(regions) if not r["is_halfspace"]), regions[-1])
    if _scalar(data, "is_fluid_top", False) or target["is_fluid"]:
        target.update(is_fluid=True, material_model="fluid")
        target["c_fluid_m_s"] = _number(data, "c_L", positive=True)
    elif target["material_model"] == "isotropic":
        target["c_L_m_s"] = _number(data, "c_L", positive=True)
        target["c_T_m_s"] = _number(data, "c_T", positive=True)
        if mode == "halfspace":
            target["c_R_m_s"] = _number(data, "c_R", positive=True)
    target["density_kg_m3"] = _number(data, "rho", positive=True)
    return regions


def _identity(mode: str, regions: list[dict[str, Any]], exact: bool) -> tuple[str, str, list[str]]:
    models = {r["material_model"] for r in regions}
    fluids = any(r["is_fluid"] for r in regions)
    if models == {"fluid"}:
        if mode == "halfspace":
            return "fluid_halfspace", "Acoustic halfspace", ["Acoustic P waves"]
        return "acoustic_waveguide", "Layered acoustic waveguide", ["Acoustic guided waves"]
    if mode == "halfspace":
        if models == {"isotropic"}:
            return (
                "solid_halfspace",
                "Solid halfspace · Rayleigh and bulk waves",
                ["Rayleigh", "P", "SV"],
            )
        if models == {"viscoelastic"}:
            return (
                "viscoelastic_halfspace",
                "Viscoelastic halfspace · surface and bulk waves",
                ["Surface-wave candidates", "Viscoelastic P-SV waves"],
            )
        if "anisotropic" not in models:
            return (
                "unknown_halfspace",
                "Halfspace · material model not established",
                ["Unclassified measured waves"],
            )
        return (
            "anisotropic_halfspace",
            "Solid halfspace · surface and quasi-bulk waves",
            ["Surface-wave candidates", "Quasi-P", "Quasi-SV"],
        )
    if any(r["is_halfspace"] and r["is_fluid"] for r in regions):
        return (
            "fluid_loaded_plate",
            "Fluid-loaded plate · guided and interface waves",
            ["Fluid-loaded guided waves", "Interface-wave candidates", "Acoustic radiation"],
        )
    if fluids:
        return (
            "solid_fluid_waveguide",
            "Solid-fluid layers · coupled guided waves",
            ["Coupled guided P-SV waves", "Acoustic waves", "Interface-wave candidates"],
        )
    if mode == "plate_on_halfspace":
        return (
            "layered_halfspace",
            "Layered halfspace · dispersive surface waves",
            ["Dispersive Rayleigh-type waves", "P-SV body waves"],
        )
    if mode == "plate_in_halfspaces":
        return (
            "embedded_plate",
            "Embedded solid layers · guided and radiating waves",
            ["Guided P-SV candidates", "Bulk radiation", "Interface-wave candidates"],
        )
    if mode not in ("1_layer", "N_layers"):
        return "unknown", "Structure not established by archive", ["Unclassified measured waves"]
    if exact:
        return (
            "free_isotropic_plate",
            "Free isotropic plate · Lamb waves",
            ["A0", "S0", "Higher Lamb modes if excited"],
        )
    if len(regions) == 1 and models == {"viscoelastic"}:
        return (
            "viscoelastic_plate",
            "Viscoelastic plate · guided P-SV waves",
            ["Viscoelastic guided waves"],
        )
    if len(regions) == 1 and models.intersection({"anisotropic", "viscoelastic"}):
        return (
            "anisotropic_plate",
            "Anisotropic plate · guided P-SV waves",
            ["Anisotropic guided waves"],
        )
    return "layered_plate", "Layered plate · guided P-SV waves", ["Guided / quasi-Lamb waves"]


def structure_profile(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return JSON-safe physics context from an NPZ archive or metadata mapping.

    Exact Lamb applicability requires archived material and boundary provenance.
    Legacy metadata can provide constituent reference lines, but never authorizes
    free-plate mode labels or anisotropic sizing bounds as measured wave speeds.
    """
    cfg = _configuration(data)
    if cfg is not None:
        summary = structure_from_config(cfg)
        mode = summary["structure_mode"]
        regions = [_region(r) for r in summary["all_regions"]]
        thickness = float(summary["H_plate"])
    else:
        try:
            raw_mode = _scalar(data, "structure_mode", "unknown")
            mode = canonicalize_mode(raw_mode) if raw_mode != "unknown" else "unknown"
        except ValueError:
            mode = "unknown"
        regions = _legacy_regions(data, mode)
        thickness = _number(data, "plate_thickness", positive=True)
    exact = bool(
        cfg is not None
        and mode in ("1_layer", "N_layers")
        and len(regions) == 1
        and regions[0]["material_model"] == "isotropic"
    )
    family, title, waves = _identity(mode, regions, exact)
    interpretations = [
        "Spectral ridges and component ratios describe measured motion; mode identity needs a matching dispersion solution."
    ]
    exclusions = [
        "Plane strain resolves x-y motion only; SH and Love waves are outside this model."
    ]
    if not exact:
        exclusions.append(
            "Free isotropic A0/S0 curves and free-plate modal labels do not apply to this profile."
        )
    if cfg is None:
        exclusions.append(
            "Full archived configuration is unavailable; region properties and boundary conditions are incomplete."
        )
    if family == "free_isotropic_plate":
        interpretations += [
            "Compare A0/S0 phase-velocity dispersion with free-plate theory; packet arrivals follow group velocity.",
            "A0/S0 reference curves do not account for higher modes or prove modal separation.",
        ]
    elif family == "solid_halfspace":
        interpretations += [
            "Compare the surface branch with the homogeneous Rayleigh speed and distinguish earlier P/SV arrivals.",
            "A Rayleigh interpretation requires surface localization; check truncation-depth convergence.",
        ]
    elif family == "layered_halfspace":
        interpretations += [
            "Inspect frequency-dependent phase velocity and x/y polarization together; multiple Rayleigh-type branches can coexist.",
            "Constituent Rayleigh speeds are comparison values, not the dispersion solution or guaranteed bounds of the layered system.",
        ]
    elif family == "fluid_loaded_plate":
        interpretations += [
            "Fluid sound-speed lines mark the onset of propagating acoustic radiation for a real horizontal phase velocity above c_f.",
            "Below c_f the corresponding fluid field is evanescent; this alone does not identify a Scholte or guided mode.",
        ]
    elif family in ("fluid_halfspace", "acoustic_waveguide"):
        interpretations += [
            "Interpret longitudinal acoustic propagation using the physical sound speed; displacement components depend on propagation direction."
        ]
        exclusions.append(
            "A fluid has no physical shear or Rayleigh branch; its small numerical shear stiffness is regularization."
        )
    else:
        interpretations += [
            "Track coupled guided branches and dispersion without assigning homogeneous-plate A/S symmetry labels."
        ]
    if any(r["is_fluid"] for r in regions) and not all(r["is_fluid"] for r in regions):
        exclusions.append(
            "Regularized-fluid displacement coupling enforces tangential continuity; quantitative interface-wave and leakage predictions require independent validation."
        )
    if any(r["material_model"] == "viscoelastic" for r in regions):
        interpretations.append(
            "Axial quasi-bulk speeds use elastic C only; Kelvin-Voigt Cv produces frequency-dependent dispersion and attenuation."
        )
        exclusions.append(
            "Receiver amplitude decay alone does not isolate intrinsic material attenuation from spreading, leakage or modal interference."
        )
    references = []

    def reference(region: dict[str, Any], key: str, label: str, kind: str, applies: str) -> None:
        speed = region.get(key)
        if speed is not None and np.isfinite(speed) and speed > 0:
            references.append(
                {
                    "label": label,
                    "kind": kind,
                    "speed_m_s": float(speed),
                    "region": region["name"],
                    "applicability": applies,
                }
            )

    for region in regions:
        suffix = f" · {region['name']}" if len(regions) > 1 else ""
        if region["is_fluid"]:
            radiation = region["is_halfspace"] and family == "fluid_loaded_plate"
            reference(
                region,
                "c_fluid_m_s",
                "Acoustic c_f" + suffix,
                "acoustic_radiation" if radiation else "acoustic",
                "Acoustic radiation threshold in this exterior fluid."
                if radiation
                else "Physical longitudinal sound speed in this constituent fluid.",
            )
        elif region["material_model"] == "isotropic":
            for key, label in (("c_L_m_s", "Bulk P"), ("c_T_m_s", "Bulk SV")):
                reference(
                    region,
                    key,
                    label + suffix,
                    "bulk",
                    "Bulk constituent speed, not a guided-mode dispersion curve.",
                )
            if family == "solid_halfspace":
                reference(
                    region,
                    "c_R_m_s",
                    "Rayleigh",
                    "rayleigh",
                    "Homogeneous isotropic traction-free halfspace; finite-depth effects are excluded.",
                )
            elif family == "layered_halfspace":
                reference(
                    region,
                    "c_R_m_s",
                    "Constituent Rayleigh" + suffix,
                    "constituent_rayleigh",
                    "Rayleigh speed of this material in isolation, not a layered-system branch.",
                )
        else:
            for key, label in (
                ("axial_qP_m_s", "Axial qP (elastic)"),
                ("axial_qSV_m_s", "Axial qSV (elastic)"),
            ):
                reference(
                    region,
                    key,
                    label + suffix,
                    "quasi_bulk",
                    "Christoffel speed along x from elastic C; not a guided mode or a viscoelastic phase speed.",
                )
    sensor_y, load_y = _number(data, "sensor_y"), _number(data, "load_y")
    sensor_regions = [
        r["name"]
        for r in regions
        if sensor_y is not None
        and r["y_bot_m"] is not None
        and r["y_top_m"] is not None
        and r["y_bot_m"] <= sensor_y <= r["y_top_m"]
    ]
    sources = [_SOURCES["surface"]]
    if exact:
        sources.append(_SOURCES["lamb"])
    if any(r["is_fluid"] for r in regions):
        sources.append(_SOURCES["fluid"])
    if any(r["material_model"] in ("anisotropic", "viscoelastic") for r in regions):
        sources.append(_SOURCES["anisotropic"])
    return {
        "family": family,
        "title": title,
        "structure_mode": mode,
        "provenance": "archived_config" if cfg is not None else "legacy",
        "regions": regions,
        "wave_families": waves,
        "interpretation": interpretations,
        "exclusions": exclusions,
        "references": references,
        "sources": sources,
        "lamb_applicable": exact,
        "plate_thickness_m": thickness,
        "preferred_component": "auto" if all(r["is_fluid"] for r in regions) else "y",
        "sensor_y_m": sensor_y,
        "load_y_m": load_y,
        "sensor_regions": sensor_regions,
        "homogeneous_plate": {
            "thickness_m": thickness,
            "c_L_m_s": regions[0]["c_L_m_s"],
            "c_T_m_s": regions[0]["c_T_m_s"],
        }
        if exact
        else None,
    }


def reference_dispersion(profile: Mapping[str, Any], frequencies_hz: Any) -> list[dict[str, Any]]:
    """Return JSON-safe constant-speed guides and applicable A0/S0 theory curves.

    Zero frequency is excluded because phase velocity is undefined there. The
    returned wavenumber is positive; callers choose propagation-direction signs.
    """
    f = np.asarray(frequencies_hz, dtype=float)
    if f.ndim != 1 or not np.isfinite(f).all() or np.any(f < 0):
        raise ValueError("Reference frequencies must be a finite nonnegative vector")
    f = np.unique(f[f > 0])
    result = []
    for ref in profile["references"]:
        speed = ref["speed_m_s"]
        result.append(
            {
                **ref,
                "frequency_hz": f.tolist(),
                "wavenumber_rad_m": (2 * np.pi * f / speed).tolist(),
                "phase_velocity_m_s": np.full(len(f), speed).tolist(),
            }
        )
    if profile["lamb_applicable"]:
        from guidedwavefem.lamb import s0_wavenumber

        plate = profile["homogeneous_plate"]
        args = (plate["thickness_m"], plate["c_L_m_s"], plate["c_T_m_s"])
        for label, root in (("A0", a0_wavenumber), ("S0", s0_wavenumber)):
            k = np.array([root(float(freq), *args) for freq in f])
            result.append(
                {
                    "label": label,
                    "kind": "lamb",
                    "frequency_hz": f.tolist(),
                    "wavenumber_rad_m": k.tolist(),
                    "phase_velocity_m_s": (2 * np.pi * f / k).tolist(),
                    "applicability": "Free homogeneous isotropic elastic plate; fundamental branch only.",
                }
            )
    return result
